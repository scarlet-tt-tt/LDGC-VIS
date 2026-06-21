import argparse
import json
import os
import os.path as osp
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn

import evaluate as tum_protocol
import LEM_SFM.evaluate as a_protocol
from reloc3r.datasets import get_data_loader
from reloc3r.reloc3r_relpose import Reloc3rRelpose


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    E, mask = cv2.findEssentialMat(kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf, method=cv2.RANSAC)
    if E is None:
        return None
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


class FeaturePoseMatcher(nn.Module):
    def _pose_from_matches(self, kpts0, kpts1, k_row, device, pose_thresh=1.0, min_matches=8):
        if len(kpts0) < min_matches:
            return torch.eye(3, dtype=torch.float32, device=device), torch.zeros(3, dtype=torch.float32, device=device)
        fx, fy, cx, cy = k_row.detach().cpu().numpy()
        k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        ret = estimate_pose(kpts0, kpts1, k_mat, k_mat, pose_thresh)
        if ret is None:
            return torch.eye(3, dtype=torch.float32, device=device), torch.zeros(3, dtype=torch.float32, device=device)
        r, t, _ = ret
        return torch.tensor(r, dtype=torch.float32, device=device), torch.tensor(t.reshape(-1), dtype=torch.float32, device=device)


class RoMaPairMatcher(FeaturePoseMatcher):
    def __init__(self, sample_thresh=0.05, pose_thresh=1.0, min_matches=8):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pose_thresh = pose_thresh
        self.min_matches = min_matches
        roma_ckpt = osp.expanduser("~/.cache/torch/hub/checkpoints/roma_outdoor.pth")
        dinov2_ckpt = osp.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth")
        from PIL import Image
        from romatch.models import roma_outdoor
        self._image_cls = Image
        if not osp.isfile(roma_ckpt):
            raise FileNotFoundError(f"Missing RoMa checkpoint: {roma_ckpt}")
        if not osp.isfile(dinov2_ckpt):
            raise FileNotFoundError(f"Missing DINOv2 checkpoint: {dinov2_ckpt}")
        weights = torch.load(roma_ckpt, map_location=self.device)
        dinov2_weights = torch.load(dinov2_ckpt, map_location="cpu")
        self.matcher = roma_outdoor(
            device=self.device,
            weights=weights,
            dinov2_weights=dinov2_weights,
            use_custom_corr=False,
        ).eval()
        if hasattr(self.matcher, "sample_thresh"):
            self.matcher.sample_thresh = sample_thresh

    def _tensor_to_pil(self, image_tensor):
        image_np = image_tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        return self._image_cls.fromarray((image_np * 255.0).astype(np.uint8))

    def _estimate_pair(self, img0, img1, k_row):
        image0 = self._tensor_to_pil(img0)
        image1 = self._tensor_to_pil(img1)
        width0, height0 = image0.size
        width1, height1 = image1.size
        warp, certainty = self.matcher.match(image0, image1, device=self.device)
        matches, _ = self.matcher.sample(warp, certainty)
        kpts0, kpts1 = self.matcher.to_pixel_coordinates(matches, height0, width0, height1, width1)
        return self._pose_from_matches(
            kpts0.detach().cpu().numpy(),
            kpts1.detach().cpu().numpy(),
            k_row,
            img0.device,
            self.pose_thresh,
            self.min_matches,
        )

    def forward(self, img0, img1, K):
        rs, ts = [], []
        for i in range(img0.shape[0]):
            try:
                r, t = self._estimate_pair(img0[i], img1[i], K[i])
            except Exception as exc:
                print(f"RoMa failed on sample {i}: {exc}")
                r = torch.eye(3, dtype=torch.float32, device=img0.device)
                t = torch.zeros(3, dtype=torch.float32, device=img0.device)
            rs.append(r.to(img0.device))
            ts.append(t.to(img0.device))
        return [torch.stack(rs, 0), torch.stack(ts, 0)]


class TinyRoMaPairMatcher(RoMaPairMatcher):
    def __init__(self, sample_thresh=0.05, pose_thresh=1.0, min_matches=8):
        nn.Module.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pose_thresh = pose_thresh
        self.min_matches = min_matches
        tiny_ckpt = osp.expanduser("~/.cache/torch/hub/checkpoints/tiny_roma_v1_outdoor.pth")
        xfeat_repo = osp.expanduser("~/.cache/torch/hub/verlab_accelerated_features_main")
        xfeat_ckpt = osp.join(xfeat_repo, "weights", "xfeat.pt")
        from PIL import Image
        from romatch.models.model_zoo import tiny_roma_v1_model
        if not osp.isfile(tiny_ckpt):
            raise FileNotFoundError(f"Missing Tiny RoMa checkpoint: {tiny_ckpt}")
        if not osp.isfile(xfeat_ckpt):
            raise FileNotFoundError(f"Missing XFeat checkpoint: {xfeat_ckpt}")
        if xfeat_repo not in sys.path:
            sys.path.insert(0, xfeat_repo)
        from modules.xfeat import XFeat as LocalXFeat
        self._image_cls = Image
        weights = torch.load(tiny_ckpt, map_location=self.device)
        xfeat = LocalXFeat(weights=xfeat_ckpt, top_k=4096).net
        self.matcher = tiny_roma_v1_model(weights=weights, xfeat=xfeat).to(self.device).eval()
        if hasattr(self.matcher, "sample_thresh"):
            self.matcher.sample_thresh = sample_thresh

    def _estimate_pair(self, img0, img1, k_row):
        image0 = self._tensor_to_pil(img0)
        image1 = self._tensor_to_pil(img1)
        width0, height0 = image0.size
        width1, height1 = image1.size
        warp, certainty = self.matcher.match(image0, image1)
        matches, _ = self.matcher.sample(warp, certainty)
        kpts0, kpts1 = self.matcher.to_pixel_coordinates(matches, height0, width0, height1, width1)
        return self._pose_from_matches(
            kpts0.detach().cpu().numpy(),
            kpts1.detach().cpu().numpy(),
            k_row,
            img0.device,
            self.pose_thresh,
            self.min_matches,
        )


class LoFTRPairMatcher(FeaturePoseMatcher):
    def __init__(self, ckpt_path=None, pose_thresh=1.0, min_matches=8):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pose_thresh = pose_thresh
        self.min_matches = min_matches
        self.ckpt_path = ckpt_path or "/home/pan/下载/indoor_ds_new.ckpt"
        if not osp.isfile(self.ckpt_path):
            raise FileNotFoundError(f"Missing LoFTR checkpoint: {self.ckpt_path}")
        from LoFTR.src.loftr import LoFTR, default_cfg
        self.matcher = LoFTR(default_cfg).to(self.device).eval()
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.matcher.load_state_dict(state_dict, strict=False)

    def _estimate_pair(self, img0, img1, k_row):
        batch = {
            "image0": img0.mean(dim=0, keepdim=True).unsqueeze(0).to(self.device),
            "image1": img1.mean(dim=0, keepdim=True).unsqueeze(0).to(self.device),
        }
        with torch.no_grad():
            self.matcher(batch)
        kpts0 = batch["mkpts0_f"].detach().cpu().numpy()
        kpts1 = batch["mkpts1_f"].detach().cpu().numpy()
        return self._pose_from_matches(kpts0, kpts1, k_row, img0.device, self.pose_thresh, self.min_matches)

    def forward(self, img0, img1, K):
        rs, ts = [], []
        for i in range(img0.shape[0]):
            try:
                r, t = self._estimate_pair(img0[i], img1[i], K[i])
            except Exception as exc:
                print(f"LoFTR failed on sample {i}: {exc}")
                r = torch.eye(3, dtype=torch.float32, device=img0.device)
                t = torch.zeros(3, dtype=torch.float32, device=img0.device)
            rs.append(r.to(img0.device))
            ts.append(t.to(img0.device))
        return [torch.stack(rs, 0), torch.stack(ts, 0)]


class EfficientLoFTRPairMatcher(LoFTRPairMatcher):
    def __init__(self, repo_dir="/tmp/EfficientLoFTR", ckpt_path="/home/pan/下载/eloftr_outdoor.ckpt", pose_thresh=1.0, min_matches=8):
        nn.Module.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pose_thresh = pose_thresh
        self.min_matches = min_matches
        if not osp.isdir(repo_dir):
            raise FileNotFoundError(f"Missing EfficientLoFTR repo: {repo_dir}")
        if not osp.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing EfficientLoFTR checkpoint: {ckpt_path}")
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from copy import deepcopy
        from src.loftr import LoFTR, full_default_cfg, reparameter
        config = deepcopy(full_default_cfg)
        config["mp"] = False
        config["half"] = False
        self.matcher = LoFTR(config=config)
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        self.matcher.load_state_dict(state_dict, strict=True)
        self.matcher = reparameter(self.matcher).eval().to(self.device)

    @staticmethod
    def _resize_to_multiple(image_tensor, multiple=32):
        _, _, height, width = image_tensor.shape
        new_height = max(multiple, (height // multiple) * multiple)
        new_width = max(multiple, (width // multiple) * multiple)
        if new_height == height and new_width == width:
            return image_tensor, 1.0, 1.0
        resized = torch.nn.functional.interpolate(image_tensor, size=(new_height, new_width), mode="bilinear", align_corners=False)
        return resized, width / float(new_width), height / float(new_height)

    def _estimate_pair(self, img0, img1, k_row):
        img0_gray = img0.mean(dim=0, keepdim=True).unsqueeze(0)
        img1_gray = img1.mean(dim=0, keepdim=True).unsqueeze(0)
        img0_in, sx0, sy0 = self._resize_to_multiple(img0_gray)
        img1_in, sx1, sy1 = self._resize_to_multiple(img1_gray)
        batch = {"image0": img0_in.to(self.device), "image1": img1_in.to(self.device)}
        with torch.no_grad():
            self.matcher(batch)
        kpts0 = batch["mkpts0_f"].detach().cpu().numpy().copy()
        kpts1 = batch["mkpts1_f"].detach().cpu().numpy().copy()
        if len(kpts0) >= self.min_matches:
            kpts0[:, 0] *= sx0
            kpts0[:, 1] *= sy0
            kpts1[:, 0] *= sx1
            kpts1[:, 1] *= sy1
        return self._pose_from_matches(kpts0, kpts1, k_row, img0.device, self.pose_thresh, self.min_matches)


class Reloc3rPairMatcher(nn.Module):
    def __init__(self, img_size=512, ckpt_path=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = ckpt_path or f"checkpoints/Reloc3r-{img_size}.pth"
        if not osp.isfile(self.ckpt_path):
            raise FileNotFoundError(f"Missing Reloc3r checkpoint: {self.ckpt_path}")
        self.model = Reloc3rRelpose(img_size=img_size).to(self.device)
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def forward(self, img0, img1, K=None):
        pose1, _ = self.model({"img": img0}, {"img": img1})
        pose = pose1["pose"]
        return [pose[:, :3, :3], pose[:, :3, 3]]


class Dust3RPairMatcher(nn.Module):
    def __init__(self, ckpt_path, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not osp.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing DUSt3R checkpoint: {ckpt_path}")
        from dust3r_main.dust3r.model import AsymmetricCroCo3DStereo
        self.model = AsymmetricCroCo3DStereo.from_pretrained(ckpt_path).to(self.device).eval()

    def _estimate_pair(self, img0, img1):
        from dust3r_main.dust3r.cloud_opt import GlobalAlignerMode, global_aligner
        from dust3r_main.dust3r.inference import inference
        h0, w0 = img0.shape[-2:]
        h1, w1 = img1.shape[-2:]
        view0 = {"img": img0.unsqueeze(0), "true_shape": np.int32([[h0, w0]]), "idx": 0, "instance": "0"}
        view1 = {"img": img1.unsqueeze(0), "true_shape": np.int32([[h1, w1]]), "idx": 1, "instance": "1"}
        output = inference([(view0, view1), (view1, view0)], self.model, self.device, batch_size=1, verbose=False)
        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PairViewer, verbose=False)
        poses = scene.get_im_poses().detach().to(self.device)
        rel = torch.linalg.inv(poses[1]) @ poses[0]
        return rel[:3, :3].float(), rel[:3, 3].float()

    def forward(self, img0, img1, K=None):
        rs, ts = [], []
        for i in range(img0.shape[0]):
            try:
                r, t = self._estimate_pair(img0[i].to(self.device), img1[i].to(self.device))
            except Exception as exc:
                print(f"DUSt3R failed on sample {i}: {exc}")
                r = torch.eye(3, dtype=torch.float32, device=self.device)
                t = torch.zeros(3, dtype=torch.float32, device=self.device)
            rs.append(r.to(img0.device))
            ts.append(t.to(img0.device))
        return [torch.stack(rs, 0), torch.stack(ts, 0)]


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline relative-pose evals with existing LEM evaluate.py/evaluate_A protocol functions.")
    parser.add_argument("--model", required=True, choices=["roma", "tinyroma", "dust3r", "reloc3r_s", "reloc3r_l", "loftr", "efficientloftr"])
    parser.add_argument("--benchmark", required=True, choices=["arkit", "scannet1500", "tum"])
    parser.add_argument("--test-dataset", default=None, help="Override reloc3r dataset string for ARKit/ScanNet.")
    parser.add_argument("--resolution", default="512,384", help="Width,height for ARKit/ScanNet.")
    parser.add_argument("--pair-mode", default="adjacent", choices=["adjacent"], help="This repo's ScanNet1500 uses the next frame, not test.npz image2.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataroot", default="/tmp/TUM", help="TUM root for evaluate.py protocol.")
    parser.add_argument("--keyframes", default="4,8", help="Comma-separated TUM keyframes.")
    parser.add_argument("--trajectory", default="", help="Optional single TUM trajectory.")
    parser.add_argument("--output-dir", default="baseline_eval_results_zs")
    parser.add_argument("--dust3r-ckpt", default="/home/pan/下载/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth")
    parser.add_argument("--reloc3r-s-ckpt", default="checkpoints/Reloc3r-224.pth")
    parser.add_argument("--reloc3r-l-ckpt", default="checkpoints/Reloc3r-512.pth")
    parser.add_argument("--loftr-ckpt", default="/home/pan/下载/indoor_ds_new.ckpt")
    parser.add_argument("--efficientloftr-repo", default=os.environ.get("EFFICIENTLOFTR_REPO", "/tmp/EfficientLoFTR"))
    parser.add_argument("--efficientloftr-ckpt", default=os.environ.get("EFFICIENTLOFTR_CKPT", "/home/pan/下载/eloftr_outdoor.ckpt"))
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def parse_resolution(value):
    width, height = value.split(",")
    return int(width), int(height)


def build_model(args, device):
    if args.model == "roma":
        return RoMaPairMatcher().to(device)
    if args.model == "tinyroma":
        return TinyRoMaPairMatcher().to(device)
    if args.model == "dust3r":
        return Dust3RPairMatcher(args.dust3r_ckpt, device=device).to(device)
    if args.model == "reloc3r_s":
        return Reloc3rPairMatcher(img_size=224, ckpt_path=args.reloc3r_s_ckpt, device=device).to(device)
    if args.model == "reloc3r_l":
        return Reloc3rPairMatcher(img_size=512, ckpt_path=args.reloc3r_l_ckpt, device=device).to(device)
    if args.model == "loftr":
        return LoFTRPairMatcher(ckpt_path=args.loftr_ckpt).to(device)
    if args.model == "efficientloftr":
        return EfficientLoFTRPairMatcher(repo_dir=args.efficientloftr_repo, ckpt_path=args.efficientloftr_ckpt).to(device)
    raise ValueError(args.model)


def default_dataset_string(args):
    width, height = parse_resolution(args.resolution)
    if args.benchmark == "arkit":
        return f"ARKitScenes(split='test', resolution=({width},{height}))"
    if args.benchmark == "scannet1500":
        return f"ScanNet1500(resolution=({width},{height}), seed=777)"
    raise ValueError(args.benchmark)


def summarize_info(info):
    return {
        "epe_cm": float(info["epes"].mean() * 100.0),
        "rot_deg": float(info["angular_error"].mean() * 180.0 / np.pi),
        "trans_cm": float(info["translation_error"].mean() * 100.0),
        "total_frames": int(info["epes"].shape[0]),
    }


def write_outputs(args, info_by_name):
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    prefix = f"{args.model}_{args.benchmark}{suffix}"
    pkl_path = osp.join(args.output_dir, f"{prefix}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(info_by_name, f)
    rows, per_pair_rows = [], []
    for name, info in info_by_name.items():
        summary = summarize_info(info)
        rows.append({"eval_name": name, **summary})
        names = info.get("names", [])
        for idx in range(info["epes"].shape[0]):
            per_pair_rows.append({
                "eval_name": name,
                "idx": idx,
                "name": names[idx] if idx < len(names) else "",
                "epe_cm": float(info["epes"][idx] * 100.0),
                "rot_deg": float(info["angular_error"][idx] * 180.0 / np.pi),
                "trans_cm": float(info["translation_error"][idx] * 100.0),
            })
    summary_df = pd.DataFrame(rows)
    per_pair_df = pd.DataFrame(per_pair_rows)
    summary_path = osp.join(args.output_dir, f"{prefix}_summary.csv")
    per_pair_path = osp.join(args.output_dir, f"{prefix}_per_pair.csv")
    json_path = osp.join(args.output_dir, f"{prefix}_summary.json")
    summary_df.to_csv(summary_path, index=False)
    per_pair_df.to_csv(per_pair_path, index=False)
    payload = {
        "model": args.model,
        "benchmark": args.benchmark,
        "summaries": rows,
        "mean_over_eval_names": {
            "epe_cm": float(summary_df["epe_cm"].mean()),
            "rot_deg": float(summary_df["rot_deg"].mean()),
            "trans_cm": float(summary_df["trans_cm"].mean()),
        },
        "paths": {"pkl": pkl_path, "summary_csv": summary_path, "per_pair_csv": per_pair_path, "summary_json": json_path},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_reloc3r_style(args, model, device):
    dataset = args.test_dataset or default_dataset_string(args)
    loader = get_data_loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_mem=True, shuffle=False, drop_last=False)
    protocol_args = SimpleNamespace(require_IMU=False, require_dicInput=False, dt=0.1, noise_gyro=0.0, noise_accel=0.0)
    info = a_protocol.evaluate_trust_region_A(
        loader,
        model,
        ["EPE3D", "RPE"],
        eval_name=f"{args.model}_{args.benchmark}",
        device=device,
        args=protocol_args,
    )
    return {dataset.split("(")[0]: info}


def run_tum(args, model):
    tum_protocol.args = SimpleNamespace(require_IMU=False, require_dicInput=False, speed=1)
    loader_args = SimpleNamespace(dataset="TUM_RGBD", dataroot=args.dataroot, cpu_workers=args.num_workers, eval_set="test")
    keyframes = [int(x) for x in args.keyframes.split(",") if x]
    loaders = tum_protocol.create_eval_loaders(loader_args, keyframes, args.batch_size, args.trajectory)
    info_by_name = {}
    for name, loader in loaders.items():
        kf = int(name.split("_keyframe_")[-1])
        info_by_name[name] = tum_protocol.evaluate_trust_region(loader, model, kf, ["EPE3D", "RPE"], eval_name=f"{args.model}_{name}")
    return info_by_name


def main():
    args = parse_args()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = build_model(args, device)
    if args.benchmark == "tum":
        info_by_name = run_tum(args, model)
    else:
        info_by_name = run_reloc3r_style(args, model, device)
    write_outputs(args, info_by_name)


if __name__ == "__main__":
    main()
