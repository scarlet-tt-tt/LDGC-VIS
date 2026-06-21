# Remote evaluation setup

This repository contains the code needed by `scripts/eval_baselines_zs.sh`, but
large datasets and most model weights must be prepared on the remote machine.
Do not push the `checkpoints/Reloc3r-*.pth` files with normal Git; each file is
larger than GitHub's regular file limit.

## Files to upload with git

Commit the code changes plus these newly added source directories/files:

```bash
git add .gitignore LEM_SFM/evaluate.py eval_baselines.py scripts/eval_baselines_zs.sh reloc3r LoFTR docs/remote_eval.md
git status --short
git commit -m "Add zero-shot baseline evaluation runner"
git push origin main
```

`checkpoints/Reloc3r-224.pth` and `checkpoints/Reloc3r-512.pth` are ignored on
purpose. Transfer them to the remote server separately, or use Git LFS if your
GitHub account has enough LFS quota.

## Remote directory layout

After cloning the repository on the server, prepare these paths:

```text
checkpoints/Reloc3r-224.pth
checkpoints/Reloc3r-512.pth
$DUST3R_CKPT
$LOFTR_CKPT
$EFFICIENTLOFTR_REPO
$EFFICIENTLOFTR_CKPT
$TUM_ROOT
$ARKITSCENES_ROOT
$SCANNET1500_ROOT
~/.cache/torch/hub/checkpoints/roma_outdoor.pth
~/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth
~/.cache/torch/hub/checkpoints/tiny_roma_v1_outdoor.pth
~/.cache/torch/hub/verlab_accelerated_features_main/weights/xfeat.pt
```

`ARKITSCENES_ROOT` must contain `Validation/new.npz` and the corresponding
`vga_wide/` and `lowres_depth/` folders. `SCANNET1500_ROOT` must contain
`test.npz` and `scannet_test_1500/`.

## Environment

Use the existing reloc3r conda environment if it already works locally. On a new
machine, install PyTorch for the server CUDA version first, then install the
Python dependencies:

```bash
cd LDGC-VIS
/home/pan/anaconda3/envs/reloc3r/bin/python -m pip install -r requirements_reloc3r.txt -r requirements_addition.txt
```

`LoFTR` is imported from this repository path, so run from the repository root.
`DUSt3R` and `EfficientLoFTR` are external source trees. Put them on the server
and point the variables below to them or install them into the environment.

## Run

Use absolute paths for all assets on the remote server:

```bash
cd LDGC-VIS
export PY=/home/pan/anaconda3/envs/reloc3r/bin/python
export OUT=baseline_eval_results_zs
export PYTHONPATH="$PWD:$PYTHONPATH"
TUM_ROOT=/path/to/TUM \
ARKITSCENES_ROOT=/path/to/arkitscenes_processed \
SCANNET1500_ROOT=/path/to/scannet1500 \
DUST3R_CKPT=/path/to/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth \
LOFTR_CKPT=/path/to/indoor_ds_new.ckpt \
EFFICIENTLOFTR_REPO=/path/to/EfficientLoFTR \
EFFICIENTLOFTR_CKPT=/path/to/eloftr_outdoor.ckpt \
bash scripts/eval_baselines_zs.sh
```

For a quick smoke test before the full run, execute one command:

```bash
PY=/home/pan/anaconda3/envs/reloc3r/bin/python
TUM_ROOT=/path/to/TUM
$PY eval_baselines.py --model reloc3r_s --benchmark tum --keyframes 4 --dataroot "$TUM_ROOT" --output-dir smoke_eval
```
