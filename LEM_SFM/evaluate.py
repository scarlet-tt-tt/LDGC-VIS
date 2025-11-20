import torch
import os, sys, argparse, pickle
import os.path as osp
import numpy as np
import pandas as pd
import torch.utils.data as data
import torchvision.utils as torch_utils
import torch.nn as nn
import LEM_SFM.models.LeastSquareTracking as ICtracking
import LEM_SFM.models.criterions as criterions
import LEM_SFM.train_utils as train_utils
import LEM_SFM.config as config
from IMU import simulate_pose_from_imu
from LEM_SFM.data.dataloader import load_data
from LEM_SFM.timers import Timers
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="imageio.plugins.pillow")
warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    module="imageio.plugins.pillow",
    message="Loading 16-bit \(uint16\) PNG as int32 due to limitations in pillow's PNG decoder."
)

def check_directory(filename):
    target_dir = os.path.dirname(filename)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

def eval_trajectories(dataset):
    return {
        'TUM_RGBD': [
            'fr1/rgbd_dataset_freiburg1_xyz',
            'fr1/rgbd_dataset_freiburg1_rpy',
            'fr1/rgbd_dataset_freiburg1_360',
            'fr1/rgbd_dataset_freiburg1_teddy',
            'fr2/rgbd_dataset_freiburg2_desk_with_person',
            'fr2/rgbd_dataset_freiburg2_rpy',
            'fr2/rgbd_dataset_freiburg2_pioneer_slam3',
            'fr3/rgbd_dataset_freiburg3_structure_notexture_far',
            'fr3/rgbd_dataset_freiburg3_structure_texture_near',
            'fr3/rgbd_dataset_freiburg3_walking_xyz'
        ]
    }[dataset]

def create_eval_loaders(args, keyframes, total_batch_size, trajectory=''):
    eval_loaders = {}
    if trajectory == '': 
        trajectories = eval_trajectories(args.dataset)
    else: 
        trajectories = [trajectory]
    for trajectory in trajectories:
        for kf in keyframes:
            np_loader = load_data(args.dataset, 
                                  dataroot=args.dataroot,
                                  keyframes=[kf],
                                  load_type=args.eval_set,
                                  trajectory=trajectory)
            eval_loaders['{:}_keyframe_{:}'.format(trajectory, kf)] = data.DataLoader(np_loader, 
                batch_size=int(total_batch_size),
                shuffle=False, num_workers=args.cpu_workers,
                pin_memory=True)
    return eval_loaders

def intrinsics_matrix_to_k(intrinsics):
    """
    将相机内参矩阵转换为[fx, fy, cx, cy]参数
    
    参数:
        intrinsics: 相机内参矩阵，形状为(batch_size, 3, 3)
    
    返回:
        K: 提取的参数，形状为(batch_size, 4)
    """
    # print(intrinsics.shape)
    # 提取对角线元素作为fx和fy，以及第三列的前两个元素作为cx和cy
    fx = intrinsics[:, 0, 0]  # 第一行第一列
    fy = intrinsics[:, 1, 1]  # 第二行第二列
    cx = intrinsics[:, 0, 2]  # 第一行第三列
    cy = intrinsics[:, 1, 2]  # 第二行第三列
    
    # # 拼接为(batch_size, 4)的张量
    # fx = torch.tensor(120.0)
    # fy = torch.tensor(160.0)
    # cx = torch.tensor(2.5550e+02)
    # cy = torch.tensor(1.4350e+02)

    K = torch.stack([fx, fy, cx, cy], dim=1)
    return K

def evaluate_trust_region_A(dataloader, net, objectives, eval_name='', epoch=None, device=False, args=False, timers=None):
    progress = tqdm(dataloader, ncols=100,
        desc='evaluate deeper inverse compositional algorithm {:}'.format(eval_name),
        total=len(dataloader))
    net.eval()
    total_frames = len(dataloader.dataset)
    outputs = {
        'R_est': np.zeros((total_frames, 3, 3)),
        't_est': np.zeros((total_frames, 3)),
        'names': []
    }
    flow_loss, rpe_loss = None, None
    if 'EPE3D' in objectives: 
        flow_loss = criterions.compute_RT_EPE_loss
        outputs['epes'] = np.zeros(total_frames)
    if 'RPE' in objectives:
        rpe_loss = criterions.compute_RPE_loss
        outputs['angular_error'] = np.zeros(total_frames)  
        outputs['translation_error'] = np.zeros(total_frames) 
    count_base = 0
    if timers: timers.tic('one iteration')
    if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'set_epoch'):
        dataloader.dataset.set_epoch(epoch)
    if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)
    for idx, batch in enumerate(progress):
        if timers: timers.tic('forward step')
        names = 'names'
        for view in batch:
            for name in 'img camera_intrinsics camera_pose depth'.split(): 
                if name not in view:
                    continue
                view[name] = view[name].to(device, non_blocking=True)
        view0, view1 = batch
        depth0_gt = view0['depth']
        depth1_gt = view1['depth']
        color0 = view0['img']
        color1 = view1['img']
        K = intrinsics_matrix_to_k(view0['camera_intrinsics'])
        pose1_np = view0['camera_pose'].cpu().numpy() 
        pose2_np = view1['camera_pose'].cpu().numpy()
        Rt = np.linalg.inv(pose2_np) @ pose1_np
        Rt = torch.tensor(Rt).to(device)
        invalid_mask_0 = (depth0_gt == depth0_gt.min()) + (depth0_gt == depth0_gt.max())
        B, _, H, W = depth0_gt.shape
        with torch.no_grad():
            R_gt, t_gt = Rt[:,:3,:3], Rt[:,:3,3]
            s_pose0 = simulate_pose_from_imu(R_gt, t_gt, dt=args.dt, noise_std_gyro=args.noise_gyro, noise_std_accel=args.noise_accel)
            output, pre_depth0 = net.forward(color0, color1, K, s_pose0)
            R, t = output
        import torch.nn.functional as F
        t_pred_norm = F.normalize(t, p=2, dim=1)
        scale = torch.sum(t_pred_norm * t_gt, dim=1, keepdim=True)
        t = t_pred_norm * scale
        if timers: timers.toc('forward step')
        outputs['R_est'][count_base:count_base+B] = R.cpu().numpy()
        outputs['t_est'][count_base:count_base+B] = t.cpu().numpy()
        if rpe_loss:             
            angle_error, trans_error = rpe_loss(R, t, R_gt, t_gt)
            outputs['angular_error'][count_base:count_base+B] = angle_error.cpu().numpy()
            outputs['translation_error'][count_base:count_base+B] = trans_error.cpu().numpy()
        if flow_loss:
            invalid_mask = (depth0_gt == depth0_gt.min()) + (depth0_gt == depth0_gt.max())
            epes3d = flow_loss(R, t, R_gt, t_gt, depth0_gt, K, invalid_mask)            
            outputs['epes'][count_base:count_base+B] = epes3d.cpu().numpy()
        outputs['names'] += names
        count_base += B
        if timers: timers.toc('evaluate')
        if timers: timers.toc('one iteration')
        if timers: timers.tic('one iteration')
    if timers: timers.print()
    return outputs
