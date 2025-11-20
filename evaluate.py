import torch

import os, sys, argparse, pickle
import os.path as osp
import numpy as np
import pandas as pd

import torch.utils.data as data
import torchvision.utils as torch_utils
import torch.nn as nn

import LEM_SFM.evaluate as eval_utils
from LEM_SFM.config import get_model_args,get_args
from reloc3r.reloc3r_relpose import Reloc3rRelpose

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

def create_eval_loaders(args, keyframes, total_batch_size,trajectory  = ''):
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
                                  load_type='test',
                                  trajectory=trajectory)
            eval_loaders['{:}_keyframe_{:}'.format(trajectory, kf)] = data.DataLoader(np_loader, 
                batch_size = int(total_batch_size),
                shuffle = False, num_workers = args.cpu_workers,
                pin_memory=True)
    return eval_loaders

def evaluate_trust_region(dataloader, net,kf, objectives, eval_name='',
        timers = None):
    progress = tqdm(dataloader, ncols=100,
        desc = 'evaluate deeper inverse compositional algorithm {:}'.format(eval_name),
        total= len(dataloader))

    net.eval()
    total_frames = len(dataloader.dataset)

    outputs = {
        'R_est': np.zeros((total_frames, 3, 3)),
        't_est': np.zeros((total_frames, 3)),
        'R_gt': np.zeros((total_frames, 3, 3)),
        't_gt': np.zeros((total_frames, 3)),
        'R_IMU': np.zeros((total_frames, 3, 3)),
        't_IMU': np.zeros((total_frames, 3)),
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

    outputs['IMU_epe_error'] = np.zeros(total_frames) 
    outputs['IMU_angular_error'] = np.zeros(total_frames) 
    outputs['IMU_translation_error'] = np.zeros(total_frames) 

    outputs['zero_epe'] = np.zeros(total_frames) 
    outputs['zero_angular'] = np.zeros(total_frames) 
    outputs['zero_translation'] = np.zeros(total_frames) 

    count_base = 0

    if timers: timers.tic('one iteration')

    count = 1

    for idx, batch in enumerate(progress):

        if timers: timers.tic('forward step')

        names = batch[-1]

        color0, color1, depth0_gt, depth1_gt, Rt, K = \
            train_utils.check_cuda(batch[:6])
        obj_mask0, obj_mask1 = None, None
        R_gt, t_gt = Rt[:,:3,:3], Rt[:,:3,3]

        B, _, H, W = depth0_gt.shape

        with torch.no_grad():
            if args.require_IMU == True and args.require_dicInput==False:
                dt = 1.0/30*kf*args.speed
                s_pose0 = simulate_pose_from_imu(R_gt, t_gt,dt = dt,noise_std_gyro=args.noise_gyro, noise_std_accel=args.noise_accel)
                output,_= net.forward(color0, color1, K,s_pose0)
                R, t = output

                IMU_R,IMU_t = s_pose0
                
            elif args.require_IMU == False and args.require_dicInput==False:
                output= net.forward(color0, color1, K)
                R, t = output

            elif args.require_IMU == False and args.require_dicInput==True:
                view0 = {}
                view1 = {}
                view0['img'] = color0
                view1['img'] = color1
                output1,output2 = net.forward(view0, view1)

                R, t = output1['pose'][:,:3,:3],output1['pose'][:,:3,3]
        
        import torch.nn.functional as F

        t_pred_norm = F.normalize(t, p=2, dim=1)

        scale = torch.sum(t_pred_norm * t_gt, dim=1, keepdim=True)

        t = t_pred_norm * scale
        
        if timers: timers.toc('forward step')

        outputs['R_est'][count_base:count_base+B] = R.cpu().numpy()
        outputs['t_est'][count_base:count_base+B] = t.cpu().numpy()
        outputs['R_gt'][count_base:count_base+B] = R_gt.cpu().numpy()
        outputs['t_gt'][count_base:count_base+B] = t_gt.cpu().numpy()

        if rpe_loss:             
            angle_error, trans_error = rpe_loss(R, t, R_gt, t_gt)
            outputs['angular_error'][count_base:count_base+B] = angle_error.cpu().numpy()
            outputs['translation_error'][count_base:count_base+B] = trans_error.cpu().numpy()
            if args.require_IMU == True and args.require_dicInput==False:
                IMU_angle_error, IMU_trans_error = rpe_loss(IMU_R, IMU_t, R_gt, t_gt)
                outputs['IMU_angular_error'][count_base:count_base+B]  = IMU_angle_error.cpu().numpy()
                outputs['IMU_translation_error'][count_base:count_base+B]  = IMU_trans_error.cpu().numpy()

        if flow_loss:
            invalid_mask = (depth0_gt == depth0_gt.min()) + (depth0_gt == depth0_gt.max())
            if obj_mask0 is not None: invalid_mask = ~obj_mask0 + invalid_mask

            epes3d = flow_loss(R, t, R_gt, t_gt, depth0_gt, K, invalid_mask)            
            outputs['epes'][count_base:count_base+B] = epes3d.cpu().numpy()
            if args.require_IMU == True and args.require_dicInput==False:
                IMU_epes3d = flow_loss(IMU_R, IMU_t, R_gt, t_gt, depth0_gt, K, invalid_mask)   
                outputs['IMU_epe_error'][count_base:count_base+B]  = IMU_epes3d.cpu().numpy()

        outputs['names'] += names

        count_base += B

        if timers: timers.toc('evaluate')
        if timers: timers.toc('one iteration')
        if timers: timers.tic('one iteration')

    if timers: timers.print()

    return outputs

if __name__ == '__main__':
    print('=> Set seed...')
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print('=> Set args...')
    model_args = get_model_args()
    args = get_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.dataroot = '/home/data/TUM'

    args.batch_per_gpu = 64

    args.keyframes = '4'

    noise_gyro = [0.48,0.48*2,0.48*0.5,0.48*0.25]
    noise_accel = [0.073,0.073*2,0.073*0.5,0.073*0.025]

    idx = 1
    args.noise_gyro = noise_gyro[idx-1]
    args.noise_accel = noise_accel[idx-1]
    args.dt = 0.1

    print('Using device : ', args.device)
    args.start_epoch = 0

    args.model = 'Classic()'
    print('=> Set Model...', args.model)

    if 'DepthPoseNet' in args.model:
        args.require_IMU=True
        args.require_dicInput=False
        args.checkpoint_Union='/home/code/reloc3r/reloc3r-main/logs/save/checkpoint_epoch57.pth.tar'
    elif 'SPSG' in args.model or 'Classic' in args.model:
        args.require_IMU=False
        args.require_dicInput=False
    elif 'LoFTR_class' in args.model:
        args.require_IMU=False
        args.require_dicInput=False
    elif 'Reloc3rRelpose' in args.model:
        args.require_IMU=False
        args.require_dicInput=True
        if '512' in args.model:
            args.checkpoint_Union ='checkpoints/Reloc3r-512.pth'
        if '224' in args.model:
            args.checkpoint_Union ='checkpoints/Reloc3r-224.pth'
    
    print('=> Set checkpoints...', args.checkpoint_Union)

    args.eval_set = 'TUM_RGBD'
    print('=> Set eval_set...', args.eval_set)

    if torch.cuda.device_count() > 1:
        print('---------------------------------------') 
        print("Use", torch.cuda.device_count(), "GPUs for evaluate!")

    print('---------------------------------------') 
    outputs = test_TrustRegion_TUM(args,model_args)
