""" 
Evaluation scripts

@author: Zhaoyang Lv
@date: March 2019
"""
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
# from LEM_SFM.model_with_depth import DepthPoseNet_IMU_free,PoseNet_IMU,DepthPoseNet_Deeplabv3,DepthPoseNet_Deeplabv3_i,Classic,DepthPoseNet,SPSG,LoFTR_class

import LEM_SFM.models.LeastSquareTracking as ICtracking
import LEM_SFM.models.criterions as criterions
import LEM_SFM.train_utils as train_utils
import LEM_SFM.config as config
from IMU import simulate_pose_from_imu

from LEM_SFM.data.dataloader import load_data
from LEM_SFM.timers import Timers
from tqdm import tqdm
# from relative_depth.evaluate import compute_scale_and_shift,compute_errors,RunningAverageDict

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

import warnings
# 忽略 imageio 中关于 Pillow 的 16位 PNG 警告
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
            
            # 'fr1/rgbd_dataset_freiburg1_360', 
            # 'fr1/rgbd_dataset_freiburg1_desk', 
            # 'fr2/rgbd_dataset_freiburg2_desk', 
            # 'fr2/rgbd_dataset_freiburg2_pioneer_360'
            'fr1/rgbd_dataset_freiburg1_xyz',
            'fr1/rgbd_dataset_freiburg1_rpy',
            'fr1/rgbd_dataset_freiburg1_360',
            # 'fr1/rgbd_dataset_freiburg1_desk',
            'fr1/rgbd_dataset_freiburg1_teddy',

            # 'fr2/rgbd_dataset_freiburg2_xyz',
            'fr2/rgbd_dataset_freiburg2_desk_with_person',
            'fr2/rgbd_dataset_freiburg2_rpy',
            'fr2/rgbd_dataset_freiburg2_pioneer_slam3',
            'fr3/rgbd_dataset_freiburg3_structure_notexture_far',
            'fr3/rgbd_dataset_freiburg3_structure_texture_near',
            'fr3/rgbd_dataset_freiburg3_walking_xyz'

            ]
    }[dataset]



def create_eval_loaders(args, keyframes, total_batch_size,trajectory  = ''):
    """ create the evaluation loader at different keyframes set-up
    """
    eval_loaders = {}

    if trajectory == '': 
        trajectories = eval_trajectories(args.dataset)
    else: 
        trajectories = [trajectory]

    for trajectory in trajectories:
        # print('trajectory',trajectory)
        for kf in keyframes:
            # print('args.eval_set',args.eval_set)
            # print('args.dataroot',args.dataroot)
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
    """ evaluate the trust-region method given the two-frame pose estimation
    :param the pytorch dataloader
    :param the network
    :param the evaluation objective names, e.g. RPE, EPE3D
    :param True if ground mask if known
    :param (optional) timing each step
    """

    """在双帧姿态估计的情况下，对信赖域方法进行评估
    ：参数pytorch数据加载器
    ：网络参数
    ：参数评估目标名称，例如RPE、EPE3D
    ：param如果已知接地掩码，则为True
    ：param（可选）为每一步计时
    """

    progress = tqdm(dataloader, ncols=100,
        desc = 'evaluate deeper inverse compositional algorithm {:}'.format(eval_name),
        total= len(dataloader))

    net.eval()
# 初始化输出字典
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

# 初始化计数器
    count_base = 0

    if timers: timers.tic('one iteration')

    count = 1
    # print('遍历数据加载器')
# 遍历数据加载器    
    for idx, batch in enumerate(progress):

        if timers: timers.tic('forward step')

        names = batch[-1]

        color0, color1, depth0_gt, depth1_gt, Rt, K = \
            train_utils.check_cuda(batch[:6])
        obj_mask0, obj_mask1 = None, None
        R_gt, t_gt = Rt[:,:3,:3], Rt[:,:3,3]

        B, _, H, W = depth0_gt.shape

# 前向传播            
        with torch.no_grad():
            if args.require_IMU == True and args.require_dicInput==False:
                dt = 1.0/30*kf*args.speed
                # s_pose0 = simulate_pose_from_imu(R_gt, t_gt,dt = dt)
                s_pose0 = simulate_pose_from_imu(R_gt, t_gt,dt = dt,noise_std_gyro=args.noise_gyro, noise_std_accel=args.noise_accel)
                output,_= net.forward(color0, color1, K,s_pose0)
                R, t = output

                IMU_R,IMU_t = s_pose0
                # R, t = s_pose0
                
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

        t_pred_norm = F.normalize(t, p=2, dim=1)  # 归一化预测向量

        # 计算最佳缩放因子 scale
        scale = torch.sum(t_pred_norm * t_gt, dim=1, keepdim=True)  # 点积计算最佳缩放因子

        # 将 t_pred 沿自身方向缩放
        t = t_pred_norm * scale  # 缩放后的 t_pred
        
        if timers: timers.toc('forward step')

        # 保存结果，count_base:count_base+B是一个切片，将B（batch）个结果保存到outputs中
        outputs['R_est'][count_base:count_base+B] = R.cpu().numpy()
        outputs['t_est'][count_base:count_base+B] = t.cpu().numpy()
        outputs['R_gt'][count_base:count_base+B] = R_gt.cpu().numpy()
        outputs['t_gt'][count_base:count_base+B] = t_gt.cpu().numpy()
        # outputs['R_IMU'][count_base:count_base+B] = IMU_R.cpu().numpy()
        # outputs['t_IMU'][count_base:count_base+B] = IMU_t.cpu().numpy()
        # metrics = RunningAverageDict()

# 计算评估指标
        # print('计算评估指标')

        if rpe_loss: # evaluate the relative pose error             
            angle_error, trans_error = rpe_loss(R, t, R_gt, t_gt)
            outputs['angular_error'][count_base:count_base+B] = angle_error.cpu().numpy()
            outputs['translation_error'][count_base:count_base+B] = trans_error.cpu().numpy()
            if args.require_IMU == True and args.require_dicInput==False:
                IMU_angle_error, IMU_trans_error = rpe_loss(IMU_R, IMU_t, R_gt, t_gt)
                outputs['IMU_angular_error'][count_base:count_base+B]  = IMU_angle_error.cpu().numpy()
                outputs['IMU_translation_error'][count_base:count_base+B]  = IMU_trans_error.cpu().numpy()


        if flow_loss:# evaluate the end-point-error loss 3D
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


def evaluate_trust_region_A(dataloader, net,objectives,args, eval_name='',
        timers = None):
    """ evaluate the trust-region method given the two-frame pose estimation
    :param the pytorch dataloader
    :param the network
    :param the evaluation objective names, e.g. RPE, EPE3D
    :param True if ground mask if known
    :param (optional) timing each step
    """

    """在双帧姿态估计的情况下，对信赖域方法进行评估
    ：参数pytorch数据加载器
    ：网络参数
    ：参数评估目标名称，例如RPE、EPE3D
    ：param如果已知接地掩码，则为True
    ：param（可选）为每一步计时
    """

    progress = tqdm(dataloader, ncols=100,
        desc = 'evaluate deeper inverse compositional algorithm {:}'.format(eval_name),
        total= len(dataloader))

    net.eval()
# 初始化输出字典
    total_frames = len(dataloader.dataset)

    # outputs = {
    #     'R_est': np.zeros((total_frames, 3, 3)),
    #     't_est': np.zeros((total_frames, 3)),
    #     'names': []
    # }
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

# 初始化计数器
    count_base = 0

    if timers: timers.tic('one iteration')

    count = 1
    epoch = 0
    if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'set_epoch'):
        dataloader.dataset.set_epoch(epoch)
    if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)
    # print('遍历数据加载器')
# 遍历数据加载器    
    for idx, batch in enumerate(progress):

        if timers: timers.tic('forward step')

        names = 'names'
        for view in batch:
            for name in 'img camera_intrinsics camera_pose depth'.split(): 
                if name not in view:
                    continue
                view[name] = view[name].to(args.device, non_blocking=True)

        view0,view1 =batch
        depth0_gt = view0['depth']
        depth1_gt = view1['depth']
        color0 = view0['img']
        color1 = view1['img']
        K = intrinsics_matrix_to_k(view0['camera_intrinsics'])
        pose1_np = view0['camera_pose'].cpu().numpy() 
        pose2_np = view1['camera_pose'].cpu().numpy()

        Rt = np.linalg.inv(pose2_np) @ pose1_np

        Rt = torch.tensor(Rt).to(args.device)


        R_gt, t_gt = Rt[:,:3,:3], Rt[:,:3,3]

        B, _, H, W = depth0_gt.shape

# 前向传播            
        with torch.no_grad():
            if args.require_IMU == True and args.require_dicInput==False:
                # s_pose0 = simulate_pose_from_imu(R_gt, t_gt,dt = dt)
                s_pose0 = simulate_pose_from_imu(R_gt, t_gt,dt = args.dt,noise_std_gyro=args.noise_gyro, noise_std_accel=args.noise_accel)
                output,_= net.forward(color0, color1, K,s_pose0)
                R, t = output

                IMU_R,IMU_t = s_pose0

                # R, t = s_pose0
                
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

        t_pred_norm = F.normalize(t, p=2, dim=1)  # 归一化预测向量

        # 计算最佳缩放因子 scale
        scale = torch.sum(t_pred_norm * t_gt, dim=1, keepdim=True)  # 点积计算最佳缩放因子

        # 将 t_pred 沿自身方向缩放
        t = t_pred_norm * scale  # 缩放后的 t_pred
        
        if timers: timers.toc('forward step')

        # 保存结果，count_base:count_base+B是一个切片，将B（batch）个结果保存到outputs中
        outputs['R_est'][count_base:count_base+B] = R.cpu().numpy()
        outputs['t_est'][count_base:count_base+B] = t.cpu().numpy()
        outputs['R_gt'][count_base:count_base+B] = R_gt.cpu().numpy()
        outputs['t_gt'][count_base:count_base+B] = t_gt.cpu().numpy()
        outputs['R_IMU'][count_base:count_base+B] = IMU_R.cpu().numpy()
        outputs['t_IMU'][count_base:count_base+B] = IMU_t.cpu().numpy()
        # metrics = RunningAverageDict()

# 计算评估指标
        # print('计算评估指标')

        if rpe_loss: # evaluate the relative pose error             
            angle_error, trans_error = rpe_loss(R, t, R_gt, t_gt)
            outputs['angular_error'][count_base:count_base+B] = angle_error.cpu().numpy()
            outputs['translation_error'][count_base:count_base+B] = trans_error.cpu().numpy()
            if args.require_IMU == True and args.require_dicInput==False:
                IMU_angle_error, IMU_trans_error = rpe_loss(IMU_R, IMU_t, R_gt, t_gt)
                outputs['IMU_angular_error'][count_base:count_base+B]  = IMU_angle_error.cpu().numpy()
                outputs['IMU_translation_error'][count_base:count_base+B]  = IMU_trans_error.cpu().numpy()


        if flow_loss:# evaluate the end-point-error loss 3D
            invalid_mask = (depth0_gt == depth0_gt.min()) + (depth0_gt == depth0_gt.max())
            # if obj_mask0 is not None: invalid_mask = ~obj_mask0 + invalid_mask

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

def test_TrustRegion_TUM(args,model_args):

    if args.time:
        timers = Timers()
    else:
        timers = None

    total_batch_size = args.batch_per_gpu *  torch.cuda.device_count()

    keyframes = [int(x) for x in args.keyframes.split(',')]
    eval_loaders = create_eval_loaders(args, keyframes, total_batch_size,args.trajectory)

    net = eval(args.model)

    net = net.to(args.device) 

    if args.checkpoint_Union !='' and 'Reloc3rRelpose' in args.model:
        ckpt = torch.load(args.checkpoint_Union)
        if 'model' in ckpt: 
            ckpt = ckpt['model']
        net.load_state_dict(ckpt, strict=False)
        print("=> Load depth model from checkpoint_Union")  
    elif args.checkpoint_Union !='':
        print('==============')
        net.load_state_dict(torch.load(args.checkpoint_Union)['state_dict'])
        print("=> Load depth model from checkpoint_Union")  

    total_params = sum(p.numel() for p in net.parameters())
    total_params_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total params: {total_params}")
    print(f"Total params trainable: {total_params_trainable}")

    # 定义评价指标
    eval_objectives = ['EPE3D', 'RPE']
    output_prefix = 'output_prefix'

    # evaluate results per trajectory per key-frame
    outputs = {}
    outputs_d={}
    for k, loader in eval_loaders.items():
        print('================',k,'================')
        traj_name, kf = k.split('_keyframe_')

        print('Evaluate trajectory Pose{:} at keyframe {:}'.format(k, keyframes))

        output_name = '{:}_{:}'.format(output_prefix, k)

        info = evaluate_trust_region(loader, net,int(kf),
            eval_objectives,
            eval_name = 'tmp/'+output_name)

        # collect results 
        outputs[k] = pd.Series([info['epes'].mean(), 
            info['angular_error'].mean(), 
            info['translation_error'].mean(), 
            info['epes'].shape[0], int(kf), traj_name], 
            index=['3D EPE', 'axis error', 'trans error', 'total frames', 'keyframe', 'trajectory'])
        
        # 输出结果
        print(outputs[k])

        checkpoint_name = args.checkpoint_Union.replace('.pth.tar', '')
        if checkpoint_name == '':
            checkpoint_name = 'nolearning'
        output_dir = osp.join(args.eval_set+'_results', checkpoint_name, k)
        output_pkl = output_dir + '.pkl'
        
        check_directory(output_pkl)
        print(output_pkl)

        with open(output_pkl, 'wb') as output: # dump per-frame results info
            info = info
            pickle.dump(info, output)

    print(""" =============================================================== """)
    print("""             Generate the final evaluation results               """)
    print(""" =============================================================== """)

    outputs_d_pd = pd.DataFrame(outputs_d).T


    outputs_pd = pd.DataFrame(outputs).T
    outputs_pd['3D EPE'] *= 100 # convert to cm
    outputs_pd['axis error'] *= (180/np.pi) # convert to degree
    outputs_pd['trans error'] *= 100 # convert to cm

    print(outputs_pd)

    stats_dict = {}
    for kf in keyframes:        
        kf_outputs = outputs_pd[outputs_pd['keyframe']==kf]

        stats_dict['mean values of trajectories keyframe {:}'.format(kf)] = pd.Series(
            [kf_outputs['3D EPE'].mean(), 
             kf_outputs['axis error'].mean(),
             kf_outputs['trans error'].mean(), kf], 
            index=['3D EPE', 'axis error', 'trans error', 'keyframe'])

        total_frames = kf_outputs['total frames'].sum()
        stats_dict['mean values of frames keyframe {:}'.format(kf)] = pd.Series(
            [(kf_outputs['3D EPE'] * kf_outputs['total frames']).sum() / total_frames, 
             (kf_outputs['axis error'] * kf_outputs['total frames']).sum() / total_frames, 
             (kf_outputs['trans error']* kf_outputs['total frames']).sum() / total_frames, kf],
            index=['3D EPE', 'axis error', 'trans error', 'keyframe'])

    stats_pd = pd.DataFrame(stats_dict).T
    print(stats_pd)

    # final_pd = outputs_pd.append(stats_pd, sort=False)
    final_pd = pd.concat([outputs_pd, stats_pd],  sort=False)

    final_pd.to_csv('{:}.csv'.format(output_dir))

    return outputs_pd


def evaluate_depth_error(dataloader, net, objectives, eval_name='',
        known_mask = False, timers = None):

    progress = tqdm(dataloader, ncols=100,desc = 'evaluate deeper inverse compositional algorithm {:}'.format(eval_name), total= len(dataloader))

    net.eval()

# 初始化输出字典
    total_frames = len(dataloader.dataset)
    # outputs = {}
    outputs = {
        'a1': [],
        'a2': [],
        'a3': [],
        'abs_rel':[],
        'rmse':[],
        'log_10':[],
        'sq_rel':[],
        'rmse_log':[],
        'silog':[],
    }

# 初始化计数器
    # count_base = 0

    if timers: timers.tic('one iteration')

    count = 1
    # from DepthAnything.metric_depth.zoedepth.utils.misc import RunningAverageDict
    # metrics = RunningAverageDict()
    # print('遍历数据加载器')
# 遍历数据加载器    
    for idx, batch in enumerate(progress):
        # print('idx',idx)

        if timers: timers.tic('forward step')

        if known_mask: # for dataset that with mask or need mask
            color0, color1, depth0_gt, depth1_gt, Rt, K, obj_mask0, obj_mask1 = \
                train_utils.check_cuda(batch[:8])
        else:
            color0, color1, depth0_gt, depth1_gt, Rt, K = \
                train_utils.check_cuda(batch[:6])
            obj_mask0, obj_mask1 = None, None

        B, _, H, W = depth0_gt.shape




# 前向传播            
        with torch.no_grad():
            # output,inv_pre_depth0 ,inv_pre_depth1,feature0_infer,feature1_infer= net.forward(color0, color1, K)
            pre_depth0= net.forward(color0, color1, K)
            # R, t = output

        if timers: timers.toc('forward step')


# 计算评估指标
        # pre_depth0 = torch.where(inv_pre_depth0 != 0, 1.0 / inv_pre_depth0,  inv_pre_depth0)
        pre_depth0 = pre_depth0.squeeze(1).float().cpu().detach().numpy()
        depth0_gt=depth0_gt.squeeze(1).float().cpu().detach().numpy()

        # pred_max = np.max(pre_depth0,axis=(1,2),keepdims=True)
        # pred_min = np.min(pre_depth0,axis=(1,2),keepdims=True)
        # pre_depth0 = (pre_depth0 - pred_min) / (pred_max - pred_min)
        mask = np.ones_like(depth0_gt)
        th_min = 0.2
        th_max = 5
        # mask[depth0_gt<th_min] = 0
        invalid_mask_0 = (depth0_gt == depth0_gt.min()) + (depth0_gt == depth0_gt.max())
        
        scale, shift = compute_scale_and_shift(pre_depth0, depth0_gt, ~invalid_mask_0)

        prediction_aligned = scale.reshape((-1, 1, 1)) * pre_depth0 + shift.reshape((-1, 1, 1))


        prediction_aligned[prediction_aligned<th_min] = th_min
        prediction_aligned[prediction_aligned>th_max] = th_max
        depth0_gt[depth0_gt<th_min] = th_min
        depth0_gt[depth0_gt>th_max] = th_max

        output = compute_errors(depth0_gt,prediction_aligned)

        # print('\noutput',idx,'\t',output)

        outputs['a1'].append(output['a1'])
        outputs['a2'].append(output['a2'])
        outputs['a3'].append(output['a3'])
        outputs['abs_rel'].append(output['abs_rel'])
        outputs['rmse'].append(output['rmse'])
        outputs['log_10'].append(output['log_10'])
        outputs['sq_rel'].append(output['sq_rel'])
        outputs['rmse_log'].append(output['rmse_log'])
        outputs['silog'].append(output['silog'])

        # print('\noutputs',idx,'\t',outputs)
        # print('\na1',idx,'\t',np.mean(outputs['a1']))

        # metrics.update(outputs)

        # count_base += B

        if timers: timers.toc('evaluate')
        if timers: timers.toc('one iteration')
        if timers: timers.tic('one iteration')

    if timers: timers.print()
    # print(outputs)
    return outputs



def compute_scale_and_shift(prediction, target, mask):
        """ compute_scale_and_shift. """
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        # print('mask',mask.shape)
        # print('prediction',prediction.shape)
        a_00 = np.sum(mask * prediction * prediction, (1, 2))
        a_01 = np.sum(mask * prediction, (1, 2))
        a_11 = np.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = np.sum(mask * prediction * target, (1, 2))
        b_1 = np.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = np.zeros_like(b_0)
        x_1 = np.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1


def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (np.ndarray): Ground truth values
        pred (np.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

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

    # noise_gyro = [0.1,0.48,0.03]
    # noise_accel = [0.5,0.073,0.05]

    noise_gyro = [0.48,0.48*2,0.48*0.5,0.48*0.25]
    noise_accel = [0.073,0.073*2,0.073*0.5,0.073*0.025]

    idx = 1
    args.noise_gyro = noise_gyro[idx-1]
    args.noise_accel = noise_accel[idx-1]
    args.dt = 0.1

    print('Using device : ', args.device)
    args.start_epoch = 0

    
    # args.model = "Reloc3rRelpose(img_size=224)"
    args.model = 'Classic()'
    # args.model = 'DepthPoseNet_Deeplabv3(model_args)'
    # args.model = 'DepthPoseNet(model_args)'
    # args.model = 'LoFTR_class()'
    # args.model = 'SPSG()'
    print('=> Set Model...', args.model)

    if 'DepthPoseNet' in args.model:
        args.require_IMU=True
        args.require_dicInput=False
        # args.checkpoint_Union = '/home/code/reloc3r/reloc3r-main/logs/TUM_RGBD/train_A_4/checkpoint_epoch189.pth.tar'
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




