import torch

import os, sys, argparse, pickle
import os.path as osp
import numpy as np
import pandas as pd


import torch.utils.data as data
import torchvision.utils as torch_utils
import torch.nn as nn

import evaluate as eval_utils

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

from reloc3r.datasets import get_data_loader  # noqa
def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader

def test_TrustRegion_A_Reloc3r(args,model_args):

    if args.time:
        timers = Timers()
    else:
        timers = None

    obj_has_mask = False
    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    print("Initialize and train the Deep Trust Region Network")

    print('Loading model: {:s}'.format(args.model))
    net = eval(args.model)

    if torch.cuda.is_available(): net.cuda()
    
    if args.checkpoint_Union !='':
        print('==============')
        ckpt = torch.load(args.checkpoint_Union)
        if 'model' in ckpt: 
            ckpt = ckpt['model']
        net.load_state_dict(ckpt, strict=False)

        print("=> Load depth model from checkpoint_Union")  
    
    logfile_name = 'eval_A'
    logger = train_utils.initialize_logger(args, logfile_name)

    train_objective = ['EPE3D']
    eval_objectives = ['EPE3D', 'RPE']

    outputs = {}
    outputs_d={}
    for test_name, testset in data_loader_test.items():

        eval_name = '{:}_{:}'.format(args.dataset, test_name)
        epoch = 0
        eval_pose_info = eval_utils.evaluate_trust_region_A(
            testset, net, eval_objectives, epoch=epoch,
            device  = args.device, 
            eval_name   = eval_name,
            timers      = timers)
        
        outputs = pd.Series([eval_pose_info['epes'].mean(), 
            eval_pose_info['angular_error'].mean(), 
            eval_pose_info['translation_error'].mean(), 
            eval_pose_info['epes'].shape[0]], 
            index=['3D EPE', 'axis error', 'trans error', 'total frames'])
        
        display_pose_dict = {"{:}_epe3d".format(eval_name): eval_pose_info['epes'].mean(), 
            "{:}_rpe_angular".format(eval_name): eval_pose_info['angular_error'].mean(), 
            "{:}_rpe_translation".format(eval_name): eval_pose_info['translation_error'].mean()}

        logger.write_to_tensorboard(display_pose_dict, epoch)
        logger.write_to_terminal_val(display_pose_dict, epoch,is_train=False)
                
    print(""" =============================================================== """)
    print("""             Generate the final evaluation results               """)
    print(""" =============================================================== """)

    outputs_pd = pd.DataFrame(outputs).T
    outputs_pd['3D EPE'] *= 100
    outputs_pd['axis error'] *= (180/np.pi)
    outputs_pd['trans error'] *= 100

    print(outputs_pd)

    return outputs_pd

def test_TrustRegion_A_classic(args,model_args):

    if args.time:
        timers = Timers()
    else:
        timers = None

    obj_has_mask = False
    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    print("Initialize and train the Deep Trust Region Network")

    print('Loading model: {:s}'.format(args.model))
    net = eval(args.model)

    if torch.cuda.is_available(): net.cuda()
    
    if args.checkpoint_Union !='':
        print('==============')
        net.load_state_dict(torch.load(args.checkpoint_Union), strict=False)

        print("=> Load depth model from checkpoint_Union")  
    
    logfile_name = 'eval_A'
    logger = train_utils.initialize_logger(args, logfile_name)

    train_objective = ['EPE3D']
    eval_objectives = ['EPE3D', 'RPE']

    outputs = {}
    outputs_d={}
    for test_name, testset in data_loader_test.items():

        eval_name = '{:}_{:}'.format(args.dataset, test_name)
        epoch = 0
        eval_pose_info = eval_utils.evaluate_trust_region_classic(
            testset, net, eval_objectives, epoch=epoch,
            device  = args.device, 
            eval_name   = eval_name,
            timers      = timers)
        outputs = pd.Series([eval_pose_info['epes'].mean(), 
            eval_pose_info['angular_error'].mean(), 
            eval_pose_info['translation_error'].mean(), 
            eval_pose_info['epes'].shape[0]], 
            index=['3D EPE', 'axis error', 'trans error', 'total frames'])
        
        display_pose_dict = {"{:}_epe3d".format(eval_name): eval_pose_info['epes'].mean(), 
            "{:}_rpe_angular".format(eval_name): eval_pose_info['angular_error'].mean(), 
            "{:}_rpe_translation".format(eval_name): eval_pose_info['translation_error'].mean()}

        logger.write_to_tensorboard(display_pose_dict, epoch)
        logger.write_to_terminal_val(display_pose_dict, epoch,is_train=False)
                
    print(""" =============================================================== """)
    print("""             Generate the final evaluation results               """)
    print(""" =============================================================== """)

    outputs_pd = pd.DataFrame(outputs).T
    outputs_pd['3D EPE'] *= 100
    outputs_pd['axis error'] *= (180/np.pi)
    outputs_pd['trans error'] *= 100

    print(outputs_pd)

    return outputs_pd

def test_TrustRegion_A_SPSG(args,model_args):

    if args.time:
        timers = Timers()
    else:
        timers = None

    obj_has_mask = False
    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    print("Initialize and train the Deep Trust Region Network")

    print('Loading model: {:s}'.format(args.model))
    net = eval(args.model)

    if torch.cuda.is_available(): net.cuda()
    
    if args.checkpoint_Union !='':
        print('==============')
        net.load_state_dict(torch.load(args.checkpoint_Union)['state_dict'], strict=False)
        print("=> Load depth model from checkpoint_Union",args.checkpoint_Union)  
    
    logfile_name = 'eval_A'
    logger = train_utils.initialize_logger(args, logfile_name)

    train_objective = ['EPE3D']
    eval_objectives = ['EPE3D', 'RPE']

    outputs = {}
    outputs_d={}
    for test_name, testset in data_loader_test.items():

        eval_name = '{:}_{:}'.format(args.dataset, test_name)
        epoch = 0
        
        eval_pose_info = eval_utils.evaluate_trust_region_SPSG(
            testset, net, eval_objectives, epoch=epoch,
            device  = args.device, 
            eval_name   = eval_name,
            timers      = timers)
        outputs = pd.Series([eval_pose_info['epes'].mean(), 
            eval_pose_info['angular_error'].mean(), 
            eval_pose_info['translation_error'].mean(), 
            eval_pose_info['epes'].shape[0]], 
            index=['3D EPE', 'axis error', 'trans error', 'total frames'])
        
        display_pose_dict = {"{:}_epe3d".format(eval_name): eval_pose_info['epes'].mean(), 
            "{:}_rpe_angular".format(eval_name): eval_pose_info['angular_error'].mean(), 
            "{:}_rpe_translation".format(eval_name): eval_pose_info['translation_error'].mean()}

        logger.write_to_tensorboard(display_pose_dict, epoch)
        logger.write_to_terminal_val(display_pose_dict, epoch,is_train=False)
                
    print(""" =============================================================== """)
    print("""             Generate the final evaluation results               """)
    print(""" =============================================================== """)

    outputs_pd = pd.DataFrame(outputs).T
    outputs_pd['3D EPE'] *= 100
    outputs_pd['axis error'] *= (180/np.pi)
    outputs_pd['trans error'] *= 100

    print(outputs_pd)

    return outputs_pd

def test_TrustRegion_A(args,model_args):

    if args.time:
        timers = Timers()
    else:
        timers = None

    obj_has_mask = False
    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    print("Initialize and train the Deep Trust Region Network")

    print('Loading model: {:s}'.format(args.model))
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
        net.load_state_dict(torch.load(args.checkpoint_Union, map_location=torch.device('cpu'))['state_dict'], strict=False)
        print("=> Load depth model from checkpoint_Union")  

    logfile_name = 'eval_A'
    logger = train_utils.initialize_logger(args, logfile_name)

    train_objective = ['EPE3D']
    eval_objectives = ['EPE3D', 'RPE']

    outputs = {}
    outputs_d={}
    for test_name, testset in data_loader_test.items():

        eval_name = '{:}_{:}'.format(args.dataset, test_name)
        epoch = 0
        eval_pose_info = eval_utils.evaluate_trust_region_A(
            testset, net, eval_objectives, args = args,
            eval_name   = eval_name,
            timers      = timers)
        
        outputs = pd.Series([eval_pose_info['epes'].mean(), 
            eval_pose_info['angular_error'].mean(), 
            eval_pose_info['translation_error'].mean(), 
            eval_pose_info['epes'].shape[0]], 
            index=['3D EPE', 'axis error', 'trans error', 'total frames'])
        
        display_pose_dict = {"{:}_epe3d".format(eval_name): eval_pose_info['epes'].mean(), 
            "{:}_rpe_angular".format(eval_name): eval_pose_info['angular_error'].mean(), 
            "{:}_rpe_translation".format(eval_name): eval_pose_info['translation_error'].mean()}

        logger.write_to_tensorboard(display_pose_dict, epoch)
        logger.write_to_terminal_val(display_pose_dict, epoch,is_train=False)
                
    print(""" =============================================================== """)
    print("""             Generate the final evaluation results               """)
    print(""" =============================================================== """)

    outputs_pd = pd.DataFrame(outputs).T
    outputs_pd['3D EPE'] *= 100
    outputs_pd['axis error'] *= (180/np.pi)
    outputs_pd['trans error'] *= 100

    print(outputs_pd)

    return outputs_pd


from LEM_SFM.config import get_model_args,get_args
from LEM_SFM.model_with_depth import DepthPoseNet_Deeplabv3,DepthPoseNet,SPSG,LoFTR_class,Classic

if __name__ == '__main__':
    print('=> Set seed...')
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print('=> Set args...')
    model_args = get_model_args()
    args = get_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.batch_per_gpu = 128
    
    # datasets
    # args.test_dataset="ARKitScenes(split='test', resolution=[(320,240)])"
    # args.test_dataset="1_000 @ ScanNet1500(resolution=(320, 240), seed=777)" 
    args.test_dataset="ScanNet1500(resolution=(320, 240), seed=777)" 

    noise_gyro = [0.1,0.48,0.03]
    noise_accel = [0.5,0.073,0.05]

    idx = 2
    args.noise_gyro = noise_gyro[idx-1]
    args.noise_accel = noise_accel[idx-1]
    args.dt = 0.1

    print('Using device : ', args.device)
    args.start_epoch = 0

    # models
    # args.model = 'DepthPoseNet_Deeplabv3(model_args)'
    args.model = 'DepthPoseNet(model_args)'

    # args.model = 'LoFTR_class()'
    # args.model = 'SPSG()'
    # args.model = "Reloc3rRelpose(img_size=512)"
    # args.model = 'Classic()'
    print('=> Set Model...', args.model)

    # checkpoints
    if 'DepthPoseNet' in args.model:
        args.require_IMU=True
        args.require_dicInput=False
        args.checkpoint_Union='checkpoint/checkpoint_LEM_SFM_S.pth.tar'
        # args.checkpoint_Union='checkpoint/checkpoint_LEM_SFM_L.pth.tar'

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


    if torch.cuda.device_count() > 1:
        print('---------------------------------------') 
        print("Use", torch.cuda.device_count(), "GPUs for evaluate!")

    print('---------------------------------------') 

    outputs = test_TrustRegion_A(args,model_args)
