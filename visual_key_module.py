import argparse
import torch
from LEM_SFM.model_with_depth import DepthPoseNet_Deeplabv3_plot
import numpy as np
from cv2 import resize, INTER_NEAREST
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import os
from tqdm import tqdm

# from LEM_SFM.relative_depth.depth_anything.dpt import DepthAnything
from IMU import simulate_pose_from_imu

from LEM_SFM.config import get_model_args,get_args


def plot_batch_FWM_subplots(dim_x,dim_y,color0,color1,feature0_infer,feature1_infer,weight1_infer,name):

    fig, axs = plt.subplots(dim_x, dim_y,figsize = (15,20))
    # print(img_list.shape)
    # print(feature_infer_list1.shape)
    # print(feature_infer_list2.shape)
    # print(x0_before.shape)
    # print(depth_infer.shape)

    # for i in range(3):
    for j in range(dim_x):
        # j=7*i+j
        num = j
        axs[j,0].imshow(color0[num].squeeze(0).cpu().detach().numpy().transpose(1,2,0), cmap='viridis')
        axs[j,0].axis('off')   
        # figname = 'min:'+str(img_list[num].min().item())+'max:'+str(img_list[num].max().item())
        # axs[j, 0].set_title(figname)

        axs[j,1].imshow(torch.sigmoid(feature0_infer[num]).squeeze(0).squeeze(0).cpu().detach(), cmap='coolwarm')
        axs[j,1].axis('off')   

        axs[j,2].imshow(color1[num].squeeze(0).cpu().detach().numpy().transpose(1,2,0), cmap='viridis')
        axs[j,2].axis('off')  


        axs[j,3].imshow(torch.sigmoid(feature1_infer[num]).squeeze(0).squeeze(0).cpu().detach(), cmap='coolwarm')
        axs[j,3].axis('off')  

        axs[j,4].imshow(weight1_infer[num].squeeze(0).squeeze(0).cpu().detach(), cmap='coolwarm')
        axs[j,4].axis('off')  


    plt.savefig(name+'.png')

def plot_batch_FWM(args,model_args):
    # ==============================================1
    from evaluate_A import build_dataset
    
    obj_has_mask = False
    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    print("Initialize and train the Deep Trust Region Network")

    print('Loading model: {:s}'.format(args.model))
    net = eval(args.model)

    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # if torch.cuda.is_available(): net.cuda()
    net.to(device=args.device)

    if args.checkpoint_Union !='':
        print('==============')
        net.load_state_dict(torch.load(args.checkpoint_Union)['state_dict'], strict=False)
        print("=> Load depth model from checkpoint_Union")  
    

    for test_name, dataloader in data_loader_test.items():

        eval_name = '{:}_{:}'.format(args.dataset, test_name)

        epoch = 0
        if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'set_epoch'):
            dataloader.dataset.set_epoch(epoch)
        if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)


        progress = tqdm(dataloader, ncols=100,
            desc = 'evaluate deeper inverse compositional algorithm {:}'.format(eval_name),
            total= len(dataloader))
        
    # 遍历数据加载器    
        for batch_idx, batch in enumerate(progress):


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
            from evaluate import intrinsics_matrix_to_k
            K = intrinsics_matrix_to_k(view0['camera_intrinsics'])
            pose1_np = view0['camera_pose'].cpu().numpy() 
            pose2_np = view1['camera_pose'].cpu().numpy()

            Rt = np.linalg.inv(pose2_np) @ pose1_np

            Rt = torch.tensor(Rt).to(args.device)

            invalid_mask_0 = (depth0_gt == depth0_gt.min()) + (depth0_gt == depth0_gt.max())

            B, _, H, W = depth0_gt.shape


            with torch.no_grad():
                R_gt, t_gt = Rt[:,:3,:3], Rt[:,:3,3]

                s_pose0 = simulate_pose_from_imu(R_gt, t_gt,dt=0.8)
                pose,depth0,depth1,feature0_infer,feature1_infer,weight1_infer = net.forward(color0, color1, K,s_pose0)
            # plot_batch_subplots( 10,5,color1,depth1,feature1_infer,weight1_infer,depth1_gt,str(batch_idx))

            name = 'FWM_ScanNet_test'+str(batch_idx)
            plot_batch_FWM_subplots( 8,5,color0,color1,feature0_infer,feature1_infer,weight1_infer,name)
            if batch_idx>25:
                break

def plot_batch_FWM(args,model_args,max_num,datasets):
    # ==============================================1
    from evaluate_A import build_dataset
    
    obj_has_mask = False
    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    print("Initialize and train the Deep Trust Region Network")

    print('Loading model: {:s}'.format(args.model))
    net = eval(args.model)

    num_gpus = torch.cuda.device_count()

    # if torch.cuda.is_available(): net.cuda()
    net.to(device=args.device)

    if args.checkpoint_Union !='':
        print('==============')
        net.load_state_dict(torch.load(args.checkpoint_Union, map_location=torch.device('cpu'))['state_dict'], strict=False)
        print("=> Load depth model from checkpoint_Union")  
    

    for test_name, dataloader in data_loader_test.items():

        eval_name = '{:}_{:}'.format(args.dataset, test_name)

        epoch = 0
        if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'set_epoch'):
            dataloader.dataset.set_epoch(epoch)
        if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)


        progress = tqdm(dataloader, ncols=100,
            desc = 'evaluate deeper inverse compositional algorithm {:}'.format(eval_name),
            total= len(dataloader))
        
        for batch_idx, batch in enumerate(progress):


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
            from evaluate import intrinsics_matrix_to_k
            K = intrinsics_matrix_to_k(view0['camera_intrinsics'])
            pose1_np = view0['camera_pose'].cpu().numpy() 
            pose2_np = view1['camera_pose'].cpu().numpy()

            Rt = np.linalg.inv(pose2_np) @ pose1_np

            Rt = torch.tensor(Rt).to(args.device)

            with torch.no_grad():
                R_gt, t_gt = Rt[:,:3,:3], Rt[:,:3,3]

                s_pose0 = simulate_pose_from_imu(R_gt, t_gt,dt=0.8)
                pose,depth0,depth1,feature0_infer,feature1_infer,weight1_infer = net.forward(color0, color1, K,s_pose0)

            name = f"data/visual_FWM/FWM_{datasets}_test{batch_idx}"
            plot_batch_FWM_subplots( 8,5,color0,color1,feature0_infer,feature1_infer,weight1_infer,name)
            if batch_idx>=max_num-1:
                break


def plot_FWM(max_num,datasets):

    print('=> Set seed...')
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print('=> Set args...')
    model_args = get_model_args()
    args = get_args()

    # print(feature_args.data_root)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.lr = 10*1e-5
    args.batch_size = 10
    
    if datasets =='ScanNet1500':
        args.test_dataset="1_000 @ ScanNet1500(resolution=(320, 240), seed=777)" 
    elif datasets =='ARKitScenes':
        args.test_dataset="1_000 @ ARKitScenes(split='test', resolution=[(320,240)])"

    print('Using device : ', args.device)
    args.start_epoch = 0

    args.model = 'DepthPoseNet_Deeplabv3_plot(model_args)'

    print('=> Set checkpoints...')
    args.checkpoint_Union = 'checkpoint/checkpoint_LEM_SFM_L.pth.tar'

    if torch.cuda.device_count() > 1:
        print('---------------------------------------') 
        print("Use", torch.cuda.device_count(), "GPUs for evaluate!")

    print('---------------------------------------') 

    plot_batch_FWM(args,model_args,max_num,datasets)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Key module visualization')
    parser.add_argument('--max_num', type=int, default=10)
    parser.add_argument('--datasets', type=str, default='ScanNet1500',choices=['ScanNet1500','ARKitScenes'])
    args = parser.parse_args()
    plot_FWM(args.max_num,args.datasets)
