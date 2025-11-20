import torch
import cv2
import os
import matplotlib.pyplot as plt

def infer_pose(img_path1,img_path2,output_folder):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Set model
    from LEM_SFM.model_with_depth import DepthPoseNet,DepthPoseNet_Deeplabv3
    from LEM_SFM.config import get_model_args
    model_args = get_model_args()
    net = DepthPoseNet_Deeplabv3(model_args).to(device)
    # or
    # net = DepthPoseNet(model_args).to(device)
    net.eval()

    # Set checkpoints
    args.checkpoint_Union = 'checkpoint/checkpoint_LEM_SFM_L.pth.tar'
    # or
    # args.checkpoint_Union = 'checkpoint/checkpoint_LEM_SFM_S.pth.tar'

    if args.checkpoint_Union !='':
        net.load_state_dict(torch.load(args.checkpoint_Union)['state_dict'])
        print("=> Load depth model")  

    # input
    img_size = (320,240)

    img1 = cv2.imread(img_path1)
    img1 = cv2.resize(img1, img_size)
    img1_tensor = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img1_tensor = img1_tensor.to(device)

    img2 = cv2.imread(img_path2)
    img2 = cv2.resize(img2, img_size)
    img2_tensor = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img2_tensor = img2_tensor.to(device)

    K=[157.27,157.24,160.75,130.015]
    K_tensor = torch.tensor(K, dtype=torch.float32).unsqueeze(0).cuda()

    R_tensor = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)
    t_tensor = torch.zeros((1, 3), dtype=torch.float32, device=device)
    pose_init = [R_tensor,t_tensor]

    # infer
    pose,_ = net(img1_tensor,img2_tensor,K_tensor,pose_init)

    # save
    output_path = os.path.join(output_folder, 'pose.txt')
    with open(output_path, 'w') as f:
        for p in pose:
            if isinstance(p, torch.Tensor):
                np_array = p.detach().cpu().numpy()
                for row in np_array:
                    f.write(' '.join(map(str, row)) + '\n')
            else:
                f.write(str(p) + '\n')
    print(f"Pose saved to {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='infer depth from image')
    parser.add_argument('--img_path1', type=str, default='data/TUM_fr3_sitting_halfsphere_1.png')
    parser.add_argument('--img_path2', type=str, default='data/TUM_fr3_sitting_halfsphere_2.png')
    parser.add_argument('--output_folder', type=str, default='data/')
    args = parser.parse_args()
    infer_pose(img_path1 = args.img_path1,img_path2 = args.img_path2 , output_folder = args.output_folder)

