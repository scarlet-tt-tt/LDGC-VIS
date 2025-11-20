import torch
import cv2
import os
import matplotlib.pyplot as plt

def infer_depth(img_path,output_folder):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Set model
    import LEM_SFM.DM.starnet_src.midas_starnet as DM
    args.model = 'DM.MidasNet_small7()'
    net = eval(args.model).to(device)

    # Set checkpoints
    args.checkpoint_Union = 'checkpoint/checkpoint_DM.pth'

    if args.checkpoint_Union !='':
        net.load_state_dict(torch.load(args.checkpoint_Union))
        print("=> Load depth model")  

    # input
    depth_size = (320,256)
    img1 = cv2.imread(img_path)
    img1 = cv2.resize(img1, depth_size)
    img1_tensor = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img1_tensor = img1_tensor.to(device)

    # infer
    depth = net(img1_tensor)

    # save
    depth = depth.squeeze().cpu().detach().numpy()

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_file_name = f"{base_name}_depth.png"
    output_path = os.path.join(output_folder, output_file_name)
    plt.imsave(output_path, depth, cmap='viridis')
    print(f"Depth map saved to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='infer depth from image')
    parser.add_argument('--img_path', type=str, default='data/7Scenes_fire.png')
    parser.add_argument('--output_folder', type=str, default='data/')
    args = parser.parse_args()
    infer_depth(img_path = args.img_path , output_folder = args.output_folder)

