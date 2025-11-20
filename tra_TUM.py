import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
from LEM_SFM.train_gate import GateModel  

# Convert relative poses to 3D trajectory points
def compute_trajectory(R, t):
    trajectory = [np.zeros(3)]  # Initial point is the origin
    current_pose = np.eye(4)  # Initial pose is the identity matrix
    for i in range(len(t)):
        # Construct the current transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R[i]
        transform[:3, 3] = t[i]
        # Update the current pose
        current_pose = current_pose @ transform
        # Extract position
        trajectory.append(current_pose[:3, 3])
    return np.array(trajectory)


def plot_combined(root1, root2, pkl_files1, pkl_files2):
    # Create plot
    fig, axes = plt.subplots(3, len(pkl_files1), figsize=(15, 8))  # Three rows of subplots

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GateModel().to(device)  # Move model to GPU
    # Load pre-trained model
    path = 'checkpoint/gate.pth'
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"load_state_dict: {path}")

    # Process each trajectory file
    for idx, (pkl_file1, pkl_file2) in enumerate(zip(pkl_files1, pkl_files2)):
        # Process files from root1 (IMU and Estimated data)
        file_path1 = os.path.join(root1, pkl_file1)
        with open(file_path1, 'rb') as f:
            data1 = pickle.load(f)
    
        R_est, t_est = data1['R_est'], data1['t_est']
        R_gt, t_gt = data1['R_gt'], data1['t_gt']
        R_IMU, t_IMU = data1['R_IMU'], data1['t_IMU']
    
        # Move data to GPU
        R_est = torch.tensor(R_est, dtype=torch.float32).to(device)
        t_est = torch.tensor(t_est, dtype=torch.float32).to(device)
        R_gt = torch.tensor(R_gt, dtype=torch.float32).to(device)
        t_gt = torch.tensor(t_gt, dtype=torch.float32).to(device)
        R_IMU = torch.tensor(R_IMU, dtype=torch.float32).to(device)
        t_IMU = torch.tensor(t_IMU, dtype=torch.float32).to(device)
    
        # Data preprocessing: concatenate features
        R_est_flat = R_est.view(R_est.size(0), -1).float()
        R_IMU_flat = R_IMU.view(R_IMU.size(0), -1).float()
        t_est = t_est.float()
        t_IMU = t_IMU.float()
    
        x = torch.cat([R_est_flat, R_IMU_flat, t_est, t_IMU], dim=1)
    
        # Model prediction
        weights = model(x)
        w_R_est, w_R_IMU, w_t_est, w_t_IMU = weights[:, 0], weights[:, 1], weights[:, 2], weights[:, 3]
    
        # Weighted results
        R_pred = w_R_est.unsqueeze(1) * R_est_flat + w_R_IMU.unsqueeze(1) * R_IMU_flat
        t_pred = w_t_est.unsqueeze(1) * t_est + w_t_IMU.unsqueeze(1) * t_IMU
    
        R_pred = R_pred.cpu().detach().numpy()
        t_pred = t_pred.cpu().detach().numpy()
    
        R_pred = R_pred.reshape(-1, 3, 3)
        R_gt = R_gt.cpu().detach().numpy().reshape(-1, 3, 3)
        R_IMU = R_IMU.cpu().detach().numpy().reshape(-1, 3, 3)
        t_pred = t_pred.reshape(-1, 3)
        t_gt = t_gt.cpu().detach().numpy().reshape(-1, 3)
        t_IMU = t_IMU.cpu().detach().numpy().reshape(-1, 3)
    
        # Compute trajectories
        trajectory_est = compute_trajectory(R_pred, t_pred)
        trajectory_gt = compute_trajectory(R_gt, t_gt)
        trajectory_IMU = compute_trajectory(R_IMU, t_IMU)
    
        lenth = min(200, len(trajectory_gt))
    
        # First row of subplots: IMU vs Ground Truth
        ax1 = axes[0, idx]

        ax1.plot(trajectory_gt[:lenth, 0], trajectory_gt[:lenth, 1], label='Ground Truth', color='blue')
        ax1.plot(trajectory_IMU[:lenth, 0], trajectory_IMU[:lenth, 1], label='IMU', color='red')
        for i in range(lenth):
            if i % 2 == 0:
                ax1.plot([trajectory_gt[i, 0], trajectory_IMU[i, 0]],
                         [trajectory_gt[i, 1], trajectory_IMU[i, 1]], color='gray', alpha=0.25)
        if idx == 0:
            ax1.set_ylabel('IMU Y(m)', fontsize=12)
            # ax1.set_title('IMU', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelrotation=90, labelsize=8.5)
        ax1.tick_params(axis='x', labelsize=8.5)
        name = pkl_file1.split('dataset_')[1].split('_keyframe')[0]
        name = name.replace('freiburg1', 'freiburg1\n')
        name = name.replace('freiburg2', 'freiburg2\n')
        name = name.replace('freiburg3', 'freiburg3\n')
        name = name.replace('_', ' ')
        ax1.set_title(f'{name}', fontsize=12)
    
        # Third row of subplots: Estimated vs Ground Truth
        ax3 = axes[2, idx]
        ax3.plot(trajectory_gt[:lenth, 0], trajectory_gt[:lenth, 1], label='Ground Truth', color='blue')
        ax3.plot(trajectory_est[:lenth, 0], trajectory_est[:lenth, 1], label='Estimated', color='red')
        for i in range(lenth):
            if i % 2 == 0:
                ax3.plot([trajectory_gt[i, 0], trajectory_est[i, 0]],
                         [trajectory_gt[i, 1], trajectory_est[i, 1]], color='gray', alpha=0.25)
        if idx == 0:
            ax3.set_ylabel('LEM-SFM Y(m)', fontsize=12)
        ax3.tick_params(axis='y', labelrotation=90, labelsize=8.5)
        ax3.tick_params(axis='x', labelsize=8.5)
    
        # Second row of subplots: ORB vs Ground Truth
        file_path2 = os.path.join(root2, pkl_file2)
        with open(file_path2, 'rb') as f:
            data2 = pickle.load(f)
    
        R_gt, t_gt = data2['R_gt'], data2['t_gt']
        R_orb, t_orb = data2['R_est'], data2['t_est']
    
        trajectory_gt = compute_trajectory(R_gt, t_gt)
        trajectory_orb = compute_trajectory(R_orb, t_orb)
    
        lenth = min(100, len(trajectory_gt))
    
        ax2 = axes[1, idx]
        ax2.plot(trajectory_gt[:lenth, 0], trajectory_gt[:lenth, 1], label='Ground Truth', color='blue')
        ax2.plot(trajectory_orb[:lenth, 0], trajectory_orb[:lenth, 1], label='ORB', color='red')
        for i in range(lenth):
            if i % 4 == 0:
                ax2.plot([trajectory_gt[i, 0], trajectory_orb[i, 0]],
                         [trajectory_gt[i, 1], trajectory_orb[i, 1]], color='gray', alpha=0.25)
        if idx == 0:
            ax2.set_ylabel('SIFT+RANSAC Y(m)', fontsize=12)

        
        ax2.tick_params(axis='y', labelrotation=90, labelsize=8.5)
        ax2.tick_params(axis='x', labelsize=8.5)
    
        # if idx == len(pkl_files1) - 1:
        ax3.set_xlabel('X(m)', fontsize=12)  # Second row shows X-axis label

        # Set sparser intervals for y-axis ticks
        ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        ax3.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

        # Calculate coordinate range for the current column
        col_x_min = min(trajectory_gt[:lenth, 0].min(), trajectory_IMU[:lenth, 0].min(), trajectory_est[:lenth, 0].min())
        col_x_max = max(trajectory_gt[:lenth, 0].max(), trajectory_IMU[:lenth, 0].max(), trajectory_est[:lenth, 0].max())
        col_y_min = min(trajectory_gt[:lenth, 1].min(), trajectory_IMU[:lenth, 1].min(), trajectory_est[:lenth, 1].min())
        col_y_max = max(trajectory_gt[:lenth, 1].max(), trajectory_IMU[:lenth, 1].max(), trajectory_est[:lenth, 1].max())
    
        # Unify axis range for the current column and ensure equal aspect ratio for x and y axes
        col_x_range = col_x_max - col_x_min
        col_y_range = col_y_max - col_y_min
        max_range = max(col_x_range, col_y_range)
        ax1.set_xlim(col_x_min, col_x_min + max_range)
        ax1.set_ylim(col_y_min, col_y_min + max_range)
        ax2.set_xlim(col_x_min, col_x_min + max_range)
        ax2.set_ylim(col_y_min, col_y_min + max_range)
        ax3.set_xlim(col_x_min, col_x_min + max_range)
        ax3.set_ylim(col_y_min, col_y_min + max_range)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(root1, 'combined_trajectories.png'), dpi=300)
    print('Combined trajectory plot saved.')


if __name__ == "__main__":

    # Define folder paths
    root1 = 'raw_data/tra_TUM'
    root2 = 'raw_data/tra_SIFT'

    # Get all .pkl files in the folders
    pkl_files1 = [f for f in os.listdir(root1) if f.endswith('.pkl')]
    pkl_files2 = [f for f in os.listdir(root2) if f.endswith('.pkl')]

    plot_combined(root1,root2, pkl_files1, pkl_files2)