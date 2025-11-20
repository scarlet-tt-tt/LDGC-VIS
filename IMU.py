import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import torch

def generate_trajectory(trajectory_type, num_frames, dt):
    """Generate ground truth for different types of trajectories"""
    poses = []
    for i in range(num_frames):
        t = i * dt
        pose = np.eye(4)
        
        if trajectory_type == "line":
            translation = np.array([t, 0, 0])
            rotation = R.from_euler('z', 0).as_matrix()
        elif trajectory_type == "circle":
            translation = np.array([np.cos(t), np.sin(t), 0])
            rotation = R.from_euler('z', t).as_matrix()
        elif trajectory_type == "spiral":
            translation = np.array([np.cos(t), np.sin(t)+t*0.1, 0])
            rotation = R.from_euler('z', t).as_matrix()
        elif trajectory_type == "wave":
            translation = np.array([t, np.sin(t), 0])
            rotation = R.from_euler('z', np.sin(t)).as_matrix()

        elif trajectory_type == "square":
            side_length = 2.0
            total_perimeter = 4 * side_length
            progress = (t % total_perimeter) / total_perimeter

            if progress < 0.25:
                x = -side_length / 2 + progress * 4 * side_length
                y = -side_length / 2
                angle = 0
            elif progress < 0.5:
                x = side_length / 2
                y = -side_length / 2 + (progress - 0.25) * 4 * side_length
                angle = np.pi / 2
            elif progress < 0.75:
                x = side_length / 2 - (progress - 0.5) * 4 * side_length
                y = side_length / 2
                angle = np.pi
            else:
                x = -side_length / 2
                y = side_length / 2 - (progress - 0.75) * 4 * side_length
                angle = 3 * np.pi / 2

            translation = np.array([x, y, 0])
            rotation = R.from_euler('z', angle).as_matrix()
        else:
            raise ValueError("Unsupported trajectory type")
        
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        poses.append(pose)
    
    return poses, np.arange(0, num_frames * dt, dt)

def compute_relative_pose(absolute_poses):
    """Compute relative poses between adjacent frames"""
    relative_poses = []
    for i in range(1, len(absolute_poses)):
        relative_pose = np.linalg.inv(absolute_poses[i - 1]) @ absolute_poses[i]
        relative_poses.append(relative_pose)
    return relative_poses

def simulate_imu(relative_poses, timestamps, noise_std_gyro=0.01, noise_std_accel=0.1):
    """Simulate IMU output"""
    imu_data = []
    for i in range(len(relative_poses)):
        dt = timestamps[i + 1] - timestamps[i]
        
        rotation_matrix = relative_poses[i][:3, :3]
        translation = relative_poses[i][:3, 3]
        
        rotation = R.from_matrix(rotation_matrix)
        angular_velocity = rotation.as_rotvec() / dt
        
        linear_acceleration = translation / dt**2
        
        angular_velocity += np.random.normal(0, noise_std_gyro, size=3)
        linear_acceleration += np.random.normal(0, noise_std_accel, size=3)
        
        imu_data.append({
            "timestamp": timestamps[i],
            "angular_velocity": angular_velocity,
            "linear_acceleration": linear_acceleration
        })
    return imu_data

def reconstruct_trajectory_from_imu(imu_data, initial_pose):
    """Reconstruct trajectory from IMU data"""
    poses = [initial_pose]
    current_pose = initial_pose.copy()
    for i in range(len(imu_data)):
        dt = imu_data[i + 1]["timestamp"] - imu_data[i]["timestamp"] if i + 1 < len(imu_data) else 0.1
        angular_velocity = imu_data[i]["angular_velocity"]
        linear_acceleration = imu_data[i]["linear_acceleration"]
        
        rotation_increment = R.from_rotvec(angular_velocity * dt).as_matrix()
        
        current_pose[:3, :3] = current_pose[:3, :3] @ rotation_increment
        
        translation_increment = linear_acceleration * dt**2
        current_pose[:3, 3] += current_pose[:3, :3] @ translation_increment
        
        poses.append(current_pose.copy())
    return poses

def visualize_trajectories_with_rotation(ground_truth, reconstructed):
    """Visualize trajectory comparison with rotation"""
    gt_positions = np.array([pose[:3, 3] for pose in ground_truth])
    rec_positions = np.array([pose[:3, 3] for pose in reconstructed])
    gt_orientations = [pose[:3, :3] for pose in ground_truth]
    
    plt.figure(figsize=(10, 6))
    plt.plot(gt_positions[:, 0], gt_positions[:, 1], label="Ground Truth", marker='o')
    plt.plot(rec_positions[:, 0], rec_positions[:, 1], label="Reconstructed", marker='x')
    
    for i in range(0, len(gt_positions), max(1, len(gt_positions) // 20)):
        position = gt_positions[i]
        orientation = gt_orientations[i]
        direction = orientation @ np.array([0.1, 0, 0])
        plt.arrow(position[0], position[1], direction[0], direction[1], color='r', head_width=0.05)
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory Comparison with Rotation")
    plt.legend()
    plt.grid()
    plt.savefig('imu')

def simulate_relative_pose_from_imu(pose1, pose2, dt=0.04, noise_std_gyro=0.01, noise_std_accel=0.1):
    relative_pose = np.linalg.inv(pose2) @ pose1
    rotation_matrix = relative_pose[:3, :3]
    translation = relative_pose[:3, 3]

    rotation = R.from_matrix(rotation_matrix)
    angular_velocity = rotation.as_rotvec() / dt
    linear_acceleration = translation / dt**2

    angular_velocity += np.random.normal(0, noise_std_gyro, size=3)
    linear_acceleration += np.random.normal(0, noise_std_accel, size=3)

    noisy_rotation_matrix = R.from_rotvec(angular_velocity * dt).as_matrix()
    noisy_translation = linear_acceleration * dt**2

    noisy_relative_pose = np.eye(4)
    noisy_relative_pose[:3, :3] = noisy_rotation_matrix
    noisy_relative_pose[:3, 3] = noisy_translation

    return noisy_relative_pose

def simulate_pose_from_imu(R_tensor, t_tensor, dt=0.04, noise_std_gyro=0.1, noise_std_accel=0.5):
    rotation_matrix = R_tensor.cpu().numpy()
    translation = t_tensor.cpu().numpy()

    rotations = R.from_matrix(rotation_matrix)
    angular_velocities = rotations.as_rotvec() / dt
    linear_accelerations = translation / dt**2

    angular_velocities += np.random.normal(0, noise_std_gyro, size=angular_velocities.shape)
    linear_accelerations += np.random.normal(0, noise_std_accel, size=linear_accelerations.shape)

    noisy_rotation_matrices = R.from_rotvec(angular_velocities * dt).as_matrix()
    noisy_translations = linear_accelerations * dt**2

    return (
        torch.tensor(noisy_rotation_matrices, dtype=R_tensor.dtype).to(R_tensor.device),
        torch.tensor(noisy_translations, dtype=t_tensor.dtype).to(t_tensor.device),
    )

def simulate_pose_from_imu_A(pose1, pose2, dt=0.04, noise_std_gyro=0.1, noise_std_accel=0.5):
    pose1_np = pose1.cpu().numpy() 
    pose2_np = pose2.cpu().numpy()
    relative_pose = np.linalg.inv(pose2_np) @ pose1_np
    rotation_matrix = relative_pose[:,:3, :3]
    translation = relative_pose[:,:3, 3]

    rotations = R.from_matrix(rotation_matrix)
    angular_velocities = rotations.as_rotvec() / dt
    linear_accelerations = translation / dt**2

    angular_velocities += np.random.normal(0, noise_std_gyro, size=angular_velocities.shape)
    linear_accelerations += np.random.normal(0, noise_std_accel, size=linear_accelerations.shape)

    noisy_rotation_matrices = R.from_rotvec(angular_velocities * dt).as_matrix()
    noisy_translations = linear_accelerations * dt**2

    return (
        torch.tensor(noisy_rotation_matrices, dtype=pose1.dtype).to(pose1.device),
        torch.tensor(noisy_translations, dtype=pose1.dtype).to(pose1.device),
    )

def plot_trajectories_with_simulation():
    """Plot circle, spiral, wave, square trajectories and their simulated trajectories"""
    trajectory_types = ["circle", "spiral", "wave", "square"]
    num_frames = 200
    dt = 0.1

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for i, trajectory_type in enumerate(trajectory_types):
        ground_truth_poses, timestamps = generate_trajectory(trajectory_type, num_frames, dt)
        relative_poses = compute_relative_pose(ground_truth_poses)
        imu_data = simulate_imu(relative_poses, timestamps, noise_std_gyro=0.1, noise_std_accel=0.5)
        reconstructed_poses = reconstruct_trajectory_from_imu(imu_data, ground_truth_poses[0])

        gt_positions = np.array([pose[:3, 3] for pose in ground_truth_poses])
        rec_positions = np.array([pose[:3, 3] for pose in reconstructed_poses])

        axes[i].plot(gt_positions[:, 0], gt_positions[:, 1], label=f"{trajectory_type.capitalize()} Ground Truth", marker='o')
        axes[i].plot(rec_positions[:, 0], rec_positions[:, 1], label=f"{trajectory_type.capitalize()} Simulated", marker='x')

        axes[i].set_title(f"{trajectory_type.capitalize()} Trajectory")
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        axes[i].grid()
        axes[i].legend()

    plt.tight_layout()
    plt.savefig("imu.png")
    plt.show()

if __name__ == "__main__":
    plot_trajectories_with_simulation()
