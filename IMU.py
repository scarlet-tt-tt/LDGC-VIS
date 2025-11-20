import numpy as np
from scipy.spatial.transform import Rotation as R  # 确保正确导入 Rotation
import matplotlib.pyplot as plt
import torch

def generate_trajectory(trajectory_type, num_frames, dt):
    """生成不同类型的轨迹真值"""
    poses = []
    for i in range(num_frames):
        t = i * dt
        pose = np.eye(4)
        
        if trajectory_type == "line":
            # 直线轨迹
            translation = np.array([t, 0, 0])
            rotation = R.from_euler('z', 0).as_matrix()
        elif trajectory_type == "circle":
            # 圆形轨迹
            translation = np.array([np.cos(t), np.sin(t), 0])
            rotation = R.from_euler('z', t).as_matrix()
        elif trajectory_type == "spiral":
            # 螺旋轨迹
            translation = np.array([np.cos(t), np.sin(t)+t*0.1, 0])
            rotation = R.from_euler('z', t).as_matrix()
        elif trajectory_type == "wave":
            # 波浪轨迹
            translation = np.array([t, np.sin(t), 0])
            rotation = R.from_euler('z', np.sin(t)).as_matrix()

        elif trajectory_type == "square":
            # 正方形轨迹
            side_length = 2.0  # 正方形边长
            total_perimeter = 4 * side_length
            progress = (t % total_perimeter) / total_perimeter

            if progress < 0.25:
                # 第一边：从左下到右下
                x = -side_length / 2 + progress * 4 * side_length
                y = -side_length / 2
                angle = 0
            elif progress < 0.5:
                # 第二边：从右下到右上
                x = side_length / 2
                y = -side_length / 2 + (progress - 0.25) * 4 * side_length
                angle = np.pi / 2
            elif progress < 0.75:
                # 第三边：从右上到左上
                x = side_length / 2 - (progress - 0.5) * 4 * side_length
                y = side_length / 2
                angle = np.pi
            else:
                # 第四边：从左上到左下
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
    """计算相邻帧之间的相对位姿"""
    relative_poses = []
    for i in range(1, len(absolute_poses)):
        relative_pose = np.linalg.inv(absolute_poses[i - 1]) @ absolute_poses[i]
        relative_poses.append(relative_pose)
    return relative_poses

def simulate_imu(relative_poses, timestamps, noise_std_gyro=0.01, noise_std_accel=0.1):
    """模拟 IMU 输出"""
    imu_data = []
    for i in range(len(relative_poses)):
        dt = timestamps[i + 1] - timestamps[i]
        
        # 提取旋转矩阵和位移
        rotation_matrix = relative_poses[i][:3, :3]
        translation = relative_poses[i][:3, 3]
        
        # 计算角速度（从旋转矩阵提取角速度）
        rotation = R.from_matrix(rotation_matrix)
        angular_velocity = rotation.as_rotvec() / dt  # 角速度 = 旋转向量 / 时间间隔
        
        # 计算线加速度
        linear_acceleration = translation / dt**2  # 加速度 = 位移 / 时间间隔^2
        
        # 添加噪声
        angular_velocity += np.random.normal(0, noise_std_gyro, size=3)
        linear_acceleration += np.random.normal(0, noise_std_accel, size=3)
        
        imu_data.append({
            "timestamp": timestamps[i],
            "angular_velocity": angular_velocity,
            "linear_acceleration": linear_acceleration
        })
    return imu_data

def reconstruct_trajectory_from_imu(imu_data, initial_pose):
    """从 IMU 数据重建轨迹"""
    poses = [initial_pose]
    current_pose = initial_pose.copy()
    for i in range(len(imu_data)):
        dt = imu_data[i + 1]["timestamp"] - imu_data[i]["timestamp"] if i + 1 < len(imu_data) else 0.1
        angular_velocity = imu_data[i]["angular_velocity"]
        linear_acceleration = imu_data[i]["linear_acceleration"]
        
        # 计算旋转增量
        rotation_increment = R.from_rotvec(angular_velocity * dt).as_matrix()
        
        # 更新旋转
        current_pose[:3, :3] = current_pose[:3, :3] @ rotation_increment
        
        # 更新平移
        translation_increment = linear_acceleration * dt**2
        current_pose[:3, 3] += current_pose[:3, :3] @ translation_increment
        
        poses.append(current_pose.copy())
    return poses

def visualize_trajectories_with_rotation(ground_truth, reconstructed):
    """可视化轨迹对比并显示旋转"""
    gt_positions = np.array([pose[:3, 3] for pose in ground_truth])
    rec_positions = np.array([pose[:3, 3] for pose in reconstructed])
    gt_orientations = [pose[:3, :3] for pose in ground_truth]
    
    plt.figure(figsize=(10, 6))
    plt.plot(gt_positions[:, 0], gt_positions[:, 1], label="Ground Truth", marker='o')
    plt.plot(rec_positions[:, 0], rec_positions[:, 1], label="Reconstructed", marker='x')
    
    # 在轨迹上绘制旋转方向
    for i in range(0, len(gt_positions), max(1, len(gt_positions) // 20)):
        position = gt_positions[i]
        orientation = gt_orientations[i]
        direction = orientation @ np.array([0.1, 0, 0])  # 绘制 x 轴方向
        plt.arrow(position[0], position[1], direction[0], direction[1], color='r', head_width=0.05)
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory Comparison with Rotation")
    plt.legend()
    plt.grid()
    plt.savefig('imu')

def simulate_relative_pose_from_imu(pose1, pose2, dt=0.04, noise_std_gyro=0.01, noise_std_accel=0.1):
    """
    输入两帧绝对位姿，模拟 IMU 数据并输出相对位姿
    :param pose1: 第一帧的绝对位姿 (4x4 numpy array)
    :param pose2: 第二帧的绝对位姿 (4x4 numpy array)
    :param dt: 两帧之间的时间间隔
    :param noise_std_gyro: 角速度噪声标准差
    :param noise_std_accel: 加速度噪声标准差
    :return: 模拟的相对位姿 (4x4 numpy array)
    """
    # 计算真实相对位姿
    # relative_pose = np.linalg.inv(pose1) @ pose2
    relative_pose = np.linalg.inv(pose2) @ pose1
    # 提取旋转矩阵和位移
    rotation_matrix = relative_pose[:3, :3]
    translation = relative_pose[:3, 3]

    # 计算真实角速度和加速度
    rotation = R.from_matrix(rotation_matrix)
    angular_velocity = rotation.as_rotvec() / dt  # 角速度 = 旋转向量 / 时间间隔
    linear_acceleration = translation / dt**2  # 加速度 = 位移 / 时间间隔^2

    # 添加噪声
    angular_velocity += np.random.normal(0, noise_std_gyro, size=3)
    linear_acceleration += np.random.normal(0, noise_std_accel, size=3)

    # 使用噪声角速度和加速度重建相对位姿
    noisy_rotation_matrix = R.from_rotvec(angular_velocity * dt).as_matrix()
    noisy_translation = linear_acceleration * dt**2

    noisy_relative_pose = np.eye(4)
    noisy_relative_pose[:3, :3] = noisy_rotation_matrix
    noisy_relative_pose[:3, 3] = noisy_translation

    return noisy_relative_pose


import numpy as np
import torch
from scipy.spatial.transform import Rotation as R  # 确保正确导入 Rotation

def simulate_pose_from_imu(R_tensor, t_tensor, dt=0.04, noise_std_gyro=0.1, noise_std_accel=0.5):
    """
    输入两帧相对位姿，模拟 IMU 数据并输出相对位姿
    :param R_tensor: R=(batch, 3, 3), 批量旋转矩阵 (PyTorch Tensor)
    :param t_tensor: t=(batch, 3), 批量平移向量 (PyTorch Tensor)
    :param noise_std_gyro: 角速度噪声标准差
    :param noise_std_accel: 加速度噪声标准差
    :return: 模拟的相对位姿 [R, t], R=(batch, 3, 3), t=(batch, 3)
    """
    # 提取旋转矩阵和位移，并转换为 NumPy 格式
    rotation_matrix = R_tensor.cpu().numpy()  # 将张量移到 CPU 并转换为 NumPy 数组
    translation = t_tensor.cpu().numpy()  # 同样处理位移

    # 计算真实角速度和加速度
    rotations = R.from_matrix(rotation_matrix)  # 批量转换为旋转对象
    angular_velocities = rotations.as_rotvec() / dt  # 角速度 = 旋转向量 / 时间间隔
    linear_accelerations = translation / dt**2  # 加速度 = 位移 / 时间间隔^2

    # 添加噪声
    angular_velocities += np.random.normal(0, noise_std_gyro, size=angular_velocities.shape)
    linear_accelerations += np.random.normal(0, noise_std_accel, size=linear_accelerations.shape)

    # 使用噪声角速度和加速度重建相对位姿
    noisy_rotation_matrices = R.from_rotvec(angular_velocities * dt).as_matrix()  # 批量计算旋转矩阵
    noisy_translations = linear_accelerations * dt**2  # 批量计算位移

    # 转换回 PyTorch 张量并返回
    return (
        torch.tensor(noisy_rotation_matrices, dtype=R_tensor.dtype).to(R_tensor.device),
        torch.tensor(noisy_translations, dtype=t_tensor.dtype).to(t_tensor.device),
    )

def simulate_pose_from_imu_A(pose1, pose2, dt=0.04, noise_std_gyro=0.1, noise_std_accel=0.5):
    # print(pose1.device)
    pose1_np = pose1.cpu().numpy() 
    pose2_np = pose2.cpu().numpy()
    # 计算真实相对位姿
    # relative_pose = np.linalg.inv(pose1_np) @ pose2_np
    relative_pose = np.linalg.inv(pose2_np) @ pose1_np
    # 提取旋转矩阵和位移
    rotation_matrix = relative_pose[:,:3, :3]
    translation = relative_pose[:,:3, 3]

    # # 提取旋转矩阵和位移，并转换为 NumPy 格式
    # rotation_matrix = R_tensor.cpu().numpy()  # 将张量移到 CPU 并转换为 NumPy 数组
    # translation = t_tensor.cpu().numpy()  # 同样处理位移

    # 计算真实角速度和加速度
    rotations = R.from_matrix(rotation_matrix)  # 批量转换为旋转对象
    angular_velocities = rotations.as_rotvec() / dt  # 角速度 = 旋转向量 / 时间间隔
    linear_accelerations = translation / dt**2  # 加速度 = 位移 / 时间间隔^2

    # 添加噪声
    angular_velocities += np.random.normal(0, noise_std_gyro, size=angular_velocities.shape)
    linear_accelerations += np.random.normal(0, noise_std_accel, size=linear_accelerations.shape)

    # 使用噪声角速度和加速度重建相对位姿
    noisy_rotation_matrices = R.from_rotvec(angular_velocities * dt).as_matrix()  # 批量计算旋转矩阵
    noisy_translations = linear_accelerations * dt**2  # 批量计算位移

    # 转换回 PyTorch 张量并返回
    return (
        torch.tensor(noisy_rotation_matrices, dtype=pose1.dtype).to(pose1.device),
        torch.tensor(noisy_translations, dtype=pose1.dtype).to(pose1.device),
    )

def plot_trajectories_with_simulation():
    """绘制 circle, spiral, wave 三种轨迹及其模拟轨迹，并保存图像"""
    trajectory_types = ["circle", "spiral", "wave",'square']
    num_frames = 200
    dt = 0.1

    # 创建子图
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for i, trajectory_type in enumerate(trajectory_types):
        # 生成轨迹真值
        ground_truth_poses, timestamps = generate_trajectory(trajectory_type, num_frames, dt)

        # 计算相对位姿
        relative_poses = compute_relative_pose(ground_truth_poses)

        # 模拟 IMU 数据
        imu_data = simulate_imu(relative_poses, timestamps, noise_std_gyro=0.1, noise_std_accel=0.5)

        # 从 IMU 数据重建轨迹
        reconstructed_poses = reconstruct_trajectory_from_imu(imu_data, ground_truth_poses[0])

        # 提取轨迹位置
        gt_positions = np.array([pose[:3, 3] for pose in ground_truth_poses])
        rec_positions = np.array([pose[:3, 3] for pose in reconstructed_poses])

        # 绘制真值轨迹
        axes[i].plot(gt_positions[:, 0], gt_positions[:, 1], label=f"{trajectory_type.capitalize()} Ground Truth", marker='o')

        # 绘制模拟轨迹
        axes[i].plot(rec_positions[:, 0], rec_positions[:, 1], label=f"{trajectory_type.capitalize()} Simulated", marker='x')

        # 设置子图标题和标签
        axes[i].set_title(f"{trajectory_type.capitalize()} Trajectory")
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        axes[i].grid()
        axes[i].legend()

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig("imu.png")
    plt.show()

# 调用绘图函数
if __name__ == "__main__":
    plot_trajectories_with_simulation()
    




# # Demo1
# if __name__ == "__main__":
#     # 选择轨迹类型
#     trajectory_type = "wave"  # 可选："line", "circle", "spiral", "wave"
#     num_frames = 100
#     dt = 0.1
    
#     # 生成轨迹真值
#     ground_truth_poses, timestamps = generate_trajectory(trajectory_type, num_frames, dt)
    
#     # 计算相对位姿
#     relative_poses = compute_relative_pose(ground_truth_poses)
    
#     # 模拟 IMU 数据
#     imu_data = simulate_imu(relative_poses, timestamps)
    
#     # 从 IMU 数据重建轨迹
#     reconstructed_poses = reconstruct_trajectory_from_imu(imu_data, ground_truth_poses[0])
    
#     # 可视化轨迹对比并显示旋转
#     visualize_trajectories_with_rotation(ground_truth_poses, reconstructed_poses)

# # # Demo2
# if __name__ == "__main__":
#     # 示例两帧绝对位姿
#     pose1 = np.eye(4)
#     pose2 = np.array([[0.866, -0.5, 0, 1],
#                       [0.5, 0.866, 0, 0],
#                       [0, 0, 1, 0],
#                       [0, 0, 0, 1]])
#     dt = 0.1  # 时间间隔

#     # 模拟相对位姿
#     noisy_relative_pose = simulate_relative_pose_from_imu(pose1, pose2, dt)

#     print("真实相对位姿:")
#     print(np.linalg.inv(pose1) @ pose2)
#     print("\n模拟的相对位姿:")
#     print(noisy_relative_pose)