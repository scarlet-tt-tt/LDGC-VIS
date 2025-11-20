import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 定义数据集类
class TrajectoryDataset(Dataset):
    def __init__(self, root):
        self.samples = []  # 存储每个样本
        self.load_data(root)

    def load_data(self, root):
        files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.pkl')]
        for file in files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                R_est, t_est = data['R_est'], data['t_est']
                R_gt, t_gt = data['R_gt'], data['t_gt']
                R_IMU, t_IMU = data['R_IMU'], data['t_IMU']

                # 将每个时间步的 R_est、t_est、R_IMU、t_IMU 作为一个样本
                for i in range(len(t_est)):
                    sample = {
                        'R_est': R_est[i],
                        't_est': t_est[i],
                        'R_gt': R_gt[i],
                        't_gt': t_gt[i],
                        'R_IMU': R_IMU[i],
                        't_IMU': t_IMU[i]
                    }
                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        R_est = torch.tensor(sample['R_est'], dtype=torch.float32)  # 转换为 float32
        t_est = torch.tensor(sample['t_est'], dtype=torch.float32)  # 转换为 float32
        R_gt = torch.tensor(sample['R_gt'], dtype=torch.float32)    # 转换为 float32
        t_gt = torch.tensor(sample['t_gt'], dtype=torch.float32)    # 转换为 float32
        R_IMU = torch.tensor(sample['R_IMU'], dtype=torch.float32)  # 转换为 float32
        t_IMU = torch.tensor(sample['t_IMU'], dtype=torch.float32)  # 转换为 float32

        return R_est, t_est, R_gt, t_gt, R_IMU, t_IMU

# 定义门控机制模型
class GateModel(nn.Module):
    def __init__(self):
        super(GateModel, self).__init__()
        self.fc1 = nn.Linear(24, 128)  # 输入特征维度为12（旋转矩阵和平移向量）
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 4)  # 输出4个权重（分别用于R_est和R_IMU的旋转与平移选择）
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        # 分割权重
        weights_1 = x[:, :2]  # 前两个权重
        weights_2 = x[:, 2:]  # 后两个权重
        
        # 分别应用 softmax
        weights_1 = self.softmax(weights_1)
        weights_2 = self.softmax(weights_2)
        
        # 拼接权重
        return torch.cat([weights_1, weights_2], dim=1)



# 修改评估函数以使用验证集
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for R_est, t_est, R_gt, t_gt, R_IMU, t_IMU in dataloader:
            # 将数据移动到 GPU
            R_est = R_est.to(device)
            t_est = t_est.to(device)
            R_gt = R_gt.to(device)
            t_gt = t_gt.to(device)
            R_IMU = R_IMU.to(device)
            t_IMU = t_IMU.to(device)

            # 数据预处理：拼接特征
            R_est_flat = R_est.view(R_est.size(0), -1).float()  # 转换为 float32
            R_IMU_flat = R_IMU.view(R_IMU.size(0), -1).float()  # 转换为 float32
            t_est = t_est.float()  # 转换为 float32
            t_IMU = t_IMU.float()  # 转换为 float32

            x = torch.cat([R_est_flat, R_IMU_flat, t_est, t_IMU], dim=1)

            weights = model(x)
            w_R_est, w_R_IMU, w_t_est, w_t_IMU = weights[:, 0], weights[:, 1], weights[:, 2], weights[:, 3]

            R_pred = w_R_est.unsqueeze(1) * R_est_flat + w_R_IMU.unsqueeze(1) * R_IMU_flat
            t_pred = w_t_est.unsqueeze(1) * t_est + w_t_IMU.unsqueeze(1) * t_IMU

            loss_R = criterion(R_pred, R_gt.view(R_gt.size(0), -1))
            loss_t = criterion(t_pred, t_gt)
            total_loss += loss_R.item() + loss_t.item()

    return total_loss / len(dataloader)

# 修改训练函数以使用训练集和验证集
def train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs, save_dir='./models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(epochs):
        model.train()
        eval_loss = evaluate(model, val_dataloader)
        print('===')
        print(f"Epoch {epoch + 1}/{epochs}, Evaluation Loss: {eval_loss}")

        for R_est, t_est, R_gt, t_gt, R_IMU, t_IMU in train_dataloader:
            # 将数据移动到 GPU
            R_est = R_est.to(device)
            t_est = t_est.to(device)
            R_gt = R_gt.to(device)
            t_gt = t_gt.to(device)
            R_IMU = R_IMU.to(device)
            t_IMU = t_IMU.to(device)

            # 数据预处理：拼接特征
            R_est_flat = R_est.view(R_est.size(0), -1).float()  # 转换为 float32
            R_IMU_flat = R_IMU.view(R_IMU.size(0), -1).float()  # 转换为 float32
            t_est = t_est.float()  # 转换为 float32
            t_IMU = t_IMU.float()  # 转换为 float32

            x = torch.cat([R_est_flat, R_IMU_flat, t_est, t_IMU], dim=1)

            # 模型预测
            weights = model(x)
            w_R_est, w_R_IMU, w_t_est, w_t_IMU = weights[:, 0], weights[:, 1], weights[:, 2], weights[:, 3]

            # 加权结果
            R_pred = w_R_est.unsqueeze(1) * R_est_flat + w_R_IMU.unsqueeze(1) * R_IMU_flat
            t_pred = w_t_est.unsqueeze(1) * t_est + w_t_IMU.unsqueeze(1) * t_IMU

            # 计算损失
            loss_R = criterion(R_pred, R_gt.view(R_gt.size(0), -1))
            loss_t = criterion(t_pred, t_gt)
            loss = loss_R + loss_t

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每个 epoch 结束后评估模型
        eval_loss = evaluate(model, val_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item()}, Evaluation Loss: {eval_loss}")

        # 保存模型参数
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth'))

if __name__ == '__main__':
    print('=> Set seed...')
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载
    root = '/media/pan/089614D79614C6DA/temp/Linux/code/LEM-SFM-package/LEM-main/logs/checkpoint_epoch151'
    dataset = TrajectoryDataset(root)
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))  # 80% 用于训练
    val_size = len(dataset) - train_size  # 20% 用于验证
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)


    # 测试数据加载
    for R_est, t_est, R_gt, t_gt, R_IMU, t_IMU in train_dataloader:
        print("R_est shape:", R_est.shape)
        print("t_est shape:", t_est.shape)
        print("R_gt shape:", R_gt.shape)
        print("t_gt shape:", t_gt.shape)
        print("R_IMU shape:", R_IMU.shape)
        print("t_IMU shape:", t_IMU.shape)
        break

    # 模型、损失函数和优化器
    model = GateModel().to(device)  # 将模型移动到 GPU
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 运行训练与评估
    train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=100)

