import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.load_data(root)

    def load_data(self, root):
        files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.pkl')]
        for file in files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                R_est, t_est = data['R_est'], data['t_est']
                R_gt, t_gt = data['R_gt'], data['t_gt']
                R_IMU, t_IMU = data['R_IMU'], data['t_IMU']

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
        R_est = torch.tensor(sample['R_est'], dtype=torch.float32)
        t_est = torch.tensor(sample['t_est'], dtype=torch.float32)
        R_gt = torch.tensor(sample['R_gt'], dtype=torch.float32)
        t_gt = torch.tensor(sample['t_gt'], dtype=torch.float32)
        R_IMU = torch.tensor(sample['R_IMU'], dtype=torch.float32)
        t_IMU = torch.tensor(sample['t_IMU'], dtype=torch.float32)

        return R_est, t_est, R_gt, t_gt, R_IMU, t_IMU

class GateModel(nn.Module):
    def __init__(self):
        super(GateModel, self).__init__()
        self.fc1 = nn.Linear(24, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        weights_1 = x[:, :2]
        weights_2 = x[:, 2:]
        weights_1 = self.softmax(weights_1)
        weights_2 = self.softmax(weights_2)
        return torch.cat([weights_1, weights_2], dim=1)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for R_est, t_est, R_gt, t_gt, R_IMU, t_IMU in dataloader:
            R_est = R_est.to(device)
            t_est = t_est.to(device)
            R_gt = R_gt.to(device)
            t_gt = t_gt.to(device)
            R_IMU = R_IMU.to(device)
            t_IMU = t_IMU.to(device)

            R_est_flat = R_est.view(R_est.size(0), -1).float()
            R_IMU_flat = R_IMU.view(R_IMU.size(0), -1).float()
            t_est = t_est.float()
            t_IMU = t_IMU.float()

            x = torch.cat([R_est_flat, R_IMU_flat, t_est, t_IMU], dim=1)

            weights = model(x)
            w_R_est, w_R_IMU, w_t_est, w_t_IMU = weights[:, 0], weights[:, 1], weights[:, 2], weights[:, 3]

            R_pred = w_R_est.unsqueeze(1) * R_est_flat + w_R_IMU.unsqueeze(1) * R_IMU_flat
            t_pred = w_t_est.unsqueeze(1) * t_est + w_t_IMU.unsqueeze(1) * t_IMU

            loss_R = criterion(R_pred, R_gt.view(R_gt.size(0), -1))
            loss_t = criterion(t_pred, t_gt)
            total_loss += loss_R.item() + loss_t.item()

    return total_loss / len(dataloader)

def train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs, save_dir='./models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(epochs):
        model.train()
        eval_loss = evaluate(model, val_dataloader)
        print('===')
        print(f"Epoch {epoch + 1}/{epochs}, Evaluation Loss: {eval_loss}")

        for R_est, t_est, R_gt, t_gt, R_IMU, t_IMU in train_dataloader:
            R_est = R_est.to(device)
            t_est = t_est.to(device)
            R_gt = R_gt.to(device)
            t_gt = t_gt.to(device)
            R_IMU = R_IMU.to(device)
            t_IMU = t_IMU.to(device)

            R_est_flat = R_est.view(R_est.size(0), -1).float()
            R_IMU_flat = R_IMU.view(R_IMU.size(0), -1).float()
            t_est = t_est.float()
            t_IMU = t_IMU.float()

            x = torch.cat([R_est_flat, R_IMU_flat, t_est, t_IMU], dim=1)

            weights = model(x)
            w_R_est, w_R_IMU, w_t_est, w_t_IMU = weights[:, 0], weights[:, 1], weights[:, 2], weights[:, 3]

            R_pred = w_R_est.unsqueeze(1) * R_est_flat + w_R_IMU.unsqueeze(1) * R_IMU_flat
            t_pred = w_t_est.unsqueeze(1) * t_est + w_t_IMU.unsqueeze(1) * t_IMU

            loss_R = criterion(R_pred, R_gt.view(R_gt.size(0), -1))
            loss_t = criterion(t_pred, t_gt)
            loss = loss_R + loss_t

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        eval_loss = evaluate(model, val_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item()}, Evaluation Loss: {eval_loss}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth'))

if __name__ == '__main__':
    print('=> Set seed...')
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = '/media/pan/089614D79614C6DA/temp/Linux/code/LEM-SFM-package/LEM-main/logs/checkpoint_epoch151'
    dataset = TrajectoryDataset(root)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    for R_est, t_est, R_gt, t_gt, R_IMU, t_IMU in train_dataloader:
        print("R_est shape:", R_est.shape)
        print("t_est shape:", t_est.shape)
        print("R_gt shape:", R_gt.shape)
        print("t_gt shape:", t_gt.shape)
        print("R_IMU shape:", R_IMU.shape)
        print("t_IMU shape:", t_IMU.shape)
        break

    model = GateModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=100)
