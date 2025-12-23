# train_n_physound.py - 自动生成
# experiments/train_n_physound.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from models.n_physound import N_PHYSound
from utils.data_loader import UnderwaterSoundDataset
from utils.metrics import accuracy, cross_env_accuracy


def train_n_physound(config):
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集和数据加载器
    train_dataset = UnderwaterSoundDataset(
        config['train_data_path'],
        config['train_labels_path']
    )
    val_dataset = UnderwaterSoundDataset(
        config['val_data_path'],
        config['val_labels_path']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # 模型初始化
    model = N_PHYSound(num_classes=config['num_classes']).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}"):
            signals, labels = batch
            signals = signals.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs, phys_params, consistency = model(signals)

            # 计算损失 (主损失 + 物理一致性正则化)
            cls_loss = criterion(outputs, labels)
            # 鼓励高一致性分数
            phys_reg = 0.1 * torch.mean(1 - consistency)
            total_loss = cls_loss + phys_reg

            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 计算指标
            train_loss += total_loss.item()
            train_acc += accuracy(outputs, labels)

        # 平均训练指标
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                signals, labels = batch
                signals = signals.to(device)
                labels = labels.to(device)

                outputs, _, _ = model(signals)
                val_loss += criterion(outputs, labels).item()
                val_acc += accuracy(outputs, labels)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        # 打印 epoch 结果
        print(f"Epoch {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'], 'best_n_physound.pth'))

    print(f"Training complete. Best Val Accuracy: {best_val_acc:.4f}")