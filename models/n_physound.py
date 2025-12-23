# models/n_physound.py
import torch
import torch.nn as nn
import torch.optim as optim
from physics.underwater_acoustics import UnderwaterAcousticModel
from preprocessing.physics_guided import PhysicsGuidedPreprocessor


class N_PHYSound(nn.Module):
    def __init__(self, num_classes=10):
        super(N_PHYSound, self).__init__()

        # 物理模型组件
        self.phys_model = UnderwaterAcousticModel()
        self.phys_preprocessor = PhysicsGuidedPreprocessor()

        # 特征学习网络
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # 物理参数预测网络 (用于反演)
        self.param_predictor = nn.Sequential(
            nn.Linear(128 * 125, 512),  # 根据输入尺寸调整
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3)  # 预测距离、深度、频率
        )

        # 分类头
        self.classifier = nn.Linear(128 * 125 + 1, num_classes)  # +1 加入一致性分数

    def forward(self, x):
        # 1. 物理引导预处理
        x = self.phys_preprocessor.process(x)
        x = x.unsqueeze(1)  # 添加通道维度

        # 2. 特征提取
        features = self.feature_extractor(x)
        batch_size, channels, length = features.shape
        features_flat = features.view(batch_size, -1)

        # 3. 物理参数预测 (反演)
        physics_params = self.param_predictor(features_flat)  # [distance, depth, frequency]

        # 4. 物理模拟与一致性计算
        consistency_scores = []
        for i in range(batch_size):
            # 模拟信号
            simulated = self.phys_model.simulate_propagation(
                x[i].cpu().detach().numpy(),
                distance=physics_params[i, 0].item(),
                depth=physics_params[i, 1].item(),
                frequency=physics_params[i, 2].item()
            )

            # 计算一致性
            score = self.phys_model.calculate_consistency(
                x[i].cpu().detach().numpy(),
                simulated
            )
            consistency_scores.append(score)

        consistency = torch.tensor(consistency_scores, device=x.device).unsqueeze(1)

        # 5. 结合特征与一致性分数进行分类
        combined = torch.cat([features_flat, consistency], dim=1)
        output = self.classifier(combined)

        return output, physics_params, consistency