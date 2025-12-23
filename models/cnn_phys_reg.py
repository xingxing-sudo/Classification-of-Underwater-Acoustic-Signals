# cnn_phys_reg.py - 自动生成
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from models.cnn_models import ResNet18
from physics.consistency import PhysicsConsistencyEvaluator
from physics.differentiable_sim import DifferentiableUnderwaterSimulator


class CNNPhysRegModel(BaseModel):
    """CNN+物理正则化模型，在传统CNN基础上加入物理约束"""

    def __init__(self, num_classes=10, in_channels=1):
        super(CNNPhysRegModel, self).__init__(num_classes)

        # 基础CNN特征提取器（使用ResNet18）
        self.cnn_backbone = ResNet18(num_classes=num_classes, in_channels=in_channels)

        # 物理参数预测分支
        self.phys_param_predictor = nn.Sequential(
            nn.Linear(512, 256),  # ResNet18最后一层特征维度为512
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # 预测distance, depth, frequency
        )

        # 可微分物理模拟器
        self.differentiable_sim = DifferentiableUnderwaterSimulator()

        # 物理一致性评估器
        self.consistency_evaluator = PhysicsConsistencyEvaluator()

        # 融合层（结合CNN特征和物理一致性分数）
        self.fusion_layer = nn.Linear(512 + 1, num_classes)

    def forward(self, x, original_audio=None):
        """
        前向传播
        :param x: 频谱图输入 [batch, in_channels, h, w]
        :param original_audio: 原始音频信号 [batch, length]（用于物理模拟，可选）
        :return: 分类结果, 物理参数, 一致性分数
        """
        # 1. CNN特征提取
        cnn_features = self.cnn_backbone.features(x)  # [batch, 512, 1, 1]
        cnn_features_flat = torch.flatten(cnn_features, 1)  # [batch, 512]

        # 2. 物理参数预测
        phys_params = self.phys_param_predictor(cnn_features_flat)  # [batch, 3]
        distance, depth, frequency = phys_params[:, 0], phys_params[:, 1], phys_params[:, 2]

        # 3. 物理模拟与一致性计算
        consistency_score = torch.ones(x.shape[0], 1, device=self.device)
        if original_audio is not None:
            # 原始音频reshape为 [batch, 1, length]
            audio_reshaped = original_audio.unsqueeze(1)

            # 可微分物理模拟
            simulated_audio = self.differentiable_sim(
                audio_reshaped, distance, depth, frequency
            )

            # 计算物理一致性分数（可微分）
            consistency_score = self.consistency_evaluator.calculate_consistency(
                audio_reshaped.squeeze(1),
                simulated_audio.squeeze(1),
                use_torch=True
            ).unsqueeze(1)

        # 4. 特征融合与分类
        fused_features = torch.cat([cnn_features_flat, consistency_score], dim=1)
        output = self.fusion_layer(fused_features)

        return output, phys_params, consistency_score

    def get_physical_regularization_loss(self, original_audio, phys_params):
        """计算物理正则化损失，鼓励模型符合物理规律"""
        # 1. 物理模拟
        audio_reshaped = original_audio.unsqueeze(1)
        simulated_audio = self.differentiable_sim(
            audio_reshaped,
            phys_params[:, 0],
            phys_params[:, 1],
            phys_params[:, 2]
        )

        # 2. 计算一致性损失（1 - 一致性分数）
        consistency_score = self.consistency_evaluator.calculate_consistency(
            audio_reshaped.squeeze(1),
            simulated_audio.squeeze(1),
            use_torch=True
        )
        phys_reg_loss = torch.mean(1 - consistency_score)

        # 3. 物理参数约束损失（确保参数在合理范围内）
        distance = phys_params[:, 0]
        depth = phys_params[:, 1]
        frequency = phys_params[:, 2]

        # 距离约束：>0
        distance_loss = torch.mean(F.relu(-distance))
        # 深度约束：0~10000
        depth_loss = torch.mean(F.relu(-depth) + F.relu(depth - 10000))
        # 频率约束：20~100000
        freq_loss = torch.mean(F.relu(-frequency + 20) + F.relu(frequency - 100000))

        # 总物理正则化损失
        total_reg_loss = phys_reg_loss + 0.1 * (distance_loss + depth_loss + freq_loss)
        return total_reg_loss