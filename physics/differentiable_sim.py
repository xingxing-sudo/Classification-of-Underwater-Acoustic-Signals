# differentiable_sim.py - 自动生成
import torch
import torch.nn as nn
import numpy as np


class DifferentiableUnderwaterSimulator(nn.Module):
    """可微分水下声波传播模拟器，支持反向传播优化"""

    def __init__(self, sample_rate=16000):
        super(DifferentiableUnderwaterSimulator, self).__init__()
        self.sample_rate = sample_rate
        # 可学习的物理参数（初始值设为典型水下环境参数）
        self.sound_speed = nn.Parameter(torch.tensor(1500.0, dtype=torch.float32))
        self.attenuation_coeff = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.water_density = nn.Parameter(torch.tensor(1025.0, dtype=torch.float32))

    def _calculate_propagation_loss(self, distance, frequency):
        """可微分计算传播损失"""
        # 球面扩展损失
        spreading_loss = 20 * torch.log10(distance + 1e-8)  # 防止log(0)
        # 吸收损失
        absorption_loss = self.attenuation_coeff * distance * frequency
        # 总损失（转换为衰减系数）
        total_loss = torch.pow(10.0, -(spreading_loss + absorption_loss) / 20.0)
        return total_loss

    def _calculate_time_delay(self, distance):
        """可微分计算传播时间延迟"""
        propagation_time = distance / (self.sound_speed + 1e-8)
        return propagation_time

    def _differentiable_roll(self, x, shift):
        """可微分的信号移位操作（替代torch.roll，支持反向传播）"""
        batch_size, channels, length = x.shape
        shift = shift.to(x.device)

        # 创建移位掩码
        indices = torch.arange(length, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        shifted_indices = (indices - shift.unsqueeze(1)) % length

        # 重新排列信号
        x_rolled = torch.gather(x, 2, shifted_indices.unsqueeze(1).repeat(1, channels, 1))
        return x_rolled

    def forward(self, source_signal, distance, depth, frequency):
        """
        可微分模拟水下声波传播
        :param source_signal: 原始信号 [batch, 1, length]
        :param distance: 传播距离 [batch]
        :param depth: 传播深度 [batch]
        :param frequency: 信号主频率 [batch]
        :return: 接收信号 [batch, 1, length]
        """
        batch_size = source_signal.shape[0]

        # 1. 计算传播损失并应用衰减
        attenuation = self._calculate_propagation_loss(distance, frequency)
        attenuated_signal = source_signal * attenuation.view(batch_size, 1, 1)

        # 2. 计算传播延迟并应用移位
        propagation_time = self._calculate_time_delay(distance)
        shift = (propagation_time * self.sample_rate).round().long()
        delayed_signal = self._differentiable_roll(attenuated_signal, shift)

        # 3. 深度影响（简单建模：深度越大，信号噪声越高）
        depth_noise = torch.randn_like(delayed_signal) * (depth.view(batch_size, 1, 1) / 1000.0)
        final_signal = delayed_signal + depth_noise

        return final_signal