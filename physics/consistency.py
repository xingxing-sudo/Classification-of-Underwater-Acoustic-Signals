# consistency.py - 自动生成
import torch
import numpy as np
from scipy.signal import correlate
from physics.underwater_acoustics import UnderwaterAcousticModel


class PhysicsConsistencyEvaluator:
    """物理一致性评估器，量化观测信号与物理模型的匹配程度"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.phys_model = UnderwaterAcousticModel()

    def _spectral_similarity(self, signal1, signal2, use_torch=False):
        """计算频谱相似度（支持numpy和torch）"""
        if use_torch:
            # torch版本（可微分）
            fft1 = torch.fft.fft(signal1, dim=-1)
            fft2 = torch.fft.fft(signal2, dim=-1)

            amp1 = torch.abs(fft1)
            amp2 = torch.abs(fft2)
            phase1 = torch.angle(fft1)
            phase2 = torch.angle(fft2)

            # 幅度谱相关系数
            amp_mean1 = torch.mean(amp1, dim=-1, keepdim=True)
            amp_mean2 = torch.mean(amp2, dim=-1, keepdim=True)
            amp_cov = torch.mean((amp1 - amp_mean1) * (amp2 - amp_mean2), dim=-1)
            amp_std1 = torch.std(amp1, dim=-1)
            amp_std2 = torch.std(amp2, dim=-1)
            amp_corr = amp_cov / (amp_std1 * amp_std2 + 1e-8)

            # 相位谱相关系数（转换为余弦相似度）
            phase_cos = torch.mean(torch.cos(phase1 - phase2), dim=-1)
            phase_corr = (phase_cos + 1) / 2  # 归一化到[0,1]

            return (amp_corr + phase_corr) / 2
        else:
            # numpy版本（非可微分，用于评估）
            fft1 = np.fft.fft(signal1)
            fft2 = np.fft.fft(signal2)

            amp1 = np.abs(fft1)
            amp2 = np.abs(fft2)
            phase1 = np.angle(fft1)
            phase2 = np.angle(fft2)

            # 幅度谱相关系数
            amp_corr = np.corrcoef(amp1, amp2)[0, 1]
            # 相位谱余弦相似度
            phase_cos = np.mean(np.cos(phase1 - phase2))
            phase_corr = (phase_cos + 1) / 2

            return (amp_corr + phase_corr) / 2

    def _temporal_similarity(self, signal1, signal2, use_torch=False):
        """计算时域相似度（互相关）"""
        if use_torch:
            # torch版本
            signal1 = signal1 - torch.mean(signal1, dim=-1, keepdim=True)
            signal2 = signal2 - torch.mean(signal2, dim=-1, keepdim=True)

            corr = torch.nn.functional.conv1d(
                signal1.unsqueeze(1),
                signal2.unsqueeze(1).flip(dims=[2]),
                padding=signal1.shape[-1] - 1
            )
            max_corr = torch.max(corr, dim=-1)[0].squeeze(1)

            # 归一化
            norm1 = torch.norm(signal1, dim=-1)
            norm2 = torch.norm(signal2, dim=-1)
            temporal_corr = max_corr / (norm1 * norm2 + 1e-8)
            return temporal_corr
        else:
            # numpy版本
            signal1 = signal1 - np.mean(signal1)
            signal2 = signal2 - np.mean(signal2)

            corr = correlate(signal1, signal2, mode='full')
            max_corr = np.max(corr)

            # 归一化
            norm1 = np.linalg.norm(signal1)
            norm2 = np.linalg.norm(signal2)
            temporal_corr = max_corr / (norm1 * norm2 + 1e-8)
            return temporal_corr

    def _physical_constraint_check(self, signal, phys_params):
        """检查信号是否符合物理约束（如能量守恒、频率范围等）"""
        distance, depth, frequency = phys_params

        # 1. 能量约束：信号能量随距离增加而衰减
        expected_energy = 1.0 / (distance ** 2 + 1e-8)
        actual_energy = np.sum(signal ** 2)
        energy_consistency = np.exp(-np.abs(actual_energy - expected_energy) / expected_energy)

        # 2. 频率约束：水下声波频率一般在20Hz~100kHz之间
        freq_valid = 1.0 if (frequency >= 20 and frequency <= 100000) else 0.1

        # 3. 深度约束：深度过深会导致信号失真
        depth_penalty = np.exp(-depth / 1000.0)

        return (energy_consistency + freq_valid + depth_penalty) / 3

    def calculate_consistency(self, observed_signal, simulated_signal, phys_params=None, use_torch=False):
        """
        计算综合物理一致性分数
        :param observed_signal: 观测信号
        :param simulated_signal: 物理模型模拟信号
        :param phys_params: 物理参数 [distance, depth, frequency]（可选）
        :param use_torch: 是否使用torch张量计算（可微分）
        :return: 一致性分数 [0,1]
        """
        # 1. 频谱相似度
        spectral_score = self._spectral_similarity(observed_signal, simulated_signal, use_torch)

        # 2. 时域相似度
        temporal_score = self._temporal_similarity(observed_signal, simulated_signal, use_torch)

        # 3. 物理约束检查（若提供物理参数）
        if phys_params is not None and not use_torch:
            constraint_score = self._physical_constraint_check(observed_signal, phys_params)
        else:
            constraint_score = 1.0 if not use_torch else torch.tensor(1.0, device=observed_signal.device)

        # 综合分数
        total_score = (0.5 * spectral_score + 0.3 * temporal_score + 0.2 * constraint_score)

        # 裁剪到[0,1]范围
        if use_torch:
            total_score = torch.clamp(total_score, 0.0, 1.0)
        else:
            total_score = np.clip(total_score, 0.0, 1.0)

        return total_score