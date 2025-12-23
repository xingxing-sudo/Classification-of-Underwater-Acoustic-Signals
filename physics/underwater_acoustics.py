# physics/underwater_acoustics.py
import numpy as np
from scipy.interpolate import interp1d


class UnderwaterAcousticModel:
    """水声物理模型，用于模拟不同条件下的声波传播"""

    def __init__(self):
        # 物理参数设置
        self.sound_speed = 1500  # 水中声速 (m/s)
        self.attenuation_coeff = 0.1  # 衰减系数
        self.water_density = 1025  # 水密度 (kg/m?)

    def simulate_propagation(self, source_signal, distance, depth, frequency):
        """
        模拟声波在水下的传播

        参数:
            source_signal: 原始信号
            distance: 传播距离 (m)
            depth: 深度 (m)
            frequency: 信号频率 (Hz)

        返回:
            received_signal: 传播后的信号
        """
        # 计算传播损失
        spreading_loss = 20 * np.log10(distance)  # 球面扩展损失
        absorption_loss = self.attenuation_coeff * distance * frequency  # 吸收损失
        total_loss = 10 ** (-(spreading_loss + absorption_loss) / 20)

        # 计算传播时间
        propagation_time = distance / self.sound_speed

        # 信号衰减应用
        attenuated_signal = source_signal * total_loss

        # 模拟延迟 (简化版)
        delayed_signal = np.roll(attenuated_signal, int(propagation_time * 16000))

        return delayed_signal

    def calculate_consistency(self, observed_signal, simulated_signal):
        """计算观测信号与模拟信号的物理一致性"""
        # 计算频谱相似度
        obs_fft = np.fft.fft(observed_signal)
        sim_fft = np.fft.fft(simulated_signal)

        # 计算幅度谱相似度
        amp_similarity = np.corrcoef(np.abs(obs_fft), np.abs(sim_fft))[0, 1]

        # 计算相位谱相似度 (简化)
        phase_similarity = np.corrcoef(np.angle(obs_fft), np.angle(sim_fft))[0, 1]

        # 综合一致性分数
        consistency_score = (amp_similarity + phase_similarity) / 2

        return np.clip(consistency_score, 0, 1)