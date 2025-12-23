# physics_guided.py - 自动生成
import numpy as np
import torch
from physics.underwater_acoustics import UnderwaterAcousticModel


class PhysicsGuidedPreprocessor:
    """物理引导的预处理类，结合水声物理模型优化输入信号"""

    def __init__(self, sample_rate=16000, normalize=True):
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.phys_model = UnderwaterAcousticModel()

    def _align_sound_speed(self, audio_signal):
        """基于声速调整信号时序，补偿不同水温下的声速差异"""
        # 计算信号的主要频率成分
        fft_result = np.fft.fft(audio_signal)
        freq = np.fft.fftfreq(len(audio_signal), 1 / self.sample_rate)
        main_freq = abs(freq[np.argmax(abs(fft_result))])

        # 基于主频率调整信号，模拟声速变化的影响
        adjusted_signal = audio_signal * (self.phys_model.sound_speed / 1500) ** 0.5
        return adjusted_signal

    def _attenuation_compensation(self, audio_signal, estimated_distance=100):
        """对信号进行衰减补偿，还原原始信号强度"""
        # 反向计算传播损失
        frequency = np.mean(np.abs(np.fft.fftfreq(len(audio_signal), 1 / self.sample_rate)))
        spreading_loss = 20 * np.log10(estimated_distance)
        absorption_loss = self.phys_model.attenuation_coeff * estimated_distance * frequency
        total_loss = 10 ** ((spreading_loss + absorption_loss) / 20)

        # 补偿衰减
        compensated_signal = audio_signal * total_loss
        return compensated_signal

    def _remove_reverberation(self, audio_signal):
        """简单去除水下混响干扰（基于谱减法）"""
        # 估计噪声（取信号前10%作为噪声段）
        noise_segment = audio_signal[:int(len(audio_signal) * 0.1)]
        noise_fft = np.fft.fft(noise_segment)
        noise_power = np.abs(noise_fft) ** 2

        # 信号FFT
        signal_fft = np.fft.fft(audio_signal)
        signal_power = np.abs(signal_fft) ** 2

        # 谱减法去混响
        enhanced_power = np.maximum(signal_power - noise_power, 1e-8)
        enhanced_fft = np.sqrt(enhanced_power) * np.exp(1j * np.angle(signal_fft))
        enhanced_signal = np.fft.ifft(enhanced_fft).real
        return enhanced_signal

    def process(self, audio_data):
        """
        物理引导预处理完整流程
        :param audio_data: 输入音频（numpy数组或torch张量）
        :return: 预处理后的音频
        """
        # 类型转换：torch张量转numpy
        if isinstance(audio_data, torch.Tensor):
            is_tensor = True
            audio_signal = audio_data.cpu().detach().numpy()
        else:
            is_tensor = False
            audio_signal = audio_data.copy()

        # 步骤1：声速对齐
        aligned_signal = self._align_sound_speed(audio_signal)

        # 步骤2：衰减补偿
        compensated_signal = self._attenuation_compensation(aligned_signal)

        # 步骤3：去混响
        enhanced_signal = self._remove_reverberation(compensated_signal)

        # 步骤4：归一化
        if self.normalize:
            enhanced_signal = enhanced_signal / (np.max(np.abs(enhanced_signal)) + 1e-8)

        # 类型还原：numpy转torch张量
        if is_tensor:
            enhanced_signal = torch.tensor(enhanced_signal, dtype=audio_data.dtype, device=audio_data.device)

        return enhanced_signal