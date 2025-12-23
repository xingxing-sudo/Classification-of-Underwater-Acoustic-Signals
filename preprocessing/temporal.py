# preprocessing/temporal.py
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa  # 可能需要添加这个导入


class TemporalPreprocessor:
    def __init__(self, sample_rate=16000, cutoff_freq=4000):
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        self.filter = self._design_filter()

    def _design_filter(self):
        """设计低通滤波器"""
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = self.cutoff_freq / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        return b, a

    def _resample(self, audio, original_sr, target_sr):
        """
        重采样音频

        Args:
            audio: 音频信号
            original_sr: 原始采样率
            target_sr: 目标采样率

        Returns:
            重采样后的音频
        """
        if original_sr == target_sr:
            return audio

        # 计算重采样长度
        num_samples = int(len(audio) * target_sr / original_sr)
        resampled = signal.resample(audio, num_samples)

        return resampled

    def load_audio(self, file_path, sr=None):
        """
        加载音频文件

        Args:
            file_path: 音频文件路径
            sr: 目标采样率（如果为None则使用self.sample_rate）

        Returns:
            音频信号和采样率
        """
        target_sr = sr or self.sample_rate

        try:
            # 加载音频
            audio, original_sr = sf.read(file_path)

            # 如果是立体声，转换为单声道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # 重采样
            if original_sr != target_sr:
                audio = self._resample(audio, original_sr, target_sr)

            return audio, target_sr

        except Exception as e:
            # 如果sf.read失败，尝试librosa
            try:
                audio, original_sr = librosa.load(file_path, sr=None, mono=True)
                if original_sr != target_sr:
                    audio = self._resample(audio, original_sr, target_sr)
                return audio, target_sr
            except Exception as e2:
                raise RuntimeError(f"无法加载音频文件 {file_path}: {e}, {e2}")

    def process(self, audio_path):
        """完整预处理流程：加载->滤波->归一化->去噪"""
        # 使用 load_audio 方法
        audio, sr = self.load_audio(audio_path)

        # 滤波
        filtered = signal.lfilter(self.filter[0], self.filter[1], audio)

        # 归一化
        normalized = filtered / np.max(np.abs(filtered))

        # 简单去噪 (基于阈值)
        threshold = np.mean(np.abs(normalized)) * 1.5
        denoised = np.where(np.abs(normalized) < threshold, 0, normalized)

        return denoised

    # 在 preprocessing/temporal.py 的 TemporalPreprocessor 类中添加

    def process_signal(self, audio_signal, sr=None):
        """
        处理音频信号（而不是文件路径）

        Args:
            audio_signal: 音频信号数组
            sr: 采样率（如果为None则使用self.sample_rate）

        Returns:
            处理后的音频信号
        """
        target_sr = sr or self.sample_rate

        # 滤波
        filtered = signal.lfilter(self.filter[0], self.filter[1], audio_signal)

        # 归一化
        normalized = filtered / np.max(np.abs(filtered))

        # 简单去噪 (基于阈值)
        threshold = np.mean(np.abs(normalized)) * 1.5
        denoised = np.where(np.abs(normalized) < threshold, 0, normalized)

        return denoised