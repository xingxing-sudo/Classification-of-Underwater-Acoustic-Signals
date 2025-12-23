#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral preprocessing module for audio signals
Extracts Mel-spectrograms, MFCCs, and other spectral features
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')



import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

class SpectralPreprocessor:

    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def extract_mel_spectrogram(self, audio_signal):
        """提取梅尔频谱图"""
        # 计算梅尔频谱
        mel_spec = librosa.feature.melspectrogram(
            y=audio_signal,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        # 转换为对数刻度
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    def extract_mfcc(self, audio_signal, n_mfcc=40):
        """提取MFCC特征"""
        mfcc = librosa.feature.mfcc(
            y=audio_signal,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mfcc=n_mfcc
        )
        # 计算MFCC的一阶和二阶差分
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        # 拼接特征
        combined_mfcc = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
        return combined_mfcc

    def extract_stft(self, audio_signal):
        """提取短时傅里叶变换结果"""
        stft = librosa.stft(
            y=audio_signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        # 计算幅度谱和相位谱
        amp_stft = np.abs(stft)
        phase_stft = np.angle(stft)
        return amp_stft, phase_stft

    def plot_spectrogram(self, spectrogram, save_path=None):
        """绘制频谱图并可选保存"""
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            spectrogram,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def process(self, audio_signal, feature_type='mel'):
        """统一的频谱特征提取入口"""
        if feature_type == 'mel':
            return self.extract_mel_spectrogram(audio_signal)
        elif feature_type == 'mfcc':
            return self.extract_mfcc(audio_signal)
        elif feature_type == 'stft':
            return self.extract_stft(audio_signal)
        else:
            raise ValueError(f"不支持的特征类型: {feature_type}，可选['mel', 'mfcc', 'stft']")