import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from preprocessing.temporal import TemporalPreprocessor
from preprocessing.spectral import SpectralPreprocessor


class UnderwaterSoundDataset(Dataset):
    """
    水下声音信号数据集类（适配按类别文件夹组织的数据）
    支持两种标签加载方式：
    1. 从标签文件加载（每行对应一个样本的标签，与音频文件顺序一致）
    2. 自动从类别文件夹名生成标签（无需标签文件）
    """

    def __init__(self, data_dir, label_path=None, sample_rate=16000, feature_type='mel',
                 max_length=4096, transform=None):
        self.data_dir = data_dir  # 数据集目录（如data/processed/train）
        self.label_path = label_path  # 标签文件路径（可选）
        self.sample_rate = sample_rate  # 音频采样率
        self.feature_type = feature_type  # 提取的特征类型（mel/mfcc/stft）
        self.max_length = max_length  # 音频信号固定长度
        self.transform = transform  # 额外数据变换

        # 初始化预处理类
        self.temporal_preprocessor = TemporalPreprocessor(sample_rate=sample_rate)
        self.spectral_preprocessor = SpectralPreprocessor(sample_rate=sample_rate)

        # 加载音频文件路径和对应标签
        self.audio_files, self.labels, self.class_to_label = self._load_data()

    def _load_data(self):
        """加载音频文件路径和标签，适配类别文件夹结构"""
        audio_files = []
        labels = []
        class_to_label = {}

        # 情况1：提供了标签文件，按标签文件加载
        if self.label_path is not None and os.path.exists(self.label_path):
            # 先获取数据目录下所有音频文件（保持顺序）
            for root, _, files in os.walk(self.data_dir):
                for file_name in files:
                    if file_name.endswith(('.wav', '.flac', '.mp3')):
                        audio_files.append(os.path.join(root, file_name))
            # 排序，保证与标签文件顺序一致
            audio_files.sort()

            # 加载标签文件
            with open(self.label_path, 'r', encoding='utf-8') as f:
                labels = [int(line.strip()) for line in f.readlines()]

            # 自动识别类别映射（从数据目录结构中提取）
            class_folders = sorted(
                [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))])
            class_to_label = {cls_name: idx for idx, cls_name in enumerate(class_folders)}

        # 情况2：未提供标签文件，从文件夹名自动生成标签
        else:
            class_folders = sorted(
                [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))])
            class_to_label = {cls_name: idx for idx, cls_name in enumerate(class_folders)}

            # 遍历每个类别文件夹，收集音频文件和标签
            for cls_name, label in class_to_label.items():
                cls_dir = os.path.join(self.data_dir, cls_name)
                for file_name in os.listdir(cls_dir):
                    if file_name.endswith(('.wav', '.flac', '.mp3')):
                        audio_files.append(os.path.join(cls_dir, file_name))
                        labels.append(label)

        # 校验音频文件数量与标签数量一致
        assert len(audio_files) == len(labels), f"音频文件数量（{len(audio_files)}）与标签数量（{len(labels)}）不匹配"
        print(f"成功加载 {len(audio_files)} 个样本，{len(class_to_label)} 个类别")
        return audio_files, labels, class_to_label

    def _pad_or_truncate(self, signal):
        """将音频信号填充或截断到固定长度"""
        signal_len = len(signal)
        if signal_len > self.max_length:
            # 截断到最大长度
            return signal[:self.max_length]
        elif signal_len < self.max_length:
            # 补零到最大长度
            pad_length = self.max_length - signal_len
            return np.pad(signal, (0, pad_length), mode='constant')
        else:
            return signal

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.audio_files)

    def __getitem__(self, idx):
        """获取单个样本（特征、原始音频、标签）"""
        # 1. 加载音频文件
        audio_path = self.audio_files[idx]
        try:
            audio_signal, sr = sf.read(audio_path)
            # 若采样率不一致，重采样（依赖temporal_preprocessor中的重采样逻辑）
            if sr != self.sample_rate:
                audio_signal = self.temporal_preprocessor._resample(audio_signal, sr, self.sample_rate)
        except Exception as e:
            raise RuntimeError(f"加载音频文件 {audio_path} 失败：{e}")

        # 2. 时域预处理（滤波、归一化等）
        processed_audio = self.temporal_preprocessor.process_signal(audio_signal)  # 直接处理信号，无需重新加载文件

        # 3. 填充/截断到固定长度
        processed_audio = self._pad_or_truncate(processed_audio)

        # 4. 提取频谱特征
        feature = None
        if self.feature_type == 'mel':
            feature = self.spectral_preprocessor.extract_mel_spectrogram(processed_audio)
        elif self.feature_type == 'mfcc':
            feature = self.spectral_preprocessor.extract_mfcc(processed_audio)
        elif self.feature_type == 'stft':
            amp_stft, _ = self.spectral_preprocessor.extract_stft(processed_audio)
            feature = amp_stft
        elif self.feature_type == 'raw':
            feature = processed_audio  # 直接使用原始时域信号
        else:
            raise ValueError(f"不支持的特征类型：{self.feature_type}，可选['mel', 'mfcc', 'stft', 'raw']")

        # 5. 调整维度（添加通道维度，适配模型输入：[C, H, W]）
        if len(feature.shape) == 2:
            feature = np.expand_dims(feature, axis=0)
        elif len(feature.shape) == 1:
            feature = np.expand_dims(feature, axis=0)

        # 6. 类型转换为torch张量
        feature = torch.tensor(feature, dtype=torch.float32)
        original_audio = torch.tensor(processed_audio, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # 7. 应用额外数据变换
        if self.transform is not None:
            feature = self.transform(feature)

        # 返回：特征矩阵、原始音频信号、标签
        return feature, original_audio, label


def create_data_loaders(train_data_dir, val_data_dir, test_data_dir,
                        train_label_path, val_label_path, test_label_path,
                        batch_size=32, num_workers=4, **dataset_kwargs):
    """
    创建训练、验证、测试数据加载器
    :param train_data_dir: 训练集目录
    :param val_data_dir: 验证集目录
    :param test_data_dir: 测试集目录
    :param train_label_path: 训练集标签文件路径
    :param val_label_path: 验证集标签文件路径
    :param test_label_path: 测试集标签文件路径
    :param batch_size: 批次大小
    :param num_workers: 数据加载线程数
    :param dataset_kwargs: 传递给UnderwaterSoundDataset的其他参数
    :return: train_loader, val_loader, test_loader
    """
    # 创建数据集
    train_dataset = UnderwaterSoundDataset(
        data_dir=train_data_dir,
        label_path=train_label_path,
        **dataset_kwargs
    )
    val_dataset = UnderwaterSoundDataset(
        data_dir=val_data_dir,
        label_path=val_label_path,
        **dataset_kwargs
    )
    test_dataset = UnderwaterSoundDataset(
        data_dir=test_data_dir,
        label_path=test_label_path,
        **dataset_kwargs
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集打乱
        num_workers=num_workers,
        pin_memory=True,  # 加速GPU数据传输
        drop_last=True  # 丢弃最后一个不完整批次
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不打乱
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不打乱
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"数据加载器创建完成：")
    print(f"  训练集：{len(train_dataset)} 样本，{len(train_loader)} 批次")
    print(f"  验证集：{len(val_dataset)} 样本，{len(val_loader)} 批次")
    print(f"  测试集：{len(test_dataset)} 样本，{len(test_loader)} 批次")

    return train_loader, val_loader, test_loader


# 测试代码（可选）
if __name__ == "__main__":
    # 测试数据集加载
    dataset = UnderwaterSoundDataset(
        data_dir="data/processed/train",
        label_path="data/splits/train_labels.txt",
        sample_rate=16000,
        feature_type="mel",
        max_length=4096
    )
    print(f"数据集样本数：{len(dataset)}")
    feature, audio, label = dataset[0]
    print(f"特征形状：{feature.shape}")
    print(f"音频形状：{audio.shape}")
    print(f"标签：{label.item()}")
    print(f"类别映射：{dataset.class_to_label}")
