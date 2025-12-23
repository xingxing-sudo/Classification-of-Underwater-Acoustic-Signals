#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
划分原始数据（按类别文件夹组织）为训练集、验证集、测试集
适配场景：data/raw/下有多个类别子文件夹（如Cargo、Passengership等），每个文件夹下存放对应类别的音频文件
"""
import os
import random
import shutil
from sklearn.model_selection import train_test_split


def split_raw_data(raw_data_dir, output_root_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    划分原始数据（按类别文件夹组织）为训练集、验证集、测试集
    适配场景：data/raw/下有多个类别子文件夹（如Cargo、Passengership等），每个文件夹下存放对应类别的音频文件
    :param raw_data_dir: 原始数据根目录（如data/raw）
    :param output_root_dir: 输出根目录（如data）
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    :param test_ratio: 测试集比例
    """
    # 校验比例和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "训练集、验证集、测试集比例之和必须为1"

    # 定义输出目录路径
    train_output_dir = os.path.join(output_root_dir, "processed/train")
    val_output_dir = os.path.join(output_root_dir, "processed/val")
    test_output_dir = os.path.join(output_root_dir, "processed/test")
    splits_output_dir = os.path.join(output_root_dir, "splits")

    # 创建目录（若不存在）
    for dir_path in [train_output_dir, val_output_dir, test_output_dir, splits_output_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 1. 获取所有类别文件夹并建立标签映射（文件夹名->数字标签，如Cargo->0）
    class_folders = sorted([f for f in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, f))])
    class_to_label = {cls_name: idx for idx, cls_name in enumerate(class_folders)}
    print(f"成功识别 {len(class_folders)} 个类别：")
    for cls, lbl in class_to_label.items():
        print(f"  类别 {cls} -> 数字标签 {lbl}")

    # 2. 遍历每个类别，划分数据并复制文件
    # 用于存储所有集的文件路径和标签（后续生成标签文件）
    all_train_info = []
    all_val_info = []
    all_test_info = []

    for cls_name, label in class_to_label.items():
        # 当前类别的原始目录
        cls_raw_dir = os.path.join(raw_data_dir, cls_name)
        # 获取当前类别下所有音频文件
        audio_files = [f for f in os.listdir(cls_raw_dir) if f.endswith(('.wav', '.flac', '.mp3'))]
        if not audio_files:
            print(f"警告：类别 {cls_name} 目录下无有效音频文件，跳过该类别")
            continue
        random.shuffle(audio_files)  # 打乱顺序
        print(f"处理类别 {cls_name}：共 {len(audio_files)} 个音频文件")

        # 划分当前类别数据
        # ✅ 关键修复：使用一次性三层划分
        n_total = len(audio_files)

        # 计算每个集合的精确数量
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))
        n_test = max(1, n_total - n_train - n_val)  # 确保至少1个

        # 如果分配超出，调整
        if n_train + n_val + n_test > n_total:
            # 优先保证训练集，减少验证集
            n_val = max(1, n_total - n_train - n_test)

        # 确保不超出
        n_train = min(n_train, n_total - 2)  # 至少留2个给其他集合
        n_val = min(n_val, n_total - n_train - 1)  # 至少留1个给测试集
        n_test = n_total - n_train - n_val

        print(f"  精确分配: 训练集={n_train}, 验证集={n_val}, 测试集={n_test}")

        # ✅ 确保无重叠的手动划分
        train_files = audio_files[:n_train]
        val_files = audio_files[n_train:n_train + n_val]
        test_files = audio_files[n_train + n_val:]

        # ✅ 验证无重复
        train_set = set(train_files)
        val_set = set(val_files)
        test_set = set(test_files)

        if len(train_set) + len(val_set) + len(test_set) != n_total:
            print(f"  ⚠️ 警告：检测到重复或遗漏！")
            print(f"    原始: {n_total}, 分配后: {len(train_set) + len(val_set) + len(test_set)}")

        if train_set.intersection(val_set) or train_set.intersection(test_set) or val_set.intersection(test_set):
            print(f"  ❌ 错误：检测到重复文件！")

        # 复制文件到对应输出目录，并记录文件信息（相对路径+标签）
        def copy_files_and_record(file_list, target_root_dir, record_list):
            for file_name in file_list:
                # 原始文件路径
                src_path = os.path.join(cls_raw_dir, file_name)
                # 目标文件路径（保留类别文件夹结构，如processed/train/Cargo/xxx.wav）
                dst_dir = os.path.join(target_root_dir, cls_name)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, file_name)
                # 复制文件
                shutil.copy2(src_path, dst_path)
                # 记录相对路径（相对于processed/xxx目录）和标签
                rel_file_path = os.path.join(cls_name, file_name)
                record_list.append((rel_file_path, label))

        # 处理训练集、验证集、测试集
        copy_files_and_record(train_files, train_output_dir, all_train_info)
        copy_files_and_record(val_files, val_output_dir, all_val_info)
        copy_files_and_record(test_files, test_output_dir, all_test_info)

    # 3. 生成标签文件
    # 标签文件格式1：每行是「文件相对路径\t数字标签」（便于核对）
    # 标签文件格式2：每行仅数字标签（适配原有数据加载逻辑）
    def write_label_files(record_list, target_dir, prefix):
        # 格式1：带文件路径的标签文件
        with open(os.path.join(target_dir, f"{prefix}_labels_with_path.txt"), 'w', encoding='utf-8') as f:
            for rel_path, label in record_list:
                f.write(f"{rel_path}\t{label}\n")
        # 格式2：仅标签（适配原有data_loader逻辑）
        with open(os.path.join(target_dir, f"{prefix}_labels.txt"), 'w', encoding='utf-8') as f:
            for _, label in record_list:
                f.write(f"{label}\n")

    # 生成训练、验证、测试集标签文件
    write_label_files(all_train_info, splits_output_dir, "train")
    write_label_files(all_val_info, splits_output_dir, "val")
    write_label_files(all_test_info, splits_output_dir, "test")

    # 4. 打印汇总信息
    print("\n" + "=" * 50)
    print("数据划分完成！汇总信息：")
    print(f"原始数据目录：{raw_data_dir}")
    print(f"输出数据目录：{output_root_dir}")
    print(f"训练集：{len(all_train_info)} 个样本，存放于 {train_output_dir}")
    print(f"验证集：{len(all_val_info)} 个样本，存放于 {val_output_dir}")
    print(f"测试集：{len(all_test_info)} 个样本，存放于 {test_output_dir}")
    print(f"标签文件：存放于 {splits_output_dir}")
    print(f"类别与标签映射：{class_to_label}")
    print("=" * 50)


if __name__ == "__main__":
    # 示例调用：直接运行该脚本即可执行数据划分
    # 适配你的数据结构：data/raw/下有4类船的分类文件夹
    split_raw_data(
        raw_data_dir="data/raw",  # 原始数据根目录
        output_root_dir="data",  # 输出根目录
        train_ratio=0.7,  # 训练集占70%
        val_ratio=0.15,  # 验证集占15%
        test_ratio=0.15  # 测试集占15%
    )