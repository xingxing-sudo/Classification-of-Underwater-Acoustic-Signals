# evaluate.py - 自动生成
import torch
import os
from tqdm import tqdm
from models.cnn_models import ResNet18, SimpleCNN
from models.sota_models import ASTModel
from models.cnn_phys_reg import CNNPhysRegModel
from models.n_physound import N_PHYSound
from utils.data_loader import create_data_loaders
from utils.metrics import (
    accuracy, cross_env_accuracy, zero_shot_accuracy,
    inference_time_evaluation, detailed_classification_report
)
from utils.config import load_config


def load_trained_model(model_name, config):
    """加载训练好的模型"""
    device = torch.device(config['train']['device'])
    checkpoint_dir = config['results']['checkpoint_dir']
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')

    # 根据模型名称加载对应模型
    if model_name == "ResNet18":
        model = ResNet18(
            num_classes=config['model']['num_classes'],
            in_channels=config['model']['in_channels']
        ).to(device)
    elif model_name == "SimpleCNN":
        model = SimpleCNN(
            num_classes=config['model']['num_classes'],
            in_channels=config['model']['in_channels']
        ).to(device)
    elif model_name == "AST":
        model = ASTModel(
            num_classes=config['model']['num_classes'],
            in_channels=config['model']['in_channels'],
            embed_dim=config['model']['embed_dim'],
            depth=config['model']['depth'],
            num_heads=config['model']['num_heads']
        ).to(device)
    elif model_name == "CNNPhysReg":
        model = CNNPhysRegModel(
            num_classes=config['model']['num_classes'],
            in_channels=config['model']['in_channels']
        ).to(device)
    elif model_name == "N_PHYSound":
        model = N_PHYSound(
            num_classes=config['model']['num_classes']
        ).to(device)
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")

    # 加载最佳检查点
    model.load_checkpoint(checkpoint_path)
    model.eval()
    return model


def evaluate_single_model(model_name, config, test_loaders=None, zero_shot_loader=None):
    """评估单个模型的性能"""
    print("=" * 60)
    print(f"开始评估模型: {model_name}")
    print("=" * 60)

    # 设备设置
    device = torch.device(config['train']['device'])

    # 加载模型
    model = load_trained_model(model_name, config)

    # 创建基础测试数据加载器
    _, _, test_loader = create_data_loaders(
        train_data_dir=config['data']['train_data_path'],
        val_data_dir=config['data']['val_data_path'],
        test_data_dir=config['data']['test_data_path'],
        train_label_path=config['data']['train_labels_path'],
        val_label_path=config['data']['val_labels_path'],
        test_label_path=config['data']['test_labels_path'],
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        sample_rate=config['data']['sample_rate'],
        feature_type=config['data']['feature_type'],
        max_length=config['data']['max_length']
    )

    # 1. 标准测试集准确率
    model.eval()
    test_top1_acc = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="计算标准测试集准确率"):
            features, _, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            top1_acc = accuracy(outputs, labels, topk=(1,))[0]
            test_top1_acc += top1_acc.item()

    avg_test_acc = test_top1_acc / len(test_loader)
    print(f"标准测试集准确率: {avg_test_acc:.4f}")

    # 2. 跨环境泛化准确率
    if test_loaders is None:
        # 若未提供跨环境测试集，使用基础测试集作为默认
        test_loaders = {"default": test_loader}
    avg_cross_acc, env_accs = cross_env_accuracy(model, test_loaders, device)

    # 3. 零样本识别准确率
    zero_shot_acc = 0.0
    if zero_shot_loader is not None:
        zero_shot_acc = zero_shot_accuracy(model, zero_shot_loader, device)

    # 4. 推理时间评估
    avg_infer_time, std_infer_time = inference_time_evaluation(model, test_loader, device)

    # 5. 详细分类报告
    class_names = [f"Class_{i}" for i in range(config['model']['num_classes'])]
    report, cm = detailed_classification_report(model, test_loader, device, class_names)

    # 整理结果
    results = {
        "standard_acc": avg_test_acc,
        "cross_env_acc": avg_cross_acc,
        "zero_shot_acc": zero_shot_acc,
        "inference_time": avg_infer_time,
        "inference_time_std": std_infer_time,
        "env_accs": env_accs,
        "classification_report": report,
        "confusion_matrix": cm
    }

    return results


def evaluate_all_models(config):
    """评估所有模型的性能"""
    # 模型列表
    model_names = ["ResNet18", "SimpleCNN", "AST", "CNNPhysReg", "N_PHYSound"]

    # 创建跨环境测试加载器（示例：使用不同信噪比的测试集）
    cross_env_test_loaders = {}
    snr_levels = ["0dB", "5dB", "10dB", "15dB"]
    for snr in snr_levels:
        test_data_dir = f"data/processed/test_snr_{snr}"
        test_label_path = f"data/splits/test_labels_{snr}.txt"
        if os.path.exists(test_data_dir) and os.path.exists(test_label_path):
            _, _, test_loader = create_data_loaders(
                train_data_dir=config['data']['train_data_path'],
                val_data_dir=config['data']['val_data_path'],
                test_data_dir=test_data_dir,
                train_label_path=config['data']['train_labels_path'],
                val_label_path=config['data']['val_labels_path'],
                test_label_path=test_label_path,
                batch_size=config['train']['batch_size'],
                num_workers=config['train']['num_workers'],
                sample_rate=config['data']['sample_rate'],
                feature_type=config['data']['feature_type'],
                max_length=config['data']['max_length']
            )
            cross_env_test_loaders[f"SNR_{snr}"] = test_loader

    # 创建零样本测试加载器
    zero_shot_loader = None
    zero_shot_data_dir = "data/processed/zero_shot_test"
    zero_shot_label_path = "data/splits/zero_shot_labels.txt"
    if os.path.exists(zero_shot_data_dir) and os.path.exists(zero_shot_label_path):
        _, _, zero_shot_loader = create_data_loaders(
            train_data_dir=config['data']['train_data_path'],
            val_data_dir=config['data']['val_data_path'],
            test_data_dir=zero_shot_data_dir,
            train_label_path=config['data']['train_labels_path'],
            val_label_path=config['data']['val_labels_path'],
            test_label_path=zero_shot_label_path,
            batch_size=config['train']['batch_size'],
            num_workers=config['train']['num_workers'],
            sample_rate=config['data']['sample_rate'],
            feature_type=config['data']['feature_type'],
            max_length=config['data']['max_length']
        )

        # 评估所有模型
        all_results = {}
        for model_name in model_names:
            try:
                results = evaluate_single_model(
                    model_name,
                    config,
                    test_loaders=cross_env_test_loaders,
                    zero_shot_loader=zero_shot_loader
                )
                all_results[model_name] = results
            except Exception as e:
                print(f"评估模型 {model_name} 失败: {e}")
                all_results[model_name] = None

        # 打印汇总结果
        print("\n" + "=" * 80)
        print("所有模型性能汇总")
        print("=" * 80)
        print(f"{'模型名称':<20} {'标准准确率':<15} {'跨环境准确率':<15} {'零样本准确率':<15} {'平均推理时间(s)':<20}")
        print("-" * 80)
        for model_name, results in all_results.items():
            if results is not None:
                print(
                    f"{model_name:<20} {results['standard_acc']:.4f} {'':<7} {results['cross_env_acc']:.4f} {'':<7} {results['zero_shot_acc']:.4f} {'':<7} {results['inference_time']:.4f} ± {results['inference_time_std']:.4f}")
            else:
                print(f"{model_name:<20} {'评估失败':<15} {'评估失败':<15} {'评估失败':<15} {'评估失败':<20}")

        return all_results