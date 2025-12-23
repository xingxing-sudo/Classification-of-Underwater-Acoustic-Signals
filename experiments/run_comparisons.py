# experiments/run_comparisons.py
import os
from utils.config import load_config
from train_baselines import train_resnet18, train_ast, train_cnn_phys_reg
from train_n_physound import train_n_physound
from evaluate import evaluate_all_models


def run_all_experiments():
    # 加载配置
    config = load_config('configs/main_config.yaml')

    # 创建结果目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)

    # 训练所有模型
    print("Training ResNet18 (传统CNN)...")
    train_resnet18(config)

    print("\nTraining AST (当前SOTA)...")
    train_ast(config)

    print("\nTraining CNN+物理正则化...")
    train_cnn_phys_reg(config)

    print("\nTraining N-PHYSound (完整版)...")
    train_n_physound(config)

    # 评估所有模型
    print("\nEvaluating all models...")
    results = evaluate_all_models(config)

    # 保存结果
    with open(os.path.join(config['results_dir'], 'comparison_results.txt'), 'w') as f:
        for model, metrics in results.items():
            f.write(f"Model: {model}\n")
            f.write(f"标准测试集准确率: {metrics['standard_acc']:.4f}\n")
            f.write(f"跨环境泛化准确率: {metrics['cross_env_acc']:.4f}\n")
            f.write(f"零样本船舶识别准确率: {metrics['zero_shot_acc']:.4f}\n")
            f.write(f"平均推理时间: {metrics['inference_time']:.4f}s\n\n")

    print("所有实验完成！结果已保存。")


if __name__ == "__main__":
    run_all_experiments()