# configs.py - 自动生成
import yaml
import os
from datetime import datetime


def load_config(config_path):
    """加载yaml配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 补充默认配置
    config = _fill_default_config(config)

    # 创建必要的目录
    _create_directories(config)

    return config


def _fill_default_config(config):
    """补充默认配置参数"""
    # 数据配置默认值
    data_defaults = {
        'sample_rate': 16000,
        'feature_type': 'mel',
        'max_length': 4096,
        'train_data_path': 'data/processed/train',
        'val_data_path': 'data/processed/val',
        'test_data_path': 'data/processed/test',
        'train_labels_path': 'data/splits/train_labels.txt',
        'val_labels_path': 'data/splits/val_labels.txt',
        'test_labels_path': 'data/splits/test_labels.txt'
    }

    # 训练配置默认值
    train_defaults = {
        'batch_size': 32,
        'epochs': 50,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'early_stopping_patience': 10,
        'save_freq': 5
    }

    # 模型配置默认值
    model_defaults = {
        'num_classes': 10,
        'in_channels': 1,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12
    }

    # 结果配置默认值
    results_defaults = {
        'checkpoint_dir': 'results/checkpoints',
        'results_dir': 'results',
        'figures_dir': 'results/figures',
        'logs_dir': 'results/logs'
    }

    # 更新配置
    config['data'] = {**data_defaults, **config.get('data', {})}
    config['train'] = {**train_defaults, **config.get('train', {})}
    config['model'] = {**model_defaults, **config.get('model', {})}
    config['results'] = {**results_defaults, **config.get('results', {})}

    return config


def _create_directories(config):
    """创建配置中指定的目录"""
    directories = [
        config['results']['checkpoint_dir'],
        config['results']['results_dir'],
        config['results']['figures_dir'],
        config['results']['logs_dir'],
        config['data']['train_data_path'],
        config['data']['val_data_path'],
        config['data']['test_data_path']
    ]

    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)


def save_config(config, save_path):
    """保存配置文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"配置已保存至: {save_path}")


def create_experiment_config(experiment_name, **kwargs):
    """创建实验配置"""
    config = {
        'experiment_name': experiment_name,
        'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **kwargs
    }
    return config


# 导入torch（避免配置加载时torch未导入）
try:
    import torch
except ImportError:
    print("警告: 未安装PyTorch，部分默认配置可能无法自动设置")