# train_baselines.py - 自动生成
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from models.cnn_models import ResNet18, SimpleCNN
from models.sota_models import ASTModel
from utils.data_loader import create_data_loaders
from utils.metrics import accuracy, detailed_classification_report
from utils.config import load_config


def train_baseline_model(model, train_loader, val_loader, criterion, optimizer, config, model_name):
    """通用基线模型训练函数"""
    device = torch.device(config['train']['device'])
    epochs = config['train']['epochs']
    checkpoint_dir = config['results']['checkpoint_dir']
    early_stopping_patience = config['train']['early_stopping_patience']
    save_freq = config['train']['save_freq']

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_top1_acc = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            features, _, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算指标
            train_loss += loss.item()
            top1_acc = accuracy(outputs, labels, topk=(1,))[0]
            train_top1_acc += top1_acc.item()

        # 平均训练指标
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_top1_acc / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_top1_acc = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
                features, _, labels = batch
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                top1_acc = accuracy(outputs, labels, topk=(1,))[0]
                val_top1_acc += top1_acc.item()

        # 平均验证指标
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_top1_acc / len(val_loader)

        # 打印日志
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc@1: {avg_train_acc:.4f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc@1: {avg_val_acc:.4f}%")

        # 保存检查点
        if (epoch + 1) % save_freq == 0:
            model.save_checkpoint(
                checkpoint_dir,
                epoch + 1,
                optimizer,
                avg_val_loss
            )

        # 保存最佳模型
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            patience_counter = 0
            model.save_checkpoint(
                checkpoint_dir,
                epoch + 1,
                optimizer,
                avg_val_loss,
                best_acc=True
            )
            print(f"新的最佳模型已保存（准确率: {best_val_acc:.4f}%）")
        else:
            patience_counter += 1
            print(f"早停计数器: {patience_counter}/{early_stopping_patience}")

        # 早停
        if patience_counter >= early_stopping_patience:
            print("早停触发，停止训练")
            break

    print(f"训练完成！最佳验证准确率: {best_val_acc:.4f}%")
    return model


def train_resnet18(config):
    """训练ResNet18基线模型"""
    print("=" * 50)
    print("开始训练ResNet18基线模型")
    print("=" * 50)

    # 设备设置
    device = torch.device(config['train']['device'])

    # 创建数据加载器
    train_loader, val_loader, _ = create_data_loaders(
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

    # 模型初始化
    model = ResNet18(
        num_classes=config['model']['num_classes'],
        in_channels=config['model']['in_channels']
    ).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )

    # 训练模型
    model = train_baseline_model(
        model, train_loader, val_loader, criterion, optimizer, config, "ResNet18"
    )

    # 生成详细分类报告
    print("生成ResNet18验证集分类报告")
    detailed_classification_report(
        model, val_loader, device,
        class_names=[f"Class_{i}" for i in range(config['model']['num_classes'])]
    )

    return model


def train_simple_cnn(config):
    """训练SimpleCNN基线模型"""
    print("=" * 50)
    print("开始训练SimpleCNN基线模型")
    print("=" * 50)

    # 设备设置
    device = torch.device(config['train']['device'])

    # 创建数据加载器
    train_loader, val_loader, _ = create_data_loaders(
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

    # 模型初始化
    model = SimpleCNN(
        num_classes=config['model']['num_classes'],
        in_channels=config['model']['in_channels']
    ).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )

    # 训练模型
    model = train_baseline_model(
        model, train_loader, val_loader, criterion, optimizer, config, "SimpleCNN"
    )

    # 生成详细分类报告
    print("生成SimpleCNN验证集分类报告")
    detailed_classification_report(
        model, val_loader, device,
        class_names=[f"Class_{i}" for i in range(config['model']['num_classes'])]
    )

    return model


def train_ast(config):
    """训练AST SOTA模型"""
    print("=" * 50)
    print("开始训练AST SOTA模型")
    print("=" * 50)

    # 设备设置
    device = torch.device(config['train']['device'])

    # 创建数据加载器（AST需要频谱图输入）
    train_loader, val_loader, _ = create_data_loaders(
        train_data_dir=config['data']['train_data_path'],
        val_data_dir=config['data']['val_data_path'],
        test_data_dir=config['data']['test_data_path'],
        train_label_path=config['data']['train_labels_path'],
        val_label_path=config['data']['val_labels_path'],
        test_label_path=config['data']['test_labels_path'],
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        sample_rate=config['data']['sample_rate'],
        feature_type='mel',  # AST固定使用梅尔频谱
        max_length=config['data']['max_length']
    )

    # 模型初始化
    model = ASTModel(
        num_classes=config['model']['num_classes'],
        in_channels=config['model']['in_channels'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads']
    ).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )

    # 训练模型
    model = train_baseline_model(
        model, train_loader, val_loader, criterion, optimizer, config, "AST"
    )

    # 生成详细分类报告
    print("生成AST验证集分类报告")
    detailed_classification_report(
        model, val_loader, device,
        class_names=[f"Class_{i}" for i in range(config['model']['num_classes'])]
    )

    return model