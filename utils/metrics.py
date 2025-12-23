# metrics.py - 自动生成
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time


def accuracy(output, target, topk=(1,)):
    """计算top-k准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cross_env_accuracy(model, test_loaders, device):
    """计算跨环境泛化准确率（多个测试环境的平均准确率）"""
    env_accs = []
    model.eval()
    with torch.no_grad():
        for loader_name, test_loader in test_loaders.items():
            total_correct = 0
            total_samples = 0
            for batch in test_loader:
                features, _, labels = batch
                features = features.to(device)
                labels = labels.to(device)

                # 适配不同模型的输出
                outputs = model(features)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

            env_acc = total_correct / total_samples
            env_accs.append(env_acc)
            print(f"环境 {loader_name} 准确率: {env_acc:.4f}")

    avg_cross_env_acc = np.mean(env_accs)
    print(f"平均跨环境准确率: {avg_cross_env_acc:.4f}")
    return avg_cross_env_acc, env_accs


def zero_shot_accuracy(model, zero_shot_loader, device, class_mapping=None):
    """计算零样本识别准确率"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in zero_shot_loader:
            features, _, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            # 适配不同模型的输出
            outputs = model(features)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, preds = torch.max(outputs, 1)

            # 类别映射（若需要）
            if class_mapping is not None:
                preds = torch.tensor([class_mapping[p.item()] for p in preds], device=device)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    zero_shot_acc = accuracy_score(all_labels, all_preds)
    print(f"零样本识别准确率: {zero_shot_acc:.4f}")
    return zero_shot_acc


def inference_time_evaluation(model, test_loader, device, num_runs=10):
    """评估模型推理时间"""
    model.eval()
    inference_times = []

    # 预热（避免首次推理包含初始化时间）
    with torch.no_grad():
        for batch in test_loader:
            features, _, _ = batch
            features = features.to(device)
            model(features)
            break

    # 正式测试
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            for batch in test_loader:
                features, _, _ = batch
                features = features.to(device)
                model(features)
            end_time = time.time()

            total_time = end_time - start_time
            avg_batch_time = total_time / len(test_loader)
            inference_times.append(avg_batch_time)

    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    print(f"平均推理时间（每批次）: {avg_inference_time:.4f} ± {std_inference_time:.4f} 秒")
    return avg_inference_time, std_inference_time


def detailed_classification_report(model, test_loader, device, class_names=None):
    """生成详细的分类报告"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            features, _, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 生成分类报告
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    )

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    print("详细分类报告:")
    print(report)
    print("混淆矩阵:")
    print(cm)

    return report, cm