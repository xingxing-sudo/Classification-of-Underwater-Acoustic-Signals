# base_model.py - 自动生成
import torch
import torch.nn as nn
import os
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """所有模型的基类，定义通用接口和方法"""

    def __init__(self, num_classes, device=None):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    @abstractmethod
    def forward(self, x):
        """前向传播接口，必须由子类实现"""
        pass

    def save_checkpoint(self, checkpoint_dir, epoch, optimizer, loss, best_acc=False):
        """保存模型检查点"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_name = 'best_model.pth' if best_acc else f'model_epoch_{epoch}.pth'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'num_classes': self.num_classes
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"模型检查点已保存至: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path, optimizer=None):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.num_classes = checkpoint['num_classes']

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 0.0)
        print(f"模型检查点已从: {checkpoint_path} 加载（epoch: {epoch}, loss: {loss:.4f}）")
        return epoch, loss

    def freeze_layers(self, layer_names):
        """冻结指定层的参数"""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
        print(f"已冻结层: {layer_names}")

    def unfreeze_layers(self, layer_names):
        """解冻指定层的参数"""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
        print(f"已解冻层: {layer_names}")