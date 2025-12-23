# models/__init__.py
from .base_model import BaseModel
from .cnn_models import ResNet18, SimpleCNN
from .sota_models import ASTModel
# ... 导出其他需要的类和函数

# 明确指定导出的内容
__all__ = [
    'BaseModel',
    'ResNet18',
    'SimpleCNN',
    'ASTModel',
    # ... 其他导出的名称
]