# N-PHYSound: 物理引导的水下声音信号分类项目

## 项目简介
本项目实现了一种基于物理引导的水下声音信号分类模型（N-PHYSound），通过融合深度学习与水声物理模型，提升模型在跨环境泛化和零样本识别任务中的性能，对比了传统CNN、AST等基线模型的效果。

## 环境配置
### 1. Conda环境创建
```bash
# 创建虚拟环境
conda create -n n-physound python=3.9
conda activate n-physound

# 安装核心依赖
conda install numpy pandas matplotlib scipy
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install scikit-learn librosa
pip install soundfile tqdm tensorboard pyyaml seaborn