# N-PHYSound: 物理引导的水下声音信号分类项目
## 项目概述
本项目是一个基于**PyCharm + Conda环境**构建的水下声音信号分类系统，核心创新在于融合深度学习与水声物理模型，提出N-PHYSound模型，解决传统音频分类模型在跨环境泛化、零样本识别任务中的性能瓶颈。项目包含完整的数据预处理、模型定义、训练评估、结果可视化流程，同时提供多种基线模型（ResNet18、SimpleCNN、AST）用于对比实验。

## 项目结构与各类/模块作用
### 整体目录结构
```
n-physound/
├── configs/                  # 配置文件目录
├── data/                     # 数据集目录
├── experiments/              # 实验执行脚本目录
├── models/                   # 模型定义目录
├── physics/                  # 物理模型模块目录
├── preprocessing/            # 数据预处理模块目录
├── results/                  # 实验结果保存目录
├── utils/                    # 工具函数目录
├── main.py                   # 项目入口文件
└── README.md                 # 项目说明文档
```

### 各目录核心类/模块详细作用
#### 1. configs/（配置文件目录）
| 文件名称               | 作用                                                                 |
|------------------------|----------------------------------------------------------------------|
| main_config.yaml       | 项目全局配置文件，统一管理数据路径、模型参数、训练超参、结果保存路径等所有可配置项 |

#### 2. data/（数据集目录，需手动创建或通过脚本生成子目录）
| 子目录/文件            | 作用                                                                 |
|------------------------|----------------------------------------------------------------------|
| raw/                   | 存放原始水下音频文件（支持.wav/.flac/.mp3格式）                     |
| processed/train/       | 划分后的训练集音频文件                                               |
| processed/val/         | 划分后的验证集音频文件                                               |
| processed/test/        | 划分后的测试集音频文件                                               |
| splits/train_labels.txt| 训练集标签文件，每行对应一个音频文件的类别标签（整数格式）           |
| splits/val_labels.txt  | 验证集标签文件，格式与训练集一致                                     |
| splits/test_labels.txt | 测试集标签文件，格式与训练集一致                                     |

#### 3. preprocessing/（数据预处理模块目录）
| 文件名称               | 核心类               | 类的作用                                                                 |
|------------------------|----------------------|--------------------------------------------------------------------------|
| temporal.py            | TemporalPreprocessor | 时域预处理类，负责音频加载、采样率统一、低通滤波、归一化、阈值去噪等时域操作 |
| spectral.py            | SpectralPreprocessor | 频谱域预处理类，负责提取梅尔频谱、MFCC、STFT等频谱特征，支持频谱图可视化   |
| physics_guided.py      | PhysicsGuidedPreprocessor | 物理引导预处理类，结合水声物理模型，实现声速对齐、衰减补偿、混响去除，优化输入信号 |

#### 4. physics/（物理模型模块目录）
| 文件名称               | 核心类               | 类的作用                                                                 |
|------------------------|----------------------|--------------------------------------------------------------------------|
| underwater_acoustics.py| UnderwaterAcousticModel | 水声物理基础模型，模拟声波水下传播（损失、延迟），计算信号物理一致性分数     |
| differentiable_sim.py  | DifferentiableUnderwaterSimulator | 可微分物理模拟器，支持反向传播，用于训练时的物理约束优化（继承nn.Module） |
| consistency.py         | PhysicsConsistencyEvaluator | 物理一致性评估器，量化观测信号与模拟信号的时域/频谱相似度，检查物理约束符合性 |

#### 5. models/（模型定义目录）
| 文件名称               | 核心类               | 类的作用                                                                 |
|------------------------|----------------------|--------------------------------------------------------------------------|
| base_model.py          | BaseModel            | 所有模型的基类（抽象类），定义通用接口（保存/加载检查点、冻结/解冻层）     |
| cnn_models.py          | ResNet18/SimpleCNN   | 传统CNN基线模型，ResNet18用于高精度特征提取，SimpleCNN用于轻量化基线对比 |
| sota_models.py         | ASTModel             | 音频SOTA模型（Audio Spectrogram Transformer），基于Transformer的频谱分类模型 |
| cnn_phys_reg.py        | CNNPhysRegModel      | CNN+物理正则化模型，在ResNet18基础上加入物理参数预测和一致性约束         |
| n_physound.py          | N_PHYSound           | 项目核心模型，融合物理引导预处理、特征提取、物理参数反演、一致性评分融合分类 |

#### 6. utils/（工具函数目录）
| 文件名称               | 核心类/函数          | 作用                                                                 |
|------------------------|----------------------|--------------------------------------------------------------------------|
| data_loader.py         | UnderwaterSoundDataset/create_data_loaders | 自定义数据集类（加载音频、预处理、返回特征/标签）；创建训练/验证/测试数据加载器 |
| metrics.py             | accuracy/cross_env_accuracy等 | 模型评估指标集合，包括Top-k准确率、跨环境准确率、零样本准确率、推理时间评估 |
| config.py              | load_config/save_config | 加载/保存yaml配置文件，自动补充默认参数、创建必要目录                     |
| data_splitter.py       | split_raw_data       | 数据划分函数，自动将原始数据按比例划分为训练/验证/测试集，生成标签文件     |
| logger.py              | setup_logger/log_experiment_config | 日志配置函数，记录实验过程、配置和指标，同时输出到控制台和日志文件         |

#### 7. experiments/（实验执行脚本目录）
| 文件名称               | 核心函数             | 作用                                                                 |
|------------------------|----------------------|--------------------------------------------------------------------------|
| train_baselines.py     | train_resnet18/train_ast等 | 基线模型训练函数，统一训练流程，支持早停、模型保存、指标打印             |
| train_n_physound.py    | train_n_physound     | N-PHYSound模型专属训练函数，包含物理正则化损失计算、多指标监控           |
| evaluate.py            | evaluate_single_model/evaluate_all_models | 单个模型/所有模型评估函数，输出标准准确率、跨环境准确率等完整指标       |
| visualize.py           | generate_visualizations/各类绘图函数 | 结果可视化函数，生成性能对比图、频谱对比图、一致性分布图等学术图表       |

#### 8. 核心入口文件
| 文件名称               | 作用                                                                 |
|------------------------|----------------------------------------------------------------------|
| main.py                | 项目总入口，支持通过命令行参数选择运行模式（训练/评估/可视化/全部）     |

## 环境搭建
### 1. Conda环境创建与依赖安装
```bash
# 1. 创建虚拟环境（Python3.9兼容性最佳）
conda create -n n-physound python=3.9 -y
conda activate n-physound

# 2. 安装核心依赖（按顺序执行）
# 基础数据处理库
conda install numpy pandas matplotlib scipy seaborn -y
# PyTorch（支持GPU，若无GPU可移除pytorch-cuda=11.7）
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
# 音频处理库
conda install librosa scikit-learn -y
# 额外依赖（pip安装）
pip install soundfile tqdm tensorboard pyyaml
```

### 2. PyCharm环境配置
1. 打开PyCharm，点击「Open」导入本项目文件夹
2. 配置项目解释器：
   - 进入「File > Settings > Project: n-physound > Python Interpreter」（Windows）
   - 或「PyCharm > Settings > Project: n-physound > Python Interpreter」（Mac）
3. 点击右上角「Add Interpreter > Add Local Interpreter」
4. 选择「Conda Environment > Existing environment」
5. 点击「...」找到Conda安装目录下的「envs/n-physound/python.exe」（Windows）或「envs/n-physound/bin/python」（Mac/Linux）
6. 点击「OK」完成配置，等待PyCharm加载依赖

## 数据准备
### 1. 原始数据存放
将水下音频文件（.wav/.flac/.mp3）放入 `data/raw/` 目录，建议音频文件名格式为「{标签}_{序号}.wav」（如「0_001.wav」、「1_002.wav」），方便自动生成标签。

### 2. 自动划分数据集
运行数据划分脚本，自动将原始数据按7:1.5:1.5的比例划分为训练集、验证集、测试集，并生成标签文件：
```bash
# 方式1：终端执行
python utils/data_splitter.py

# 方式2：PyCharm中运行
# 右键点击utils/data_splitter.py > Run 'data_splitter'
```
执行完成后，会自动生成 `data/processed/` 和 `data/splits/` 目录及对应文件。

### 3. 配置文件调整
根据实际数据情况，修改 `configs/main_config.yaml` 中的关键参数：
```yaml
# 数据配置
data:
  sample_rate: 16000        # 音频采样率（与原始数据一致）
  feature_type: mel         # 特征类型（mel/mfcc/stft）
  max_length: 4096          # 音频固定长度（按需调整）
  num_classes: 10           # 类别数量（与实际标签一致）

# 训练配置
train:
  batch_size: 32            # 批次大小（GPU显存不足可改为16/8）
  epochs: 50                # 训练轮次
  lr: 0.0001                # 学习率
  device: cuda              # 设备（cuda/cpu）
```

## 项目启动
### 1. 命令行启动（推荐）
在PyCharm终端或系统终端中，进入项目根目录，执行以下命令：
```bash
# 激活Conda环境（若未激活）
conda activate n-physound

# 可选运行模式
## 模式1：运行全部流程（训练+评估+可视化）
python main.py --mode all

## 模式2：仅训练所有模型
python main.py --mode train

## 模式3：仅评估已训练模型
python main.py --mode evaluate

## 模式4：仅生成可视化结果
python main.py --mode visualize
```

### 2. PyCharm图形化启动
1. 打开 `main.py` 文件
2. 点击文件右上角的运行按钮（绿色三角），或右键点击文件 > 「Run 'main'」
3. 若需修改运行模式，可通过「Run > Edit Configurations」，在「Parameters」中添加 `--mode all`（或其他模式）
4. 点击「Run」开始执行

### 3. 单独运行某一模块
若无需运行全部流程，可单独执行指定脚本：
```bash
# 仅训练ResNet18基线模型
python experiments/train_baselines.py

# 仅训练N-PHYSound核心模型
python experiments/train_n_physound.py

# 仅评估所有模型
python experiments/evaluate.py

# 仅生成可视化图表
python experiments/visualize.py
```

## 结果查看
1. **模型权重**：保存于 `results/checkpoints/`，包含「best_model.pth」（最佳模型）和各轮次检查点
2. **评估报告**：保存于 `results/comparison_results.txt`，包含所有模型的准确率、推理时间等对比数据
3. **可视化图表**：保存于 `results/figures/`，包含性能对比图、频谱对比图、一致性分布图等
4. **训练日志**：保存于 `results/logs/`，包含详细的训练过程、参数配置和指标变化
5. **控制台输出**：实时打印训练进度、验证指标、评估结果，方便实时监控

## 注意事项
1. 若使用GPU训练，需确保已安装对应版本的CUDA（推荐11.7），否则修改 `configs/main_config.yaml` 中 `device: cpu`
2. 音频文件格式若不兼容，可通过 `librosa` 转换为.wav格式后再放入 `data/raw/`
3. 若出现「内存不足」错误，可减小 `configs/main_config.yaml` 中的 `batch_size` 和 `max_length`
4. 训练完成后，建议先评估模型再生成可视化，确保有足够的实验数据用于绘图

## 常见问题
1. **Q：导入模块时提示「No module named 'physics'」？**
   A：确保在项目根目录执行脚本，或在PyCharm中将项目根目录标记为「Sources Root」（右键根目录 > 「Mark Directory as > Sources Root」）。

2. **Q：加载音频时提示「File format not supported」？**
   A：安装额外音频解码器，执行 `pip install ffmpeg-python`，或转换音频为.wav格式。

3. **Q：GPU训练时提示「CUDA out of memory」？**
   A：减小 `batch_size`（如改为16/8）、降低 `max_length`，或使用 `torch.cuda.empty_cache()` 清理显存。

4. **Q：数据划分后标签文件为空？**
   A：检查原始音频文件名格式是否为「{标签}_{序号}.wav」，或修改 `utils/data_splitter.py` 中的标签提取逻辑以匹配自定义文件名。

## 总结
本项目提供了一套完整的物理引导水下声音分类解决方案，各类模块职责清晰、可扩展性强：
- 预处理模块支持多种音频特征提取和物理优化
- 物理模块提供可微分模拟和一致性评估，支撑模型泛化能力
- 模型模块包含从基线到核心的完整模型体系
- 工具和实验模块简化了训练、评估和可视化流程

通过上述启动步骤，即可快速完成从环境搭建到实验落地的全流程，最终得到可用于学术研究或工程应用的音频分类模型。
Cargo") label=0 ;;
    "Passengership") label=1 ;;
    "Tanker") label=2 ;;
    "Tug") label=3 ;;
    *
) continue ;;