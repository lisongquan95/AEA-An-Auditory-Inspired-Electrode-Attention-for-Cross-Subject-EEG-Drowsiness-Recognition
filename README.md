# AEA-An-Auditory-Inspired-Electrode-Attention-for-Cross-Subject-EEG-Drowsiness-Recognition

# AEA-ICNN: 基于电极注意力机制的脑电疲劳检测模型

## 📋 项目简介

本项目实现了一个用于脑电（EEG）疲劳检测的深度学习模型——AEA-ICNN（Electrode Attention ICNN）。能够自动学习不同电极对疲劳检测的重要性，实现高精度的疲劳状态分类。
<img width="7344" height="2813" alt="Fig 2" src="https://github.com/user-attachments/assets/2ede22b6-12c4-4b2a-a2f2-6691553b516a" />
### 核心特点


- **电极注意力机制**：自动学习30个电极通道的重要性权重
- **深度可分离卷积**：高效提取时间-空间特征
- **留一被试交叉验证**：11个被试的交叉验证评估
- **早停机制**：防止过拟合，加速训练
- **余弦退火学习率**：动态调整学习率

## 🏗️ 模型架构

### AEA (Electrode Attention) 模块

```
输入 EEG 信号 (B, 1, 30, 384)
  ↓
计算每个电极的统计特征（最大值、最小值、均值、标准差）
  ↓
生成波动度量 (abs(max - min))
  ↓
平均池化 + 最大池化
  ↓
双路注意力网络 (Conv2d + ReLU + Conv2d + Sigmoid)
  ↓
合并生成注意力权重 (B, 1, 30, 1)
```

### AEA-ICNN 完整网络

```
输入 EEG 信号 (B, 1, 30, 384)
  ↓
AEA 模块 (生成电极注意力权重)
  ↓
逐点卷积 (1 → 16通道)
  ↓
深度可分离卷积 (时间维度特征提取)
  ↓
ReLU 激活
  ↓
批归一化
  ↓
全局平均池化
  ↓
展平
  ↓
全连接层 (2分类: 清醒/疲劳)
  ↓
LogSoftmax
  ↓
输出 (B, 2)
```

## 📁 文件结构

```
├── model.py              # 模型定义文件
│   ├── AEA              # 电极注意力模块
│   └── AEA_ICNN         # 完整网络模型
│
├── train.py             # 训练和测试脚本
│   ├── run()            # 主训练函数
│   └── 超参数搜索       # 批量搜索最佳batch_size
│
└── README.md            # 项目说明文档
```

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
PyTorch >= 1.8.0
NumPy
SciPy
scikit-learn
```

### 安装依赖

```bash
pip install torch numpy scipy scikit-learn
```

### 数据准备

将EEG数据集保存为 `../dataset.mat (https://figshare.com/articles/dataset/EEG_driver_drowsiness_dataset/14273687)` 文件，需要包含以下字段：

- `EEGsample`: EEG信号数据，形状为 (样本数, 30, 384)
- `substate`: 疲劳状态标签，0=清醒，1=疲劳
- `subindex`: 被试编号，1-11

### 运行训练

#### 1. 使用默认参数训练

```python
from train import run

# 执行一次训练
acc = run(batch_size=24, n_epoch=50, lr=0.001, patience=26)
print(f"准确率: {acc:.4f}")
```

#### 2. 批量搜索超参数

```bash
python train.py
```

该脚本会自动搜索batch_size从20到59的最佳值。

#### 3. 单独导入模型使用

```python
from model import AEA_ICNN
import torch

# 初始化模型
model = AEA_ICNN(classes=2)

# 创建随机输入 (batch_size=10, 1通道, 30电极, 384时间点)
input_data = torch.randn(10, 1, 30, 384)

# 前向传播
output = model(input_data)
print(f"输出形状: {output.shape}")  # (10, 2)
```

## 📊 实验设置

### 数据集

- **被试数**: 11人
- **电极数**: 30个
- **采样率**: 128 Hz
- **时长**: 3秒/样本
- **样本长度**: 384个时间点 (3 × 128)
- **类别**: 2类（清醒/疲劳）

### 训练策略

- **交叉验证**: 留一被试交叉验证（Leave-One-Subject-Out, LOSO）
- **优化器**: Adam
- **学习率**: 0.001
- **学习率调度**: 余弦退火
- **早停耐心值**: 26 epochs
- **最大训练轮数**: 50 epochs
- **批量大小**: 20-59（可搜索）

### 评估指标

- **准确率** (Accuracy)
- **召回率** (Recall)
- **F1分数** (F1-Score)

## 📈 性能表现

在默认参数下，模型在11个被试上的表现(不同的服务器会有较大的差距，实验使用Nvidia A40，CDUA 12.8, python 3.8)：

| 指标 | 平均值 |
|------|--------|
| 准确率 | ~83.31% |
| 召回率 | ~81.97% |
| F1分数 | ~83.59% |

*注：具体结果可能因随机种子和数据分布有所不同*

## 🧠 模型原理

### 为什么使用电极注意力？

1. **电极重要性差异**：不同电极记录的大脑区域对疲劳的敏感度不同
2. **噪声过滤**：抑制噪声较大的电极，增强有效信号
3. **自适应权重**：模型自动学习最优的电极组合

### 统计特征的意义

- **最大值-最小值**：反映信号幅度的波动范围
- **均值**：信号的直流偏置
- **标准差**：信号的稳定程度

这些特征能够刻画电极信号的"活跃度"，从而判断电极对疲劳检测的贡献。

## 🔧 调参建议

### Batch Size
- 较小值（20-30）：更频繁的梯度更新，可能更稳定
- 较大值（40-50）：更快的训练，可能收敛更快
- 建议范围：20-60

### Learning Rate
- 默认0.001，可尝试0.0001或0.01
- 配合学习率调度器使用效果更好

### Patience
- 控制早停的耐心值
- 较大值（20-30）：更多训练机会
- 较小值（5-10）：更快停止，防止过拟合

### Epochs
- 建议设置较大的值，配合早停使用
- 默认50通常足够

## 🐛 常见问题

### 1. 显存不足

```python
# 减小batch_size
acc = run(batch_size=16, n_epoch=50)
```

### 2. 训练过慢

```python
# 减少最大epoch数或增大patience
acc = run(batch_size=32, n_epoch=30, patience=10)
```

### 3. 准确率不稳定

- 检查数据是否归一化
- 调整随机种子
- 调整学习率

## 📝 引用

如果您使用本代码，请引用：

```bibtex

  author={Songquan Li, Jun Yuan, Mengyao Li, Hanming Wang, Rongbo,Zhu},
  booktitle={Proceedings of 2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 
  title={Listening to the Brain: An Auditory-Inspired Electrode Attention for Cross-Subject EEG Drowsiness Recognition}, 
  year={2025},
  volume={},
  number={},
  pages={},

## 📄 许可证

MIT License


---

**注意**: 本项目仅用于研究目的，请勿用于临床诊断。

