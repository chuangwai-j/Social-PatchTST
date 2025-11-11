# Social-PatchTST: 基于Transformer的多飞机交互轨迹预测系统

## 项目概述

Social-PatchTST是一个专门用于多飞机轨迹预测和冲突检测的深度学习模型，结合了当前最先进的PatchTST时序预测技术和Social Transformer多智能体交互建模技术。该系统能够同时建模多架飞机的飞行模式和相互之间的空间关系，实现高精度的轨迹预测和安全监控。

### 🎯 核心优势

- **真实场景建模**: 基于同一时间、同一空域的真实飞机交互数据
- **多智能体注意力**: 使用Transformer自注意力机制建模飞机间的社交交互
- **时空Patch机制**: 将长时间序列分割为重叠的patch，有效捕捉局部时序模式
- **相对位置编码**: 将物理距离编码为注意力偏置，强化安全约束
- **多任务学习**: 联合优化位置、速度、高度和最小距离预测
- **端到端训练**: 从原始ADS-B数据到轨迹预测的完整流程

### 🚀 技术特色

1. **Social Transformer架构**
   - 建模飞机间的复杂交互关系
   - 支持可变数量的飞机同时预测
   - 相对位置编码增强空间感知

2. **PatchTST时序编码**
   - 将240点轨迹分割为重叠patch
   - 降低计算复杂度，提升预测精度
   - 有效捕捉短期和长期时序依赖

3. **多任务损失函数**
   - 位置预测：准确的航迹预测
   - 速度预测：考虑动态特性
   - 高度预测：支持垂直机动
   - 距离约束：最小安全距离监控

## 📁 项目结构

```
Social-PatchTST/
├── config/
│   ├── social_patchtst_config.yaml  # 主配置文件
│   └── config_manager.py            # 配置管理工具
├── data/
│   └── dataset/
│       ├── scene_dataset.py         # 场景数据集��载器
│       └── data_processor.py         # 场景数据生成器
├── model/
│   ├── social_patchtst.py           # 完整模型
│   ├── patchtst.py                  # 时序编码器
│   ├── social_transformer.py        # 社交编码器
│   ├── prediction_decoder.py        # 预测解码器
│   └── relative_position_encoding.py # 相对位置编码
├── tools/
│   ├── train.py                     # 训练脚本
│   └── inference.py                 # 推理脚本
└── README.md                        # 项目说明
```

## 🔄 数据处理流程

### 场景数据生成

系统将原始ADS-B数据处理为交互场景：

```
原始ADS-B数据 → 场景生成器 → 交互场景
                          ├─ 场景001/
                          │   ├─ ego.csv      # 目标飞机240点轨迹
                          │   └─ neighbors.csv # 同时空域邻居轨迹
                          └─ 场景002/
                              ├─ ego.csv
                              └─ neighbors.csv
```

**场景生成特性**:
- **滑动窗口**: 240点（20分钟）窗口，10点滑动步长
- **时空一致性**: 同一时刻、同一空域的飞机交互
- **质量控制**: 只保留有真实交互的场景
- **并行处理**: 多核并行加速大规模数据处理

## 🛠️ 使用方法

### 1. 环境准备
```bash
# 安装依赖
pip install torch pandas numpy tqdm scikit-learn matplotlib
```

### 2. 数据准备
```bash
# 从ADS-B数据生成场景（一次运行）
python data/dataset/data_processor.py \
    --input-dir /mnt/d/adsb \
    --output-dir /mnt/d/model/adsb_scenes \
    --max-files 1000 \
    --stride 10
```

### 3. 模型训练
```bash
# 直接训练（配置文件已包含所有路径）
python tools/train.py
```

### 4. 批量预测
```bash
# 批量预测场景数据
python tools/inference.py --batch_predict --config config/social_patchtst_config.yaml
```

### 5. 模型测试
```bash
# 测试数据加载和模型功能
python tools/train.py --test --config config/social_patchtst_config.yaml
```

## 🧠 模型架构

### 三层Transformer架构

```
输入: [batch_size, max_aircrafts, 120, features]
     ↓
1. Temporal Encoder (PatchTST)
   - 单机时序模式学习
   - Patching: patch_length=16, stride=8
   - 输出: [batch_size, max_aircrafts, n_patches, 512]
     ↓
2. Social Encoder
   - 多机交互建模
   - 相对位置编码
   - 输出: [batch_size, max_aircrafts, n_patches, 512]
     ↓
3. Prediction Decoder
   - 多步轨迹预测
   - 输出: [batch_size, max_aircrafts, 120, 5]
```

### 关键技术

- **真实距离矩阵**: 基于同一时刻的位置计算飞机间距
- **可变邻居支持**: 掩码机制处理不同数量的邻居
- **高效注意力**: Patch机制降低O(n²)复杂度
- **混合精度训练**: 支持FP16加速训练

## ⚙️ 配置参数

### 核心配置
```yaml
# 数据配置
data:
  history_length: 120        # 历史序列长度 (10分钟)
  prediction_length: 120     # 预测序列长度 (10分钟)
  sampling_interval: 5       # 采样间隔（秒）

# 模型参数
patchtst:
  patch_length: 16           # Patch长度
  stride: 8                  # 滑动步长
  d_model: 512               # 模型维度

# 社交交互
social_transformer:
  max_aircrafts: 50          # 最大飞机数
  rpe:
    max_distance: 100       # 最大考虑距离（海里）
    distance_bins: 20        # 距离分箱数
  interaction_threshold: 10  # 交互距离阈值

# 训练参数
training:
  batch_size: 4             # 批大小
  learning_rate: 0.0001      # 学习率
  epochs: 100                # 训练轮数
```

## 📊 性能指标

### 评估维度

**轨迹预测精度**
- 位置误差：RMSE, MAE (经纬度)
- 高度误差：RMSE, MAE (气压高度)
- 速度误差：RMSE, MAE (地速、航向)

**安全性指标**
- 最小距离违规率
- 碰撞风险评分
- 交互场景识别准确率

**系统性能**
- 数据处理效率：场景生成速度
- 训练效率：每epoch时间
- 推理延迟：实时预测能力

## 🔧 高级功能

### 1. 自定义场景生成
```bash
# 调整滑动窗口密度
python data/dataset/data_processor.py --stride 5

# 处理更多原始文件
python data/dataset/data_processor.py --max-files 500
```

### 2. 模型调优
```python
from config.config_manager import load_config

# 动态修改配置
config = load_config('config/social_patchtst_config.yaml')
config.set('social_transformer.max_aircrafts', 100)
config.set('patchtst.d_model', 768)
config.save('config/custom_config.yaml')
```

### 3. 多GPU训练
```yaml
device:
  gpu_ids: [0, 1, 2, 3]     # 多GPU训练
  mixed_precision: true     # 混合精度
```

## 📈 应用场景

### 航空交通管理
- **冲突预测**: 提前识别潜在飞行冲突
- **路径优化**: 推荐安全高效的飞行路径
- **流量管理**: 优化空域利用率

### 飞行安全监控
- **异常检测**: 识别异常飞行行为
- **风险预警**: 实时安全风险评估
- **事后分析**: 飞行事件调查支持

### 航空研究
- **轨迹建模**: 空中交通模式研究
- **行为分析**: 飞行员行为模式研究
- **政策评估**: 空管政策效果评估

## 🛡️ 系统要求

### 硬件配置
- **CPU**: 8核以上推荐（场景生成）
- **内存**: 32GB+推荐（大规模数据处理）
- **GPU**: 16GB+ VRAM推荐（模型训练）
- **存储**: 500GB+可用空间（场景数据）

### 软件环境
- **Python**: 3.8+
- **PyTorch**: 1.12+
- **pandas**: 1.5+
- **numpy**: 1.21+

## 📚 核心文件说明

### 数据处理
- `data/dataset/data_processor.py` - 场景数据生成器
- `data/dataset/scene_dataset.py` - 场景数据集加载器

### 模型实现
- `model/social_patchtst.py` - 完整模型实现
- `model/patchtst.py` - 时序编码器
- `model/social_transformer.py` - 社交编码器
- `model/prediction_decoder.py` - 预测解码器

### 工具脚本
- `tools/train.py` - 训练脚本（支持测试模式）
- `tools/inference.py` - 推理脚本（支持批量预测）

## 🎯 快速开始

### 完整工作流
```bash
# 1. 生成场景数据
python data/dataset/data_processor.py \
    --input-dir /mnt/d/adsb \
    --output-dir /mnt/d/model/adsb_scenes \
    --max-files 100

# 2. 训练模型
python tools/train.py

# 3. 预测验证
python tools/inference.py --batch_predict
```

## 📄 项目信息

**版本**: 完整实现版本
**更新日期**: 2025-11-11
**架构**: Social-PatchTST (场景数据 + 三层Transformer)
**数据支持**: ADS-B轨迹数据

---

**注意**: 这是一个面向航空研究和应用的完整系统，包含从原始数据处理到模型预测的全套工具链。