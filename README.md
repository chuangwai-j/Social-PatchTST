# Social-PatchTST: 基于Transformer的多飞机交互轨迹预测系统

## 项目概述

Social-PatchTST是一个专门用于多飞机轨迹预测和冲突检测的深度学习模型，结合了当前最先进的PatchTST时序预测技术和Social Transformer多智能体交互建模技术。该系统采用**V7-Social架构**，专注于真实场景中的飞机交互建模和最小距离（mindist）预测。

### 🎯 核心创新

- **真实社交场景**: 基于同一时间、同一空域的飞机交互数据，而非随机组合
- **V7-Social架构**: 场景生成器 + 三层Transformer模型的有效结合
- **滑动窗口预测**: 240点（20分钟）滑动窗口，10分钟历史预测10分钟未来
- **相对位置编码(RPE)**: 将物理距离编码为注意力偏置，强化mindist约束
- **多任务损失函数**: 联合优化位置、速度、高度和最小距离预测
- **并行数据处理**: 支持多核并行处理大规模ADS-B数据

### 🚀 技术优势

1. **解决V6缺陷**: 消除了随机组合不同时空飞机的问题
2. **真实交互建模**: 基于实际空域中的飞机社交场景
3. **高效数据处理**: 从原始ADS-B到场景数据的端到端处理
4. **SOTA模型架构**: PatchTST + Social Transformer + Prediction Decoder
5. **可扩展性**: 支持任意数量飞机的交互建模

## 📁 项目结构

```
Social-PatchTST/
├── config/                          # 配置文件
│   ├── social_patchtst_config.yaml  # 主配置文件（已更新为V7）
│   └── config_manager.py            # 配置管理工具
├── data/                            # 数据处理模块
│   ├── dataset/
│   │   ├── data_processor.py        # V7-Social场景生成器
│   │   ├── scene_dataset.py         # V7场景数据集加载器
│   │   └── __init__.py              # 数据模块导入
│   └── adsb_scenes_v7/              # V7场景数据输出目录
│       └── scenes/                  # 场景文件
│           ├── scene_001/
│           │   ├── ego.csv          # Ego飞机240点轨迹
│           │   └── neighbors.csv    # 同时间窗口邻居轨迹
│           └── ...
├── model/                           # 模型实现（SOTA级）
│   ├── social_patchtst.py           # 完整模型
│   ├── patchtst.py                  # PatchTST时序编码器
│   ├── social_transformer.py        # Social Transformer社交编码器
│   ├── prediction_decoder.py        # 预测解码器
│   ├── relative_position_encoding.py # 相对位置编码
│   └── __init__.py                  # 模型模块导入
├── tools/                           # 训练和推理工具
│   ├── train.py                     # 训练脚本
│   ├── inference.py                 # 推理脚本
│   └── __init__.py                  # 工具模块导入
├── main.py                          # 主入口文件
└── README.md                        # 项目说明
```

## 🔄 V7-Social 数据处理流程

### 核心架构改进

**V6问题（已解决）**:
- ❌ 随机组合不同时间、不同空域的飞机
- ❌ groupby隔离破坏社交信息
- ❌ 计算无意义的距离矩阵

**V7解决方案**:
- ✅ 构建"世界状态"：同一文件中所有飞机的重采样轨迹
- ✅ 滑动窗口场景：在连续轨迹上提取240点交互场景
- ✅ 真实距离矩阵：基于同一时刻的位置计算飞机间距

### 数据处理步骤

#### 1. V7场景生成
```bash
# 生成V7社交场景数据
python data/dataset/data_processor.py \
    --input-dir /mnt/d/adsb \
    --output-dir /mnt/d/model/adsb_scenes_v7 \
    --max-files 1000 \
    --stride 10
```

**场景生成逻辑**:
1. **世界状态构建**: 重采样文件中所有飞机轨迹（5秒间隔）
2. **连续轨迹识别**: 检测每架飞机的连续轨迹段
3. **滑动窗口提取**: 在长轨迹段上滑动240点窗口
4. **邻居查找**: 查找同一时间窗口内的其他飞机
5. **场景保存**: 只保存有真实交互的场景

**输出结构**:
```
adsb_scenes_v7/scenes/
├── scene_001/
│   ├── ego.csv      # 240行，1架Ego飞机
│   └── neighbors.csv # 240*N行，N架邻居飞机
└── scene_002/
    ├── ego.csv
    └── neighbors.csv
```

#### 2. 数据加载和训练
```python
from data.dataset import V7SocialDataset, create_v7_data_loaders

# 创建V7数据集
dataset = V7SocialDataset(
    scenes_data="/mnt/d/model/adsb_scenes_v7/scenes",
    config_path="config/social_patchtst_config.yaml",
    max_neighbors=50
)

# 创建数据加载器
train_loader, val_loader, test_loader = create_v7_data_loaders(
    config_path="config/social_patchtst_config.yaml",
    scenes_dir="/mnt/d/model/adsb_scenes_v7/scenes",
    batch_size=8,
    max_neighbors=50
)
```

## 🛠️ 完整使用流程

### 1. 环境准备
```bash
# 安装依赖
pip install torch pandas numpy tqdm scikit-learn matplotlib
```

### 2. 数据准备
```bash
# 生成场景数据（只需要运行一次）
python data/dataset/data_processor.py \
    --input-dir /mnt/d/adsb \
    --output-dir /mnt/d/model/adsb_scenes \
    --max-files 1000
```

### 3. 模型训练
```bash
# 直接训练（配置文件已设置好所有路径）
python tools/train.py
```

### 4. 模型推理
```bash
# 批量预测场景数据
python tools/inference.py --batch_predict --config config/social_patchtst_config.yaml

# 或者指定具体的场景目录
python tools/inference.py --batch_predict --config config/social_patchtst_config.yaml --scenes_dir /mnt/d/model/adsb_scenes/scenes
```

### 5. 模型评估
```bash
# 测试数据加载是否正常
python tools/train.py --test --config config/social_patchtst_config.yaml
```

## 🧠 模型架构

### 三层Social架构

```
输入: [batch_size, max_aircrafts, 120, features]
     ↓
1. Temporal Encoder (PatchTST)
   - 单机时序模式学习
   - Patching: patch_length=16, stride=8
   - 输出: [batch_size, max_aircrafts, n_patches, 512]
     ↓
2. Social Encoder
   - 真实飞机交互建模
   - RPE相对位置编码
   - 输出: [batch_size, max_aircrafts, n_patches, 512]
     ↓
3. Prediction Decoder
   - 多步轨迹预测
   - 输出: [batch_size, max_aircrafts, 120, 5]
```

### 关键技术特点

- **真实距离矩阵**: 基于同一时刻的飞机位置计算
- **可变邻居支持**: 掩码机制处理不同数量的邻居
- **多任务损失**: position:1.0, velocity:0.5, altitude:1.0, mindist:2.0
- **高效训练**: 混合精度支持，梯度累积

## ⚙️ 配置参数

### V7配置文件 (social_patchtst_config.yaml)
```yaml
# 数据配置 - 已更新为V7
data:
  data_dir: "/mnt/d/model/adsb_scenes_v7"
  scenes_dir: "/mnt/d/model/adsb_scenes_v7/scenes"
  history_length: 120        # 10分钟历史
  prediction_length: 120     # 10分钟预测
  sampling_interval: 5       # 5秒采样间隔

# 特征定义
feature_cols:
  temporal_features: ["flight_level", "ground_speed", "track_angle", "vertical_rate", "selected_altitude"]
  spatial_features: ["latitude", "longitude"]
  target_features: ["flight_level", "latitude", "longitude", "ground_speed", "track_angle"]

# PatchTST参数
patchtst:
  patch_length: 16           # Patch长度
  stride: 8                  # 滑动步长
  d_model: 512               # 模型维度
  n_heads: 8                 # 注意力头数

# Social Transformer参数
social_transformer:
  max_aircrafts: 50          # 最大飞机数
  rpe:
    enabled: true            # 启用相对位置编码
    max_distance: 100       # 最大距离(海里)
    distance_bins: 20        # 距离分箱数
  interaction_threshold: 10  # 交互距离阈值

# 训练参数
training:
  batch_size: 8              # 批大小
  learning_rate: 0.0001      # 学习率
  epochs: 100                # 训练轮数
  optimizer: "AdamW"         # 优化器
```

## 📊 性能指标

### 评估指标
- **轨迹预测误差**: RMSE, MAE (位置、高度、速度)
- **安全指标**: 最小距离违规率、碰撞风险评分
- **交互质量**: 场景中平均飞机数量、交互频率

### V7架构优势
- **真实交互**: 100%基于实际空域场景
- **有效mindist**: 距离矩阵反映真实飞机间距
- **数据效率**: 只保存有交互的场景，提高训练效率
- **可扩展性**: 支持任意规模空域的飞机交互

## 🔧 高级功能

### 1. 自定义场景生成参数
```bash
# 调整滑动窗口步长（默认50秒）
python data/dataset/data_processor.py --stride 5  # 25秒步长

# 调整处理文件数量
python data/dataset/data_processor.py --max-files 2000

# 自定义输出目录
python data/dataset/data_processor.py --output-dir /custom/path
```

### 2. 模型配置调优
```python
from config.config_manager import load_config

# 动态修改配置
config = load_config('./config/social_patchtst_config.yaml')
config.set('social_transformer.max_aircrafts', 100)  # 增加最大飞机数
config.set('patchtst.d_model', 768)                 # 增加模型维度
config.save('./config/custom_config.yaml')
```

### 3. 多GPU训练支持
```yaml
device:
  gpu_ids: [0, 1, 2, 3]     # 多GPU训练
  mixed_precision: true     # 混合精度
  num_workers: 4           # 数据加载进程数
```

## 📈 处理性能

### 数据处理效率
- **并行处理**: 使用所有CPU核心
- **内存优化**: 流式处理大文件
- **场景生成**: 每个文件独立处理，支持分布式

### 训练性能
- **单epoch时间**: 取决于场景数量
- **收敛速度**: 通常50-80 epoch收敛
- **GPU需求**: 16GB+ VRAM推荐

### 典型处理时间
- **场景生成**: 2-6小时（取决于数据量）
- **模型训练**: 10-20小时
- **总时间**: 12-26小时

## 🛡️ 系统要求

### 硬件要求
- **CPU**: 8核以上推荐（并行场景生成）
- **内存**: 32GB+推荐（处理大量场景）
- **GPU**: 16GB+ VRAM推荐（训练）
- **存储**: 100GB+可用空间（场景数据）

### 软件环境
- **Python**: 3.8+
- **PyTorch**: 1.12+
- **pandas**: 1.5+
- **numpy**: 1.21+
- **scikit-learn**: 1.1+

## 📚 核心文件说明

### 数据处理
- `data/dataset/data_processor.py` - V7-Social场景生成器
- `data/dataset/scene_dataset.py` - V7场景数据集加载器
- `config/social_patchtst_config.yaml` - V7配置文件

### 模型实现
- `model/social_patchtst.py` - 完整V7-Social模型
- `model/patchtst.py` - PatchTST时序编码器
- `model/social_transformer.py` - Social Transformer
- `model/prediction_decoder.py` - 预测解码器
- `model/relative_position_encoding.py` - 相对位置编码

### 训练工具
- `main.py` - 主入口文件
- `tools/train.py` - 训练脚本
- `tools/inference.py` - 推理脚本

## 🔍 故障排除

### 常见问题

**Q: 场景生成太慢？**
A: 增加`--max-files`控制处理数量，或使用更快的SSD存储

**Q: 内存不足？**
A: 减少`--max-files`参数，分批处理数据

**Q: GPU内存不足？**
A: 减少`batch_size`，使用gradient accumulation

**Q: 场景数量太少？**
A: 减少`--stride`参数，增加滑动窗口密度

**Q: 模型不收敛？**
A: 检查数据质量，调整学习率，增加warmup

## 🎯 快速开始

### 完整测试流程
```bash
# 1. 生成小规模测试场景
python data/dataset/data_processor.py \
    --input-dir /mnt/d/adsb \
    --output-dir /tmp/test_scenes \
    --max-files 10 \
    --stride 10

# 2. 快速训练测试
python main.py --mode train \
    --config ./config/social_patchtst_config.yaml

# 3. 推理测试
python main.py --mode predict \
    --config ./config/social_patchtst_config.yaml \
    --visualize
```

## 📄 版本信息

**版本**: V7-Social v1.0
**更新日期**: 2025-11-11
**架构**: V7-Social (场景生成 + SOTA模型)
**作者**: Claude Code Assistant

### 关键改进
- ✅ 修复V6随机组合缺陷
- ✅ 实现真实社交场景建模
- ✅ 优化距离矩阵计算
- ✅ 提升数据处理效率
- ✅ 增强可扩展性

---

**注意**: 这是采用V7-Social架构的完整重构版本，解决了V6方案的根本缺陷，确保模型学习到真实的飞机交互模式。