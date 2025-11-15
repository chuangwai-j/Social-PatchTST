"""
Social-PatchTST 场景数据集加载器
支持从CSV文件加载分层采样的场景数据，可直接用于模型训练
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import os
from pathlib import Path
import warnings
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config_manager import load_config

warnings.filterwarnings('ignore')


class PrecomputedScaler:
    """使用预计算均值和标准差的标准化器"""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        """
        初始化预计算标准化器

        Args:
            mean: 特征均值数组
            std: 特征标准差数组
        """
        self.mean_ = mean.astype(np.float64)
        self.scale_ = std.astype(np.float64)

        # 避免除零错误
        self.scale_[self.scale_ == 0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        应用标准化变换

        Args:
            X: 输入数据 [n_samples, n_features]

        Returns:
            标准化后的数据
        """
        return (X.astype(np.float64) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """兼容方法，直接返回transform结果"""
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        反向标准化变换

        Args:
            X: 标准化后的数据

        Returns:
            反标准化后的数据
        """
        return (X.astype(np.float64) * self.scale_) + self.mean_

# 实际CSV中的列名定义
CSV_FEATURE_COLUMNS = {
    'temporal_features': [
        'latitude', 'longitude',      # 局部 ENU (m)
        'flight_level',              # 气压高度(米)
        'ground_speed', 'track_angle'  # 将用于计算 vx, vy (m/s)
    ],
    'spatial_features': ['latitude', 'longitude'],
    'target_features': ['latitude', 'longitude'],
    'static_features': ['aircraft_type', 'callsign', 'target_address']
}


class SocialPatchTSTDataset(Dataset):
    """
    Social-PatchTST 场景数据集
    直接从train/val/test文件夹中按顺序读取场景数据
    """

    def __init__(self, data_dir: str, max_neighbors: int = 20, sequence_length: int = 600, paths_file: str = None):
        """
        从数据目录初始化场景数据集

        Args:
            data_dir: 数据根目录路径
            max_neighbors: 最大邻居数量
            sequence_length: 序列长度
            paths_file: 路径文件txt (train_paths.txt, val_paths.txt, test_paths.txt)
        """
        self.data_dir = Path(data_dir)
        self.max_neighbors = max_neighbors
        self.sequence_length = sequence_length

        # 获取特征列定义
        self.temporal_features = CSV_FEATURE_COLUMNS['temporal_features']
        self.spatial_features = CSV_FEATURE_COLUMNS['spatial_features']
        self.target_features = CSV_FEATURE_COLUMNS['target_features']

        print(f"📂 从路径文件加载场景: {paths_file}")
        # 高效读取路径文件，去重处理
        self.scenes = []
        seen_scenes = set()  # 防重复

        if paths_file and os.path.exists(paths_file):
            with open(paths_file, 'r') as f:
                for line_num, line in enumerate(f):
                    scene_path = line.strip()
                    if not scene_path:
                        continue

                    scene_name = os.path.basename(scene_path)

                    # 防重复检查
                    if scene_name in seen_scenes:
                        continue
                    seen_scenes.add(scene_name)

                    ego_path = os.path.join(scene_path, "ego.csv")
                    neighbor_path = os.path.join(scene_path, "neighbors.csv")

                    self.scenes.append({
                        'scene_id': scene_name,
                        'ego_path': ego_path,
                        'neighbor_path': neighbor_path,
                        'layer': self._extract_layer_from_name(scene_name)
                    })

                    # 减少打印频率 - 每50k个场景打印一次
                    if len(self.scenes) % 50000 == 0:
                        print(f"   已加载 {len(self.scenes)} 个唯一场景...")

        print(f"✅ 发现 {len(self.scenes)} 个唯一场景")

        # 快速验证数据完整性
        print("🔍 验证数据完整性...")
        self._verify_data_integrity()

        # 初始化标准化器
        self._initialize_scalers()

    def _extract_layer_from_name(self, scene_name: str) -> str:
        """从场景名称中提取层级信息"""
        # 这里可以根据你的命名规则来提取层级
        # 暂时返回默认值
        return "default"

    def _verify_data_integrity(self):
        """跳过数据完整性验证，避免扫描文件夹"""
        print("⚡ 跳过完整性验证，直接使用路径文件中的场景")
        self.valid_scenes = self.scenes
        print(f"✅ 直接使用全部场景数量: {len(self.valid_scenes)}")

    def _initialize_scalers(self):
        """从配置文件初始化数据标准化器，使用预计算的统计信息"""
        print("🔧 从配置文件初始化数据标准化器...")

        try:
            # 加载配置文件
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                     "config", "social_patchtst_config.yaml")
            config = load_config(config_path)

            # 获取统计数据
            statistics = config.get('data.statistics', {})

            if not statistics:
                print("⚠️  配置文件中未找到统计信息，将使用原始数据")
                self.feature_scaler = None
                return

            # 使用主要特征的统计信息 (latitude, longitude, flight_level, vx, vy)
            main_stats = statistics.get('main_features', {})

            if not main_stats.get('mean') or not main_stats.get('std'):
                print("⚠️  配置文件中缺少主要特征的统计信息，将使用原始数据")
                self.feature_scaler = None
                return

            # 创建自定义的标准化器，使用预计算的均值和标准差
            self.feature_scaler = PrecomputedScaler(
                mean=np.array(main_stats['mean']),
                std=np.array(main_stats['std'])
            )

            print(f"✅ 标准化器已从配置文件加载")
            print(f"   特征顺序: {main_stats['feature_names']}")
            print(f"   均值: {main_stats['mean']}")
            print(f"   标准差: {main_stats['std']}")

        except Exception as e:
            print(f"⚠️  从配置文件加载统计信息失败: {e}")
            print("   将使用原始数据")
            self.feature_scaler = None

    def _process_features(self, df):
        """
        处理特征：提取基本特征并计算速度向量
        Args:
            df: 原始DataFrame
        Returns:
            processed_features: 处理后的特征数组 [seq_len, 5]
        """
        # 提取基本特征
        lat = df['latitude'].values
        lon = df['longitude'].values
        flight_level = df['flight_level'].values
        ground_speed = df['ground_speed'].values
        track_angle = df['track_angle'].values

        # 转换速度向量 (m/s)
        # 注意：track_angle 单位是度，需要转换为弧度
        track_rad = np.deg2rad(track_angle)
        vx = ground_speed * np.sin(track_rad)  # 东向速度
        vy = ground_speed * np.cos(track_rad)  # 北向速度

        # 组合特征 [lat, lon, flight_level, vx, vy]
        processed_features = np.column_stack([
            lat, lon, flight_level, vx, vy
        ])

        return processed_features

    def _load_single_scene(self, idx):
        """加载单个场景的数据"""
        scene = self.valid_scenes[idx]
        scene_id = scene['scene_id']

        try:
            # 加载ego数据
            ego_df = pd.read_csv(scene['ego_path'])
            ego_features = self._process_features(ego_df)  # [seq_len, 5]

            # 加载neighbor数据
            neighbors_df = pd.read_csv(scene['neighbor_path'])

            # 选择最多max_neighbors个邻居（保持顺序）
            if len(neighbors_df) > self.max_neighbors:
                neighbors_df = neighbors_df.head(self.max_neighbors)

            neighbor_features_list = []
            # 按飞机ID分组处理邻居数据
            for aircraft_id, neighbor_group in neighbors_df.groupby('target_address'):
                neighbor_features = self._process_features(neighbor_group)
                neighbor_features_list.append(neighbor_features)

            return {
                'scene_id': scene_id,
                'ego_features': ego_features,
                'neighbor_features': neighbor_features_list,
                'layer': scene['layer']
            }

        except Exception as e:
            print(f"⚠️  加载场景 {scene_id} 失败: {e}")
            return None

    def __len__(self):
        return len(self.valid_scenes)

    def __getitem__(self, idx):
        """获取单个数据样本，转换为模型期望的格式"""
        scene_data = self._load_single_scene(idx)

        if scene_data is None:
            # 返回空样本
            n_aircrafts = self.max_neighbors + 1  # ego + neighbors
            seq_len = self.sequence_length
            n_temporal_features = len(self.temporal_features)

            return {
                'scene_id': f"empty_{idx}",
                'temporal': torch.zeros(n_aircrafts, seq_len, n_temporal_features),
                'spatial': torch.zeros(n_aircrafts, 2),  # lat, lon
                'targets': torch.zeros(n_aircrafts, 120, 4),  # pred_len, targets
                'distance_matrix': torch.eye(n_aircrafts),  # 单位矩阵
                'aircraft_ids': [f"empty_{i}" for i in range(n_aircrafts)],
                'layer': 'Unknown'
            }

        # 处理ego特征
        ego_features = scene_data['ego_features']  # [seq_len, 5]
        if self.feature_scaler is not None:
            ego_features = self.feature_scaler.transform(ego_features)

        # 确保序列长度一致
        if len(ego_features) > self.sequence_length:
            ego_features = ego_features[:self.sequence_length]
        elif len(ego_features) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(ego_features), ego_features.shape[1]))
            ego_features = np.vstack([ego_features, padding])

        # 处理邻居特征
        neighbor_features_list = scene_data['neighbor_features']
        n_aircrafts = min(len(neighbor_features_list) + 1, self.max_neighbors + 1)  # +1 for ego

        # 初始化张量
        temporal_data = torch.zeros(n_aircrafts, self.sequence_length, len(self.temporal_features))
        spatial_data = torch.zeros(n_aircrafts, 2)  # lat, lon
        aircraft_ids = ['ego']

        # 第0架飞机是ego
        temporal_data[0] = torch.from_numpy(ego_features).float()
        spatial_data[0] = torch.from_numpy(ego_features[-1, :2]).float()  # 最后位置的lat, lon

        # 填充邻居数据
        for i, neigh_feat in enumerate(neighbor_features_list[:self.max_neighbors]):
            if i + 1 >= n_aircrafts:
                break

            if neigh_feat.ndim == 1:
                neigh_feat = neigh_feat.reshape(1, -1)

            if self.feature_scaler is not None:
                neigh_feat = self.feature_scaler.transform(neigh_feat)

            if len(neigh_feat) > self.sequence_length:
                neigh_feat = neigh_feat[:self.sequence_length]
            elif len(neigh_feat) < self.sequence_length:
                padding = np.zeros((self.sequence_length - len(neigh_feat), neigh_feat.shape[1]))
                neigh_feat = np.vstack([neigh_feat, padding])

            temporal_data[i + 1] = torch.from_numpy(neigh_feat).float()
            spatial_data[i + 1] = torch.from_numpy(neigh_feat[-1, :2]).float()
            aircraft_ids.append(f"neighbor_{i}")

        # 创建距离矩阵 (基于当前位置)
        distance_matrix = torch.zeros(n_aircrafts, n_aircrafts)
        for i in range(n_aircrafts):
            for j in range(n_aircrafts):
                if i != j:
                    # 计算欧几里得距离
                    dist = torch.norm(spatial_data[i] - spatial_data[j])
                    distance_matrix[i, j] = dist

        # 创建目标数据 (基于最后的位置)
        # 简化：目标是预测未来位置，这里使用最后位置作为目标基础
        last_position = temporal_data[:, -1, :4]  # [n_aircrafts, 4] - lat,lon,flight_level,vx
        targets = last_position.unsqueeze(1).repeat(1, 120, 1)  # [n_aircrafts, 120, 4]

        # 返回数据，不要添加batch维度（DataLoader会处理）
        return {
            'scene_id': scene_data['scene_id'],
            'temporal': temporal_data,  # [n_aircrafts, seq_len, features]
            'spatial': spatial_data,    # [n_aircrafts, 2]
            'targets': targets,         # [n_aircrafts, 120, 4]
            'distance_matrix': distance_matrix,  # [n_aircrafts, n_aircrafts]
            'aircraft_ids': aircraft_ids,  # List of IDs
            'layer': scene_data['layer']
        }


def create_social_patchtst_loaders(config_path: str = None, batch_size: int = 32,
                                  max_neighbors: int = 20, sequence_length: int = 600,
                                  num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建Social-PatchTST数据加载器

    Args:
        config_path: 配置文件路径，如果为None则使用默认配置
        batch_size: 批大小
        max_neighbors: 每个场景最大邻居数量
        sequence_length: 序列长度
        num_workers: 数据加载器工作进程数

    Returns:
        train_loader, val_loader, test_loader
    """
    # 加载配置文件
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "config", "social_patchtst_config.yaml")

    config = load_config(config_path)

    # 从配置文件获取数据目录
    scenes_dir = config.get('data.scenes_dir') or config.get('data.data_dir')
    if not scenes_dir:
        raise ValueError("配置文件中未找到 data.scenes_dir 或 data.data_dir")

    scenes_path = Path(scenes_dir)

    # 路径文件路径
    train_paths_file = scenes_path / "train_paths.txt"
    val_paths_file = scenes_path / "val_paths.txt"
    test_paths_file = scenes_path / "test_paths.txt"

    print("🚀 创建Social-PatchTST数据加载器")
    print(f"   配置文件: {config_path}")
    print(f"   数据目录: {scenes_path}")
    print(f"   训练路径: {train_paths_file}")
    print(f"   验证路径: {val_paths_file}")
    print(f"   测试路径: {test_paths_file}")

    # 检查路径文件是否存在
    if not train_paths_file.exists():
        raise FileNotFoundError(f"训练路径文件不存在: {train_paths_file}")
    if not val_paths_file.exists():
        raise FileNotFoundError(f"验证路径文件不存在: {val_paths_file}")
    if not test_paths_file.exists():
        raise FileNotFoundError(f"测试路径文件不存在: {test_paths_file}")

    # 创建数据集
    train_dataset = SocialPatchTSTDataset(str(scenes_path), max_neighbors, sequence_length, str(train_paths_file))
    val_dataset = SocialPatchTSTDataset(str(scenes_path), max_neighbors, sequence_length, str(val_paths_file))
    test_dataset = SocialPatchTSTDataset(str(scenes_path), max_neighbors, sequence_length, str(test_paths_file))

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,  # 如果有worker就保持存活
        pin_memory=True  # 加速CPU到GPU传输
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True
    )

    print("✅ 数据加载器创建成功!")
    print(f"   训练集: {len(train_dataset)} 样本 ({len(train_loader)} batches)")
    print(f"   验证集: {len(val_dataset)} 样本 ({len(val_loader)} batches)")
    print(f"   测试集: {len(test_dataset)} 样本 ({len(test_loader)} batches)")
    print(f"   特征维度: {len(CSV_FEATURE_COLUMNS['temporal_features'])}")

    return train_loader, val_loader, test_loader


def get_feature_info():
    """获取特征信息"""
    return {
        'temporal_features': CSV_FEATURE_COLUMNS['temporal_features'],
        'spatial_features': CSV_FEATURE_COLUMNS['spatial_features'],
        'target_features': CSV_FEATURE_COLUMNS['target_features'],
        'n_temporal_features': len(CSV_FEATURE_COLUMNS['temporal_features']),
        'n_target_features': len(CSV_FEATURE_COLUMNS['target_features'])
    }