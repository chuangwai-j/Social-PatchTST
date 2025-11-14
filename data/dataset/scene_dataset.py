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
        # 从txt文件读取场景路径
        self.scenes = []
        if paths_file and os.path.exists(paths_file):
            with open(paths_file, 'r') as f:
                for line in f:
                    scene_path = line.strip()
                    if scene_path:
                        scene_name = os.path.basename(scene_path)
                        ego_path = os.path.join(scene_path, "ego.csv")
                        neighbor_path = os.path.join(scene_path, "neighbors.csv")

                        if os.path.exists(ego_path) and os.path.exists(neighbor_path):
                            self.scenes.append({
                                'scene_id': scene_name,
                                'ego_path': ego_path,
                                'neighbor_path': neighbor_path,
                                'layer': self._extract_layer_from_name(scene_name)
                            })
        print(f"✅ 发现 {len(self.scenes)} 个有效场景")

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
        """验证数据完整性"""
        # 抽样验证前100个场景
        sample_size = min(100, len(self.scenes))
        valid_count = 0

        for idx in range(sample_size):
            scene = self.scenes[idx]
            ego_path = scene['ego_path']
            neighbor_path = scene['neighbor_path']

            if os.path.exists(ego_path) and os.path.exists(neighbor_path):
                valid_count += 1

        validity_rate = valid_count / sample_size
        if validity_rate >= 0.9:
            print(f"✅ 数据完整性良好 ({validity_rate:.1%})，使用全部场景")
            self.valid_scenes = self.scenes
        else:
            print(f"⚠️  数据完整性较低 ({validity_rate:.1%})，建议检查数据")
            self.valid_scenes = self.scenes  # 仍使用全部数据

        print(f"最终使用场景数量: {len(self.valid_scenes)}")

    def _initialize_scalers(self):
        """初始化数据标准化器"""
        print("🔧 初始化数据标准化器...")

        sample_size = min(50, len(self.valid_scenes))
        all_features = []

        for i in range(sample_size):
            try:
                scene = self.valid_scenes[i]

                # 加载ego数据并处理特征
                ego_df = pd.read_csv(scene['ego_path'])
                ego_features = self._process_features(ego_df)
                all_features.append(ego_features)

                # 加载邻居数据样本并处理特征
                neighbors_df = pd.read_csv(scene['neighbor_path'])
                # 处理前几个邻居来收集特征
                neighbor_groups = neighbors_df.groupby('target_address')
                for aircraft_id, neighbor_group in list(neighbor_groups)[:3]:  # 限制为前3个邻居
                    neighbor_features = self._process_features(neighbor_group)
                    all_features.append(neighbor_features)

            except Exception as e:
                continue

        if all_features:
            all_features = np.vstack(all_features)
            self.feature_scaler = StandardScaler()
            self.feature_scaler.fit(all_features)
            print(f"✅ 标准化器已拟合，特征维度: {all_features.shape}")
        else:
            self.feature_scaler = None
            print("⚠️  无法拟合标准化器，将使用原始数据")

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
        """获取单个数据样本"""
        scene_data = self._load_single_scene(idx)

        if scene_data is None:
            # 返回空样本
            return {
                'scene_id': f"empty_{idx}",
                'ego_features': torch.zeros(self.sequence_length, 5),  # 5维特征
                'neighbor_features': torch.zeros(self.max_neighbors, self.sequence_length, 5),  # 5维特征
                'target': torch.zeros(2),  # [lat, lon]
                'layer': 'Unknown'
            }

        # 处理ego特征
        ego_features = scene_data['ego_features']
        if self.feature_scaler is not None:
            ego_features = self.feature_scaler.transform(ego_features)

        # 确保序列长度一致
        if len(ego_features) > self.sequence_length:
            ego_features = ego_features[:self.sequence_length]
        elif len(ego_features) < self.sequence_length:
            # 填充
            padding = np.zeros((self.sequence_length - len(ego_features), ego_features.shape[1]))
            ego_features = np.vstack([ego_features, padding])

        # 处理邻居特征
        neighbor_features = scene_data['neighbor_features']
        neighbor_tensor = torch.zeros(self.max_neighbors, self.sequence_length, len(self.temporal_features))

        for i, neigh_feat in enumerate(neighbor_features[:self.max_neighbors]):
            if neigh_feat.ndim == 1:
                neigh_feat = neigh_feat.reshape(1, -1)

            if self.feature_scaler is not None:
                neigh_feat = self.feature_scaler.transform(neigh_feat)

            if len(neigh_feat) > self.sequence_length:
                neigh_feat = neigh_feat[:self.sequence_length]
            elif len(neigh_feat) < self.sequence_length:
                padding = np.zeros((self.sequence_length - len(neigh_feat), neigh_feat.shape[1]))
                neigh_feat = np.vstack([neigh_feat, padding])

            neighbor_tensor[i] = torch.from_numpy(neigh_feat).float()

        # 创建目标（使用最后一个时间步的位置作为预测目标）
        # 新的特征顺序: [latitude(0), longitude(1), flight_level(2), vx(3), vy(4)]
        target_data = ego_features[-1, [0, 1]]  # [lat, lon]

        return {
            'scene_id': scene_data['scene_id'],
            'ego_features': torch.from_numpy(ego_features).float(),
            'neighbor_features': neighbor_tensor,
            'target': torch.from_numpy(target_data).float(),
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
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
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