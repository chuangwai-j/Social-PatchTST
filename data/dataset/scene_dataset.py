"""
V7-Social 场景数据集加载器
适配新的场景数据结构 (ego.csv + neighbors.csv)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional
import os
import glob
import warnings
from config.config_manager import load_config

warnings.filterwarnings('ignore')


class SceneDataset(Dataset):
    """场景数据集"""

    def __init__(self, scenes_data, config_path: str, max_neighbors: int = 50):
        """
        初始化场景数据集

        Args:
            scenes_data: 可以是场景目录路径或场景路径列表
            config_path: 配置文件路径
            max_neighbors: 每个场景最大邻居数量
        """
        self.config = load_config(config_path)
        self.data_config = self.config.data_config
        self.max_neighbors = max_neighbors

        # 获取特征列定义
        self.temporal_features = self.data_config['feature_cols']['temporal_features']
        self.spatial_features = self.data_config['feature_cols']['spatial_features']
        self.static_features = self.data_config['feature_cols']['static_features']
        self.target_features = self.data_config['feature_cols']['target_features']

        # 处理输入参数
        if isinstance(scenes_data, str):
            # 如果是字符串，假设是目录路径
            scenes_dir = scenes_data
            self.scene_dirs = []
            for scene_id in os.listdir(scenes_dir):
                scene_path = os.path.join(scenes_dir, scene_id)
                if os.path.isdir(scene_path):
                    ego_path = os.path.join(scene_path, "ego.csv")
                    neighbors_path = os.path.join(scene_path, "neighbors.csv")
                    if os.path.exists(ego_path) and os.path.exists(neighbors_path):
                        self.scene_dirs.append(scene_path)
        else:
            # 如果是列表，假设是场景路径列表
            self.scene_dirs = scenes_data

        print(f"���到 {len(self.scene_dirs)} 个有效场景")

        # 初始化归一化器和编码器（需要在第一个场景上拟合）
        self.scalers = {}
        self.encoders = {}
        self.is_fitted = False

        # 拟合预处理器
        self._fit_processors()

    def _fit_processors(self):
        """在第一个场景上拟合预处理器"""
        if not self.scene_dirs:
            raise ValueError("没有找到有效的场景数据")

        print("拟合数据预处理器...")

        # 使用第一个场景的数据进行拟合
        first_scene = self.scene_dirs[0]
        ego_df = pd.read_csv(os.path.join(first_scene, "ego.csv"))
        neighbors_df = pd.read_csv(os.path.join(first_scene, "neighbors.csv"))

        # 合并所有数据用于拟合
        all_data = pd.concat([ego_df, neighbors_df], ignore_index=True)

        # 为数值特征拟合StandardScaler
        numeric_features = self.temporal_features + self.spatial_features
        for feature in numeric_features:
            if feature in all_data.columns:
                self.scalers[feature] = StandardScaler()
                values = all_data[feature].values.reshape(-1, 1)
                self.scalers[feature].fit(values)
                print(f"  - {feature}: mean={self.scalers[feature].mean_[0]:.3f}, std={self.scalers[feature].scale_[0]:.3f}")

        # 为分类特征拟合LabelEncoder
        categorical_features = ['aircraft_type']
        for feature in categorical_features:
            if feature in all_data.columns:
                self.encoders[feature] = LabelEncoder()
                values = all_data[feature].astype(str).values
                self.encoders[feature].fit(values)
                print(f"  - {feature}: {len(self.encoders[feature].classes_)} 个类别")

        self.is_fitted = True
        print("数据预处理器拟合完成!")

    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据（归一化和编码）"""
        if not self.is_fitted:
            raise ValueError("预处理器未拟合")

        df_transformed = df.copy()

        # 数值特征归一化
        numeric_features = self.temporal_features + self.spatial_features
        for feature in numeric_features:
            if feature in df_transformed.columns and feature in self.scalers:
                values = df_transformed[feature].values.reshape(-1, 1)
                df_transformed[feature] = self.scalers[feature].transform(values).flatten()

        # 分类特征编码
        for feature in ['aircraft_type']:
            if feature in df_transformed.columns and feature in self.encoders:
                values = df_transformed[feature].astype(str).values
                # 处理未见过的类别
                mask = ~np.isin(values, self.encoders[feature].classes_)
                if mask.any():
                    values[mask] = self.encoders[feature].classes_[0]
                df_transformed[feature] = self.encoders[feature].transform(values)

        return df_transformed

    def _calculate_distance_matrix(self, ego_positions: np.ndarray, neighbor_positions: np.ndarray) -> np.ndarray:
        """
        计算距离矩阵

        Args:
            ego_positions: Ego飞机的位置 (240, 2) -> [lat, lon]
            neighbor_positions: 邻居飞机的位置 (N, 240, 2)

        Returns:
            distance_matrix: (N+1, N+1) 的距离矩阵
        """
        history_length = self.data_config['history_length']

        # 使用历史轨迹的最后一个位置计算距离
        ego_last_pos = ego_positions[history_length-1]  # (2,)
        neighbor_last_positions = neighbor_positions[:, history_length-1, :]  # (N, 2)

        # 初始化距离矩阵 (ego + neighbors)
        n_neighbors = len(neighbor_last_positions)
        distance_matrix = np.zeros((n_neighbors + 1, n_neighbors + 1))

        # 计算邻居到ego的距离
        for i in range(n_neighbors):
            dist = self._haversine_distance(
                ego_last_pos[0], ego_last_pos[1],
                neighbor_last_positions[i, 0], neighbor_last_positions[i, 1]
            )
            distance_matrix[0, i+1] = dist
            distance_matrix[i+1, 0] = dist

        # 计算邻居之间的距离
        for i in range(n_neighbors):
            for j in range(n_neighbors):
                if i != j:
                    dist = self._haversine_distance(
                        neighbor_last_positions[i, 0], neighbor_last_positions[i, 1],
                        neighbor_last_positions[j, 0], neighbor_last_positions[j, 1]
                    )
                    distance_matrix[i+1, j+1] = dist

        return distance_matrix

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """计算两点间的大圆距离（海里）"""
        # 转换为弧度
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # 使用Haversine公式
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # 地球半径（海里）
        earth_radius_nm = 3440.065

        return c * earth_radius_nm

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        """
        获取单个场景的数据

        Returns:
            dict: 包含以下键:
                - temporal: (max_aircrafts, history_length, n_temporal_features)
                - spatial: (max_aircrafts, history_length, n_spatial_features)
                - targets: (max_aircrafts, future_length, n_target_features)
                - distance_matrix: (max_aircrafts, max_aircrafts)
                - mask: (max_aircrafts,) 标记哪些位置是有效数据
        """
        scene_path = self.scene_dirs[idx]

        # 读取数据
        ego_df = pd.read_csv(os.path.join(scene_path, "ego.csv"))
        neighbors_df = pd.read_csv(os.path.join(scene_path, "neighbors.csv"))

        # 数据预处理
        ego_df = self._transform_data(ego_df)
        neighbors_df = self._transform_data(neighbors_df)

        # 提取时间维度参数
        history_length = self.data_config['history_length']
        future_length = self.data_config['prediction_length']
        total_length = history_length + future_length

        # 提取ego数据
        ego_temporal = ego_df[self.temporal_features].values  # (240, n_temporal)
        ego_spatial = ego_df[self.spatial_features].values    # (240, n_spatial)
        ego_targets = ego_df[self.target_features].values    # (240, n_target)

        # 分离历史和未来
        ego_temporal_history = ego_temporal[:history_length]
        ego_spatial_history = ego_spatial[:history_length]
        ego_targets_future = ego_targets[history_length:]

        # 提取邻居数据
        neighbor_data = []
        for neighbor_id, neighbor_group in neighbors_df.groupby('target_address'):
            if len(neighbor_group) == total_length:  # 确保是完整的240点轨迹
                neighbor_temporal = neighbor_group[self.temporal_features].values
                neighbor_spatial = neighbor_group[self.spatial_features].values
                neighbor_targets = neighbor_group[self.target_features].values

                neighbor_temporal_history = neighbor_temporal[:history_length]
                neighbor_spatial_history = neighbor_spatial[:history_length]
                neighbor_targets_future = neighbor_targets[history_length:]

                neighbor_data.append({
                    'temporal': neighbor_temporal_history,
                    'spatial': neighbor_spatial_history,
                    'targets': neighbor_targets_future
                })

        # 限制邻居数量
        if len(neighbor_data) > self.max_neighbors:
            # 随机选择max_neighbors个邻居
            indices = np.random.choice(len(neighbor_data), self.max_neighbors, replace=False)
            neighbor_data = [neighbor_data[i] for i in indices]

        # 构建批次数据
        n_aircrafts = len(neighbor_data) + 1  # +1 for ego
        max_aircrafts = min(n_aircrafts, self.max_neighbors + 1)

        # 初始化张量
        n_temporal = len(self.temporal_features)
        n_spatial = len(self.spatial_features)
        n_target = len(self.target_features)

        temporal_tensor = torch.zeros(max_aircrafts, history_length, n_temporal)
        spatial_tensor = torch.zeros(max_aircrafts, history_length, n_spatial)
        targets_tensor = torch.zeros(max_aircrafts, future_length, n_target)
        mask = torch.zeros(max_aircrafts, dtype=torch.bool)

        # 填充ego数据（第一个位置）
        temporal_tensor[0] = torch.FloatTensor(ego_temporal_history)
        spatial_tensor[0] = torch.FloatTensor(ego_spatial_history)
        targets_tensor[0] = torch.FloatTensor(ego_targets_future)
        mask[0] = True

        # 填充邻居数据
        for i, neighbor in enumerate(neighbor_data[:self.max_neighbors]):
            temporal_tensor[i+1] = torch.FloatTensor(neighbor['temporal'])
            spatial_tensor[i+1] = torch.FloatTensor(neighbor['spatial'])
            targets_tensor[i+1] = torch.FloatTensor(neighbor['targets'])
            mask[i+1] = True

        # 计算距离矩阵
        distance_matrix = self._calculate_distance_matrix(
            ego_spatial_history,
            spatial_tensor[1:][mask[1:]].numpy()
        )

        # 填充到max_aircrafts大小
        full_distance_matrix = torch.zeros(max_aircrafts, max_aircrafts)
        actual_size = distance_matrix.shape[0]
        full_distance_matrix[:actual_size, :actual_size] = torch.FloatTensor(distance_matrix)

        return {
            'temporal': temporal_tensor,
            'spatial': spatial_tensor,
            'targets': targets_tensor,
            'distance_matrix': full_distance_matrix,
            'mask': mask,
            'scene_id': os.path.basename(scene_path)
        }


def create_data_loaders(config_path: str, scenes_dir: str, batch_size: int = 8,
                          max_neighbors: int = 50, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建场景数据加载器

    Args:
        config_path: 配置文件路径
        scenes_dir: 场景根目录
        batch_size: 批大小
        max_neighbors: 每个场景最大邻居数量
        num_workers: 数据加载器工作进程数

    Returns:
        训练、验证和测试数据加载器
    """
    # 假设数据已经按8:1:1分割
    all_scenes = [os.path.join(scenes_dir, d) for d in os.listdir(scenes_dir)
                  if os.path.isdir(os.path.join(scenes_dir, d))]

    np.random.shuffle(all_scenes)
    n_total = len(all_scenes)

    train_end = int(0.8 * n_total)
    val_end = int(0.9 * n_total)

    train_scenes = all_scenes[:train_end]
    val_scenes = all_scenes[train_end:val_end]
    test_scenes = all_scenes[val_end:]

    print(f"训练场景: {len(train_scenes)}")
    print(f"验证场景: {len(val_scenes)}")
    print(f"测试场景: {len(test_scenes)}")

    # 创建数据集
    train_dataset = SceneDataset(train_scenes, config_path, max_neighbors)
    val_dataset = SceneDataset(val_scenes, config_path, max_neighbors)
    test_dataset = SceneDataset(test_scenes, config_path, max_neighbors)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试V7数据集加载器
    config_path = "../../config/social_patchtst_config.yaml"
    scenes_dir = "/mnt/d/model/adsb_scenes_v7/scenes"

    try:
        dataset = V7SocialDataset(scenes_dir, config_path)
        print(f"数据集大小: {len(dataset)}")

        # 测试一个样本
        sample = dataset[0]
        print(f"时序数据形状: {sample['temporal'].shape}")
        print(f"空间数据形状: {sample['spatial'].shape}")
        print(f"目标数据形状: {sample['targets'].shape}")
        print(f"距离矩阵形状: {sample['distance_matrix'].shape}")
        print(f"掩码形状: {sample['mask'].shape}")
        print(f"场景ID: {sample['scene_id']}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()