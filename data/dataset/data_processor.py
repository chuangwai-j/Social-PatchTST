"""
数据预处理模块
负责读取、处理和转换ADS-B数据，使其能够直接输入Social-PatchTST模型
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from config.config_manager import load_config


class ADSBDataProcessor:
    """ADS-B数据预处理器"""

    def __init__(self, config_path: str):
        """
        初始化数据处理器

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.data_config = self.config.data_config

        # 初始化特征列
        self.temporal_features = self.data_config['feature_cols']['temporal_features']
        self.spatial_features = self.data_config['feature_cols']['spatial_features']
        self.static_features = self.data_config['feature_cols']['static_features']
        self.target_features = self.data_config['feature_cols']['target_features']

        # 所有特征列（用于归一化）
        self.all_features = self.temporal_features + self.spatial_features

        # 初始化归一化器和编码器
        self.scalers = {}
        self.encoders = {}
        self.is_fitted = False

    def fit_scalers(self, df: pd.DataFrame) -> None:
        """
        在训练数据上拟合归一化器和编码器

        Args:
            df: 训练数据DataFrame
        """
        print("拟合数据预处理器...")

        # 为数值特征拟合StandardScaler
        for feature in self.all_features:
            if feature in df.columns:
                self.scalers[feature] = StandardScaler()
                values = df[feature].values.reshape(-1, 1)
                self.scalers[feature].fit(values)
                print(f"  - {feature}: mean={self.scalers[feature].mean_[0]:.3f}, std={self.scalers[feature].scale_[0]:.3f}")

        # 为分类特征拟合LabelEncoder
        categorical_features = ['aircraft_type']  # 只编码aircraft_type
        for feature in categorical_features:
            if feature in df.columns:
                self.encoders[feature] = LabelEncoder()
                values = df[feature].astype(str).values
                self.encoders[feature].fit(values)
                print(f"  - {feature}: {len(self.encoders[feature].classes_)} 个类别")

        self.is_fitted = True
        print("数据预处理器拟合完成!")

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换数据（归一化和编码）

        Args:
            df: 原始DataFrame

        Returns:
            转换后的DataFrame
        """
        if not self.is_fitted:
            raise ValueError("请先调用 fit_scalers() 拟合预处理器")

        df_transformed = df.copy()

        # 数值特征归一化
        for feature in self.all_features:
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
                    # 将未知类别设为0（第一个类别）
                    values[mask] = self.encoders[feature].classes_[0]
                df_transformed[feature] = self.encoders[feature].transform(values)

        return df_transformed

    def calculate_distance(self, lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """
        计算两点间的大圆距离（海里）

        Args:
            lat1, lon1: 第一个点的纬度和经度
            lat2, lon2: 第二个点的纬度和经度

        Returns:
            距离（海里）
        """
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

    def create_sequences(self, df: pd.DataFrame) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        创建模型输入序列

        Args:
            df: 处理后的DataFrame

        Returns:
            输入数据字典和目标张量
        """
        history_length = self.data_config['history_length']
        prediction_length = self.data_config['prediction_length']
        sequence_length = history_length + prediction_length

        # 按飞机分组
        grouped = df.groupby('target_address')

        all_inputs = []
        all_targets = []

        for aircraft_id, group in grouped:
            if len(group) < sequence_length:
                continue  # 跳过太短的轨迹

            # 确保数据按时间排序
            group = group.sort_values('timestamp')

            # 滑动窗口生成序列
            for i in range(len(group) - sequence_length + 1):
                sequence = group.iloc[i:i+sequence_length]

                # 检查时间连续性
                time_diffs = sequence['timestamp'].diff().dropna()
                if not all(time_diffs <= self.data_config['sampling_interval'] * 1.5):
                    continue  # 跳过时间不连续的序列

                # 提取特征
                temporal_data = sequence[self.temporal_features].values
                spatial_data = sequence[self.spatial_features].values
                target_data = sequence[self.target_features].values

                # 分离历史和未来
                history_temporal = temporal_data[:history_length]
                history_spatial = spatial_data[:history_length]
                target = target_data[history_length:]  # 未来数据作为目标

                # 构建输入字典
                inputs = {
                    'temporal': torch.FloatTensor(history_temporal),
                    'spatial': torch.FloatTensor(history_spatial),
                    'aircraft_id': aircraft_id,
                    'start_time': sequence.iloc[0]['timestamp']
                }

                all_inputs.append(inputs)
                all_targets.append(torch.FloatTensor(target))

        if not all_inputs:
            raise ValueError("没有生成有效的序列数据")

        return all_inputs, all_targets

    def create_multi_aircraft_batch(self, inputs_list: List[Dict], targets_list: List[torch.Tensor],
                                   max_aircrafts: int = 50) -> Dict[str, torch.Tensor]:
        """
        创建多飞机批次数据

        Args:
            inputs_list: 输入序列列表
            targets_list: 目标序列列表
            max_aircrafts: 最大飞机数量

        Returns:
            批次数据字典
        """
        # 随机选择飞机序列
        if len(inputs_list) > max_aircrafts:
            indices = np.random.choice(len(inputs_list), max_aircrafts, replace=False)
            selected_inputs = [inputs_list[i] for i in indices]
            selected_targets = [targets_list[i] for i in indices]
        else:
            selected_inputs = inputs_list
            selected_targets = targets_list

        # 堆叠数据
        batch_data = {
            'temporal': torch.stack([inp['temporal'] for inp in selected_inputs]),
            'spatial': torch.stack([inp['spatial'] for inp in selected_inputs]),
            'targets': torch.stack(selected_targets),
            'aircraft_ids': [inp['aircraft_id'] for inp in selected_inputs],
            'start_times': [inp['start_time'] for inp in selected_inputs]
        }

        # 计算飞机间距离矩阵（用于社交注意力）
        n_aircrafts = len(selected_inputs)
        if n_aircrafts > 1:
            positions = []
            for inp in selected_inputs:
                # 使用最后一个历史点的位置
                last_pos = inp['spatial'][-1]  # [lat, lon]
                positions.append(last_pos.numpy())

            positions = np.array(positions)
            distances = np.zeros((n_aircrafts, n_aircrafts))

            for i in range(n_aircrafts):
                for j in range(n_aircrafts):
                    if i != j:
                        distances[i, j] = self.calculate_distance(
                            positions[i, 0], positions[i, 1],
                            positions[j, 0], positions[j, 1]
                        )

            batch_data['distance_matrix'] = torch.FloatTensor(distances)
        else:
            batch_data['distance_matrix'] = torch.zeros(1, 1)

        return batch_data


class ADSBDataset(Dataset):
    """ADS-B数据集类"""

    def __init__(self, df: pd.DataFrame, processor: ADSBDataProcessor,
                 max_aircrafts_per_batch: int = 50):
        """
        初始化数据集

        Args:
            df: 原始数据DataFrame
            processor: 数据处理器
            max_aircrafts_per_batch: 每批次最大飞机数
        """
        self.processor = processor
        self.max_aircrafts_per_batch = max_aircrafts_per_batch

        # 处理数据
        print("预处理数据...")
        self.df_processed = processor.transform_data(df)

        # 创建序列
        print("创建序列...")
        self.inputs_list, self.targets_list = processor.create_sequences(self.df_processed)
        print(f"生成了 {len(self.inputs_list)} 个序列")

        # 分组为批次
        self.create_batches()

    def create_batches(self):
        """创建批次"""
        # 简单的顺序分组
        batch_size = self.max_aircrafts_per_batch
        self.batches = []

        for i in range(0, len(self.inputs_list), batch_size):
            end_idx = min(i + batch_size, len(self.inputs_list))
            if end_idx - i >= 2:  # 至少需要2架飞机才能有社交交互
                batch_inputs = self.inputs_list[i:end_idx]
                batch_targets = self.targets_list[i:end_idx]
                self.batches.append((batch_inputs, batch_targets))

        print(f"创建了 {len(self.batches)} 个批次")

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch_inputs, batch_targets = self.batches[idx]
        return self.processor.create_multi_aircraft_batch(
            batch_inputs, batch_targets, self.max_aircrafts_per_batch
        )


def create_data_loaders(config_path: str, batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器

    Args:
        config_path: 配置文件路径
        batch_size: 批大小
        num_workers: 数据加载器工作进程数

    Returns:
        训练、验证和测试数据加载器
    """
    config = load_config(config_path)
    data_config = config.data_config

    # 初始化数据处理器
    processor = ADSBDataProcessor(config_path)

    # 读取数据
    print("读取数据文件...")
    train_df = pd.read_csv(f"{data_config['data_dir']}/{data_config['train_file']}")
    val_df = pd.read_csv(f"{data_config['data_dir']}/{data_config['val_file']}")
    test_df = pd.read_csv(f"{data_config['data_dir']}/{data_config['test_file']}")

    print(f"训练集: {len(train_df)} 行")
    print(f"验证集: {len(val_df)} 行")
    print(f"测试集: {len(test_df)} 行")

    # 在训练数据上拟合预处理器
    processor.fit_scalers(train_df)

    # 创建数据集
    train_dataset = ADSBDataset(train_df, processor)
    val_dataset = ADSBDataset(val_df, processor)
    test_dataset = ADSBDataset(test_df, processor)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, processor


if __name__ == "__main__":
    # 测试数据处理器
    config_path = "../config/social_patchtst_config.yaml"

    try:
        train_loader, val_loader, test_loader, processor = create_data_loaders(config_path)
        print("数据加载器创建成功!")

        # 测试一个批次
        batch = next(iter(train_loader))
        print(f"批次数据键: {batch.keys()}")
        print(f"时序数据形状: {batch['temporal'].shape}")
        print(f"空间数据形状: {batch['spatial'].shape}")
        print(f"目标数据形状: {batch['targets'].shape}")
        print(f"距离矩阵形状: {batch['distance_matrix'].shape}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()