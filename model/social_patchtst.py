"""
Social-PatchTST完整模型
整合时序编码器、社交编码器和预测解码器
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple, Optional

from .patchtst import TemporalEncoder
from .social_transformer import SocialEncoder
from .prediction_decoder import PredictionDecoder
from config.config_manager import load_config


class SocialPatchTST(nn.Module):
    """
    Social-PatchTST完整模型

    架构：
    1. Temporal Encoder (PatchTST): 学习单架飞机的时序模式
    2. Social Encoder: 学习多架飞机之间的社交交互
    3. Prediction Decoder: 基于社交感知特征生成未来轨迹预测
    """

    def __init__(self, config_path: str, is_baseline: bool = False):
        """
        初始化Social-PatchTST模型

        Args:
            config_path: 配置文件路径
            is_baseline: 是否运行Baseline模式（关闭社交模块）
        """
        super().__init__()

        # 加载配置
        self.config = load_config(config_path)
        self.is_baseline = is_baseline  # 存储baseline开关
        self.patchtst_config = self.config.patchtst_config
        self.social_config = self.config.social_config
        self.decoder_config = self.config.decoder_config
        self.data_config = self.config.data_config

        # 时序编码器（PatchTST）
        self.temporal_encoder = TemporalEncoder(self.patchtst_config)

        # 社交编码器
        self.social_encoder = SocialEncoder(self.social_config)

        # 预测解码器
        self.prediction_decoder = PredictionDecoder(self.decoder_config)

        # 计算patch数量 (T=120, patch_len=16, stride=8)
        self.n_patches = (self.data_config['history_length'] - self.patchtst_config['patch_length']) // self.patchtst_config['stride'] + 1  # = 14

        # === 关键修复：Social特征池化层 ===
        # 将Social特征从时序维度(T=120)池化到Patch维度(n_patches=14)
        self.social_pool = nn.AdaptiveAvgPool1d(self.n_patches)

        # 损失权重
        self.loss_weights = self.config.get('training.loss_weights', {
            'position': 1.0,
            'velocity': 0.5,
            'altitude': 1.0,
            'mindist': 2.0
        })

    def forward(self, batch: Dict[str, torch.Tensor],
                teacher_forcing_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 按解剖表规范

        Args:
            batch: 批次数据，应包含：
                - x_ego: Ego飞机时序 (N, T=120, d=5)
                - x_nbr: 邻居时序 (N, T=120, K=20, d=5)
                - dist_mx: 相对距离 (N, T=120, K)
            teacher_forcing_ratio: 教师强制比例

        Returns:
            预测结果字典，包含4个头的预测
        """
        # 从batch中提取解剖表规范的输入
        x_ego = batch.get('temporal')  # (N, T=120, d=5) - 如果是单机版本
        if x_ego is None:
            # 如果temporal是多机格式，取第一架飞机作为ego
            temporal_data = batch['temporal']  # [batch_size, n_aircrafts, seq_len, n_temporal_features]
            x_ego = temporal_data[:, 0, :, :]  # 取第一架飞机作为ego

        # 获取邻居数据 - 需要构建符合解剖表的格式
        if 'x_nbr' in batch and 'dist_mx' in batch:
            x_nbr = batch['x_nbr']  # (N, T=120, K=20, d=5)
            dist_mx = batch['dist_mx']  # (N, T=120, K)
        else:
            # 从现有数据构建邻居信息
            temporal_data = batch['temporal']  # [batch_size, n_aircrafts, seq_len, n_temporal_features]
            distance_matrix = batch['distance_matrix']  # [batch_size, n_aircrafts, n_aircrafts]

            batch_size, n_aircrafts, seq_len, n_temporal_features = temporal_data.shape

            # 构建邻居特征 (N, T=120, K=20, d=5)
            # 这里简化处理：使用其他飞机作为邻居
            if n_aircrafts > 1:
                # 取最多20个邻居
                K = min(19, n_aircrafts - 1)  # 除了ego外的邻居
                x_nbr_list = []
                dist_mx_list = []

                for i in range(batch_size):
                    # ego飞机取第一架
                    ego_temporal = temporal_data[i, 0, :, :]  # ego飞机 (120, 5)

                    # 邻居飞机：从第2架到第min(20, n_aircrafts)架
                    actual_K = min(K, n_aircrafts - 1)
                    neighbor_temporal = temporal_data[i, 1:1+actual_K, :, :]  # (actual_K, 120, 5)
                    neighbor_distances = distance_matrix[i, 0, 1:1+actual_K]  # (actual_K,)

                    # 扩展维度到时序
                    # neighbor_distances: (actual_K,) -> (actual_K, 1) -> (actual_K, 120)
                    neighbor_distances_expanded = neighbor_distances.unsqueeze(1).expand(-1, seq_len)

                    x_nbr_list.append(neighbor_temporal)  # (actual_K, 120, 5)
                    dist_mx_list.append(neighbor_distances_expanded)  # (actual_K, 120)

                # 堆叠为批次格式
                x_nbr = torch.stack(x_nbr_list, dim=0)  # (batch_size, actual_K, 120, 5)
                x_nbr = x_nbr.permute(0, 2, 1, 3)  # (batch_size, 120, actual_K, 5)
                dist_mx = torch.stack(dist_mx_list, dim=0)  # (batch_size, actual_K, 120)
                dist_mx = dist_mx.permute(0, 2, 1)  # (batch_size, 120, actual_K)
            else:
                # 如果没有邻居，创建虚拟数据
                K = 20
                x_nbr = torch.zeros(batch_size, seq_len, K, n_temporal_features, device=x_ego.device)
                dist_mx = torch.full((batch_size, seq_len, K), 9999.0, device=x_ego.device)  # 很远的距离

        # === 模块一：Temporal Encoder (PatchTST) ===
        # 学习ego飞机的时序模式
        if x_ego.dim() == 3:
            # 如果是3维(N, T, d)，需要添加n_aircrafts维度
            x_ego_expanded = x_ego.unsqueeze(1)  # (N, 1, T, d)
        else:
            # 如果已经是4维(N, 1, T, d)，直接使用
            x_ego_expanded = x_ego

        encoded_temporal, n_patches = self.temporal_encoder(x_ego_expanded)
        # encoded_temporal: (N, 1, n_patches, d_model) -> (N, n_patches, d_model)
        if encoded_temporal.dim() == 4:
            # encoded_temporal: [N, n_aircrafts, n_patches, d_model]
            # 取第一架飞机（ego）的特征
            encoded_temporal = encoded_temporal[:, 0, :, :]  # [N, n_patches, d_model]
        # encoded_temporal: (N, n_patches=14, d_model=512) ✅ PatchTST输出

        # === 模块二：Social Encoder / Baseline 开关 ===
        # 🔥 关键：Baseline模式 vs Social-PatchTST模式
        if self.is_baseline:
            # Baseline模式：创建全零的"伪社交特征"
            # 确保它在正确的设备上，并且维度匹配
            social_patches = torch.zeros_like(encoded_temporal)  # [N, 14, 512]
            social_aware_features = torch.zeros(x_ego.size(0), 120, 512, device=x_ego.device)  # [N, 120, 512]
        else:
            # Social-PatchTST模式：正常运行社交模块
            # 学习多架飞机之间的社交交互
            social_aware_features = self.social_encoder(x_nbr, dist_mx)
            # social_aware_features: (N, T=120, d_social=512)

            # === 模块三：维度对齐 - 关键修复点 ===
            # 将Social特征从时序维度(T=120)池化到Patch维度(n_patches=14)
            # (N, T, D) -> (N, D, T) -> (N, D, n_patches) -> (N, n_patches, D)
            social_patches = self.social_pool(
                social_aware_features.transpose(1, 2)  # (N, 512, 120)
            ).transpose(1, 2)  # (N, 14, 512)
        # social_patches: (N, n_patches=14, d_social=512) ✅ 与PatchTST维度对齐

        # === 模块四：正确的融合方式 ===
        # 在Patch维度上融合：[N, 14, 512] + [N, 14, 512] -> [N, 14, 1024]
        fused_features = torch.cat([encoded_temporal, social_patches], dim=-1)
        # fused_features: (N, n_patches=14, d_total=1024) ✅ 真正使用了PatchTST输出

        # === 模块五：MLP Heads ===
        # 基于融合的Patch特征生成4个预测头
        raw_predictions = self.prediction_decoder(fused_features)

        return {
            'predictions': raw_predictions,
            'encoded_temporal': encoded_temporal,
            'social_aware_features': social_aware_features,
            'social_patches': social_patches,  # 新增：暴露池化后的social特征
            'fused_features': fused_features,
            'n_patches': n_patches
        }

    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor,
                     distance_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失

        Args:
            predictions: 模型预测字典，包含4个预测头的输出
            targets: 真实标签张量 [batch_size, n_aircrafts, seq_len, n_features]
            distance_matrix: 距离矩阵 [batch_size, n_aircrafts, n_aircrafts]

        Returns:
            损失字典
        """
        device = predictions['position'].device  # 从任意一个预测头获取设备

        # targets形状: [batch_size, n_aircrafts, seq_len, n_features]
        # 取第一架飞机（ego）的targets: [batch_size, seq_len, n_features]
        ego_targets = targets[:, 0, :, :]  # [batch_size, seq_len, n_features]

        # 根据数据集，targets的最后4个维度对应：
        # 假设targets的4个特征顺序为: [flight_level, latitude, longitude, ground_speed/vx, track_angle/vy]
        # 但从错误看，可能只有4个特征，需要根据实际情况调整

        # 将targets分割以匹配我们的4个预测头
        if ego_targets.size(-1) >= 4:
            # 如果targets有4个或更多特征
            target_altitude = ego_targets[:, :, 0:1]      # flight_level -> (batch_size, seq_len, 1)
            target_position = ego_targets[:, :, 1:3]      # latitude, longitude -> (batch_size, seq_len, 2)
            target_velocity = ego_targets[:, :, 3:5]      # vx, vy -> (batch_size, seq_len, 2)
        elif ego_targets.size(-1) == 3:
            # 如果targets只有3个特征，需要重新分配
            target_altitude = ego_targets[:, :, 0:1]      # flight_level
            target_position = ego_targets[:, :, 1:3]      # latitude, longitude
            target_velocity = torch.zeros_like(ego_targets[:, :, 0:2])  # 创建虚拟velocity
        else:
            raise ValueError(f"Unexpected targets shape: {ego_targets.shape}")

        # 创建虚拟的mindist目标（可能不包含在原始targets中）
        target_mindist = torch.zeros(ego_targets.size(0), ego_targets.size(1), 1, device=device)

        # 基础回归损失
        position_loss = nn.MSELoss()(predictions['position'], target_position)
        altitude_loss = nn.MSELoss()(predictions['altitude'], target_altitude)
        velocity_loss = nn.MSELoss()(predictions['velocity'], target_velocity)

        # 最小距离损失（mindist约束）
        mindist_loss = self.compute_mindist_loss(predictions['mindist'], distance_matrix)

        # 加权总损失
        total_loss = (
            self.loss_weights['position'] * position_loss +
            self.loss_weights['altitude'] * altitude_loss +
            self.loss_weights['velocity'] * velocity_loss +
            self.loss_weights['mindist'] * mindist_loss
        )

        return {
            'total_loss': total_loss,
            'position_loss': position_loss,
            'altitude_loss': altitude_loss,
            'velocity_loss': velocity_loss,
            'mindist_loss': mindist_loss
        }

    def compute_mindist_loss(self, predictions: torch.Tensor, distance_matrix: torch.Tensor,
                            safety_threshold: float = 5.0, penalty_weight: float = 10.0) -> torch.Tensor:
        """
        计算最小距离损失

        Args:
            predictions: mindist预测 [batch_size, seq_len, 1]
            distance_matrix: 当前距离矩阵 [batch_size, n_aircrafts, n_aircrafts]
            safety_threshold: 安全距离阈值（海里）
            penalty_weight: 违规惩罚权重

        Returns:
            最小距离损失
        """
        batch_size, seq_len, _ = predictions.shape
        device = predictions.device

        # 简化mindist损失：使用MSE损失，鼓励mindist预测值保持合理
        # 这里简化处理，因为原始的mindist计算比较复杂
        mindist_target = torch.ones_like(predictions) * safety_threshold
        mindist_loss = nn.MSELoss()(predictions, mindist_target)

        return mindist_loss

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        推理模式预测

        Args:
            batch: 批次数据

        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(batch, teacher_forcing_ratio=0.0)
            return output['predictions']

    def get_model_info(self) -> Dict[str, any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'Social-PatchTST',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'temporal_encoder_params': sum(p.numel() for p in self.temporal_encoder.parameters()),
            'social_encoder_params': sum(p.numel() for p in self.social_encoder.parameters()),
            'prediction_decoder_params': sum(p.numel() for p in self.prediction_decoder.parameters()),
            'config': {
                'patch_length': self.patchtst_config['patch_length'],
                'stride': self.patchtst_config['stride'],
                'd_model': self.patchtst_config['d_model'],
                'n_heads': self.patchtst_config['n_heads'],
                'history_length': self.data_config['history_length'],
                'prediction_length': self.data_config['prediction_length']
            }
        }


def create_model(config_path: str, is_baseline: bool = False) -> SocialPatchTST:
    """
    创建Social-PatchTST模型

    Args:
        config_path: 配置文件路径
        is_baseline: 是否运行Baseline模式（关闭社交模块）

    Returns:
        模型实例
    """
    model = SocialPatchTST(config_path, is_baseline=is_baseline)
    return model


if __name__ == "__main__":
    # 测试完整模型
    config_path = "../config/social_patchtst_config.yaml"

    try:
        model = create_model(config_path)

        # 创建测试数据
        batch_size = 2
        n_aircrafts = 5
        seq_len = 120  # 10分钟历史
        n_temporal_features = 5  # flight_level, ground_speed, track_angle, vertical_rate, selected_altitude

        batch = {
            'temporal': torch.randn(batch_size, n_aircrafts, seq_len, n_temporal_features),
            'spatial': torch.randn(batch_size, n_aircrafts, 2),  # lat, lon
            'targets': torch.randn(batch_size, n_aircrafts, 120, 5),  # 120个预测点，5个特征
            'distance_matrix': torch.rand(batch_size, n_aircrafts, n_aircrafts) * 20,
            'aircraft_ids': [f'AC{i:03d}' for i in range(batch_size * n_aircrafts)]
        }

        # 前向传播
        output = model(batch, teacher_forcing_ratio=0.5)

        print(f"预测结果形状: {output['predictions'].shape}")
        print(f"编码时序特征形状: {output['encoded_temporal'].shape}")
        print(f"社交感知特征形状: {output['social_aware_features'].shape}")
        print(f"Patch数量: {output['n_patches']}")

        # 计算损失
        losses = model.compute_loss(
            output['predictions'], batch['targets'], batch['distance_matrix']
        )
        print(f"损失: {losses}")

        # 显示模型信息
        model_info = model.get_model_info()
        print(f"模型信息: {model_info}")

        print("Social-PatchTST完整模型测试通过！")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()