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
from .prediction_decoder import PredictionDecoder, ReversePatching, TrajectoryPostProcessor
from config.config_manager import load_config


class SocialPatchTST(nn.Module):
    """
    Social-PatchTST完整模型

    架构：
    1. Temporal Encoder (PatchTST): 学习单架飞机的时序模式
    2. Social Encoder: 学习多架飞机之间的社交交互
    3. Prediction Decoder: 基于社交感知特征生成未来轨迹预测
    """

    def __init__(self, config_path: str):
        """
        初始化Social-PatchTST模型

        Args:
            config_path: 配置文件路径
        """
        super().__init__()

        # 加载配置
        self.config = load_config(config_path)
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

        # 反向Patching
        self.reverse_patching = ReversePatching(
            patch_length=self.patchtst_config['patch_length'],
            stride=self.patchtst_config['stride'],
            output_dim=5  # flight_level, latitude, longitude, ground_speed, track_angle
        )

        # 轨迹后处理器
        self.post_processor = TrajectoryPostProcessor()

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
                    ego_temporal = temporal_data[i, 0, :, :]  # ego飞机
                    neighbor_temporal = temporal_data[i, 1:1+K, :, :]  # 邻居飞机
                    neighbor_distances = distance_matrix[i, 0, 1:1+K]  # ego到邻居的距离

                    # 扩展维度到时序
                    neighbor_distances_expanded = neighbor_distances.unsqueeze(1).expand(-1, seq_len, -1)

                    x_nbr_list.append(neighbor_temporal)
                    dist_mx_list.append(neighbor_distances_expanded)

                x_nbr = torch.stack(x_nbr_list, dim=0)  # (batch_size, K, seq_len, d)
                x_nbr = x_nbr.permute(0, 2, 1, 3)  # (batch_size, seq_len, K, d)
                dist_mx = torch.stack(dist_mx_list, dim=0)  # (batch_size, seq_len, K)
            else:
                # 如果没有邻居，创建虚拟数据
                K = 20
                x_nbr = torch.zeros(batch_size, seq_len, K, n_temporal_features, device=x_ego.device)
                dist_mx = torch.full((batch_size, seq_len, K), 9999.0, device=x_ego.device)  # 很远的距离

        # === 模块一：Temporal Encoder (PatchTST) ===
        # 学习ego飞机的时序模式
        encoded_temporal, n_patches = self.temporal_encoder(x_ego)
        # encoded_temporal: (N, n_patches, d_model)

        # === 模块二：Social Encoder ===
        # 学习多架飞机之间的社交交互
        social_aware_features = self.social_encoder(x_nbr, dist_mx)
        # social_aware_features: (N, T, d_social=512)

        # === 模块三：融合方式（解剖表规范）===
        # (N, T, d) + (N, T, d_social) → (N, T, d + d_social)
        fused_features = torch.cat([x_ego, social_aware_features], dim=-1)
        # fused_features: (N, T=120, d + d_social=517)

        # === 模块四：MLP Heads ===
        # 基于融合特征生成4个预测头
        raw_predictions = self.prediction_decoder(fused_features)

        return {
            'predictions': raw_predictions,
            'encoded_temporal': encoded_temporal,
            'social_aware_features': social_aware_features,
            'fused_features': fused_features,
            'n_patches': n_patches
        }

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                     distance_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失

        Args:
            predictions: 模型预测 [batch_size, n_aircrafts, seq_len, output_dim]
            targets: 真实标签 [batch_size, n_aircrafts, seq_len, output_dim]
            distance_matrix: 距离矩阵 [batch_size, n_aircrafts, n_aircrafts]

        Returns:
            损失字典
        """
        device = predictions.device

        # 基础回归损失
        position_loss = nn.MSELoss()(predictions[:, :, :, :2], targets[:, :, :, :2])  # lat, lon
        altitude_loss = nn.MSELoss()(predictions[:, :, :, 0:1], targets[:, :, :, 0:1])  # flight_level
        velocity_loss = nn.MSELoss()(predictions[:, :, :, 2:4], targets[:, :, :, 2:4])  # ground_speed, track_angle

        # 最小距离损失（mindist约束）
        mindist_loss = self.compute_mindist_loss(predictions, distance_matrix)

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
            predictions: 预测轨迹 [batch_size, n_aircrafts, seq_len, output_dim]
            distance_matrix: 当前距离矩阵 [batch_size, n_aircrafts, n_aircrafts]
            safety_threshold: 安全距离阈值（海里）
            penalty_weight: 违规惩罚权重

        Returns:
            最小距离损失
        """
        batch_size, n_aircrafts, seq_len, _ = predictions.shape
        device = predictions.device

        # 只考虑位置信息 (lat, lon)
        predicted_positions = predictions[:, :, :, :2]  # [batch_size, n_aircrafts, seq_len, 2]

        mindist_violations = 0

        # 对每个时间步计算最小距离违规
        for t in range(seq_len):
            current_positions = predicted_positions[:, :, t, :]  # [batch_size, n_aircrafts, 2]

            # 计算预测距离（简化计算，实际应该使用大圆距离）
            pred_distances = torch.cdist(current_positions, current_positions)  # [batch_size, n_aircrafts, n_aircrafts]

            # 计算违规程度（小于安全阈值的距离）
            violations = torch.clamp(safety_threshold - pred_distances, min=0)
            violations = violations * (violations > 0)  # 只保留正值的违规

            # 累积违规
            mindist_violations += violations.sum()

        # 归一化损失
        mindist_loss = penalty_weight * mindist_violations / (batch_size * seq_len * n_aircrafts)

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


def create_model(config_path: str) -> SocialPatchTST:
    """
    创建Social-PatchTST模型

    Args:
        config_path: 配置文件路径

    Returns:
        模型实例
    """
    model = SocialPatchTST(config_path)
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