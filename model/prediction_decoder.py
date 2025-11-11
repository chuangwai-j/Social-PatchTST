"""
预测解码器模块
基于社交感知的时序特征生成未来轨迹预测
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PredictionDecoder(nn.Module):
    """
    预测解码器
    基于编码后的特征生成未来轨迹预测
    """

    def __init__(self, config: dict):
        """
        初始化预测解码器

        Args:
            config: 解码器配置
        """
        super().__init__()

        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.n_layers = config['n_layers']
        self.d_ff = config['d_ff']
        self.dropout = config['dropout']

        # 输出特征维度
        # 预测：flight_level, latitude, longitude, ground_speed, track_angle
        self.output_dim = 5

        # 解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.n_layers
        )

        # 输出投影层
        self.output_projection = nn.Linear(self.d_model, self.output_dim)

        # 位置编码（用于预测序列）
        self.prediction_position_embedding = nn.Parameter(
            torch.randn(1000, self.d_model)  # 假设最多1000个预测步
        )

    def generate_prediction_sequence(self, batch_size: int, prediction_length: int) -> torch.Tensor:
        """
        生成预测序列的位置编码

        Args:
            batch_size: 批大小
            prediction_length: 预测序列长度

        Returns:
            位置编码 [batch_size, prediction_length, d_model]
        """
        if prediction_length <= self.prediction_position_embedding.size(0):
            pos_embed = self.prediction_position_embedding[:prediction_length].unsqueeze(0)
        else:
            # 如果预测长度超过预定义的位置编码，则扩展
            pos_embed = self.prediction_position_embedding.unsqueeze(0).repeat(
                1, math.ceil(prediction_length / self.prediction_position_embedding.size(0)), 1
            )[:, :prediction_length, :]

        pos_embed = pos_embed.expand(batch_size, -1, -1)
        return pos_embed

    def forward(self, encoded_features: torch.Tensor, target_sequence: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        前向传播

        Args:
            encoded_features: 编码器输出 [batch_size, n_aircrafts, n_patches, d_model]
            target_sequence: 目标序列（训练时使用）[batch_size, n_aircrafts, prediction_length, output_dim]
            teacher_forcing_ratio: 教师强制比例

        Returns:
            预测序列 [batch_size, n_aircrafts, prediction_length, output_dim]
        """
        batch_size, n_aircrafts, n_patches, d_model = encoded_features.shape
        prediction_length = 120  # 预测未来10分钟（120个5秒点）

        # 重塑编码器输出作为解码器的记忆
        memory = encoded_features.view(batch_size * n_aircrafts, n_patches, d_model)

        # 生成预测序列的位置编码
        if target_sequence is not None:
            # 训练模式：使用目标序列
            target_length = target_sequence.size(2)
            target_embed = self.generate_prediction_sequence(batch_size * n_aircrafts, target_length)
        else:
            # 推理模式：自回归生成
            target_length = prediction_length
            target_embed = self.generate_prediction_sequence(batch_size * n_aircrafts, target_length)

        # 解码器输入（训练时使用目标序列，推理时使用预测序列）
        if target_sequence is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # 教师强制：使用真实的目标序列
            # 将目标序列投影到d_model维度
            decoder_input = torch.zeros(batch_size * n_aircrafts, target_length, d_model)
            if target_sequence.size(3) == self.output_dim:
                # 如果目标序列维度匹配，直接投影
                decoder_input = decoder_input + target_sequence.permute(0, 1, 3, 2).contiguous().view(
                    batch_size * n_aircrafts, target_length, self.output_dim
                )
                # 扩展到d_model维度
                decoder_input = decoder_input.repeat(1, 1, d_model // self.output_dim + 1)[:, :, :d_model]
        else:
            # 使用位置编码作为解码器输入
            decoder_input = target_embed

        # Transformer解码
        decoded_output = self.transformer_decoder(
            tgt=decoder_input,  # [batch_size*n_aircrafts, target_length, d_model]
            memory=memory,      # [batch_size*n_aircrafts, n_patches, d_model]
        )

        # 输出投影
        predictions = self.output_projection(decoded_output)
        # predictions: [batch_size*n_aircrafts, target_length, output_dim]

        # 重塑回原始形状
        predictions = predictions.view(
            batch_size, n_aircrafts, target_length, self.output_dim
        )

        return predictions


class ReversePatching(nn.Module):
    """
    反向Patching模块
    将patch级别的预测还原为时间序列
    """

    def __init__(self, patch_length: int, stride: int, output_dim: int):
        """
        初始化反向Patching

        Args:
            patch_length: patch长度
            stride: patch步长
            output_dim: 输出特征维度
        """
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        self.output_dim = output_dim

    def forward(self, patch_predictions: torch.Tensor, original_length: int) -> torch.Tensor:
        """
        将patch预测还原为完整时间序列

        Args:
            patch_predictions: patch级别预测 [batch_size, n_aircrafts, n_patches, output_dim]
            original_length: 原始序列长度

        Returns:
            完整时间序列预测 [batch_size, n_aircrafts, prediction_length, output_dim]
        """
        batch_size, n_aircrafts, n_patches, output_dim = patch_predictions.shape

        # 计算预测序列长度
        prediction_length = (n_patches - 1) * self.stride + self.patch_length

        # 创建输出张量
        output = torch.zeros(batch_size, n_aircrafts, prediction_length, output_dim)

        # 将patch预测填入输出张量
        for i in range(n_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_length
            output[:, :, start_idx:end_idx, :] += patch_predictions[:, :, i, :]

        # 计算每个时间点的覆盖次数
        coverage = torch.zeros(batch_size, n_aircrafts, prediction_length, 1)
        for i in range(n_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_length
            coverage[:, :, start_idx:end_idx, :] += 1

        # 平均化重叠区域
        output = output / coverage.clamp(min=1)

        # 确保输出长度为期望的预测长度
        if prediction_length > 120:  # 期望预测长度
            output = output[:, :, :120, :]

        return output


class TrajectoryPostProcessor(nn.Module):
    """
    轨迹后处理器
    对预测结果进行后处理，确保物理合理性
    """

    def __init__(self):
        super().__init__()

        # 平滑层（减少预测抖动）
        self.smoothing_conv = nn.Conv1d(
            in_channels=5, out_channels=5, kernel_size=3, padding=1, groups=5
        )
        nn.init.constant_(self.smoothing_conv.weight, 1/3)
        nn.init.constant_(self.smoothing_conv.bias, 0)

    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        对预测结果进行后处理

        Args:
            predictions: 原始预测 [batch_size, n_aircrafts, seq_len, output_dim]

        Returns:
            后处理后的预测
        """
        batch_size, n_aircrafts, seq_len, output_dim = predictions.shape

        # 重塑为 [batch_size*n_aircrafts, output_dim, seq_len]
        predictions_reshaped = predictions.permute(0, 1, 3, 2).contiguous().view(
            batch_size * n_aircrafts, output_dim, seq_len
        )

        # 应用平滑
        smoothed = self.smoothing_conv(predictions_reshaped)

        # 重塑回原始形状
        smoothed = smoothed.view(batch_size, n_aircrafts, output_dim, seq_len).permute(0, 1, 3, 2)

        return smoothed


if __name__ == "__main__":
    # 测试预测解码器
    batch_size = 2
    n_aircrafts = 5
    n_patches = 13
    d_model = 512

    # 创建测试数据
    encoded_features = torch.randn(batch_size, n_aircrafts, n_patches, d_model)
    target_sequence = torch.randn(batch_size, n_aircrafts, 120, 5)  # 5个输出特征

    # 解码器配置
    config = {
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 2048,
        'dropout': 0.1
    }

    # 创建预测解码器
    decoder = PredictionDecoder(config)

    # 前向传播
    predictions = decoder(encoded_features, target_sequence, teacher_forcing_ratio=0.5)

    print(f"编码特征形状: {encoded_features.shape}")
    print(f"预测结果形状: {predictions.shape}")
    print("预测解码器测试通过！")

    # 测试反向Patching
    reverse_patching = ReversePatching(patch_length=16, stride=8, output_dim=5)
    final_predictions = reverse_patching(predictions, 120)
    print(f"最终预测形状: {final_predictions.shape}")