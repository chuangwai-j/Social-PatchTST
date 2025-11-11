"""
PatchTST (时序分块Transformer) 模块
用于学习单架飞机的时序模式
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class PatchTSTEmbedding(nn.Module):
    """
    PatchTST输入嵌入层
    """

    def __init__(self, patch_length: int, stride: int, d_model: int, n_features: int):
        """
        初始化PatchTST嵌入层

        Args:
            patch_length: 每个patch的长度
            stride: patch滑动步长
            d_model: 模型维度
            n_features: 特征数量
        """
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        self.d_model = d_model
        self.n_features = n_features

        # Patch嵌入层
        self.patch_embedding = nn.Linear(patch_length * n_features, d_model)

        # 位置编码
        self.position_embedding = nn.Parameter(
            torch.randn(1000, d_model)  # 假设最多1000个patch
        )

    def create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        将时间序列转换为patch

        Args:
            x: 输入时间序列 [batch_size, seq_len, n_features]

        Returns:
            patch序列 [batch_size, n_patches, patch_length * n_features]
        """
        batch_size, seq_len, n_features = x.shape

        # 计算patch数量
        n_patches = (seq_len - self.patch_length) // self.stride + 1

        # 使用unfold创建patch
        # x: [batch_size, seq_len, n_features]
        # 我们需要将特征维度和patch长度合并
        x_reshaped = x.permute(0, 2, 1)  # [batch_size, n_features, seq_len]

        # 对每个特征维度创建patch
        patches_list = []
        for i in range(n_features):
            feature_patches = x_reshaped[:, i, :].unfold(1, self.patch_length, self.stride)
            # feature_patches: [batch_size, n_patches, patch_length]
            patches_list.append(feature_patches)

        # 合并所有特征的patch
        patches = torch.stack(patches_list, dim=-1)
        # patches: [batch_size, n_patches, patch_length, n_features]

        # 重塑为 [batch_size, n_patches, patch_length * n_features]
        patches = patches.view(batch_size, n_patches, -1)

        return patches

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        前向传播

        Args:
            x: 输入时间序列 [batch_size, seq_len, n_features]

        Returns:
            embedded_patches: 嵌入后的patch [batch_size, n_patches, d_model]
            n_patches: patch数量
        """
        batch_size, seq_len, n_features = x.shape

        # 创建patch
        patches = self.create_patches(x)  # [batch_size, n_patches, patch_length * n_features]
        batch_size, n_patches, _ = patches.shape

        # Patch嵌入
        embedded_patches = self.patch_embedding(patches)  # [batch_size, n_patches, d_model]

        # 添加位置编码
        if n_patches <= self.position_embedding.size(0):
            pos_embed = self.position_embedding[:n_patches].unsqueeze(0)
        else:
            # 如果patch数量超过预定义的位置编码，则扩展
            pos_embed = self.position_embedding.unsqueeze(0).repeat(
                1, math.ceil(n_patches / self.position_embedding.size(0)), 1
            )[:, :n_patches, :]

        embedded_patches = embedded_patches + pos_embed

        return embedded_patches, n_patches


class PatchTSTEncoder(nn.Module):
    """
    PatchTST编码器
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float = 0.1):
        """
        初始化PatchTST编码器

        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: Transformer层数
            d_ff: 前馈网络维度
            dropout: Dropout率
        """
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入patch序列 [batch_size, n_patches, d_model]

        Returns:
            编码后的patch序列 [batch_size, n_patches, d_model]
        """
        output = self.transformer_encoder(x)
        return output


class TemporalEncoder(nn.Module):
    """
    时序编码器（使用PatchTST）
    为每架飞机独立学习时序模式
    """

    def __init__(self, config: dict):
        """
        初始化时序编码器

        Args:
            config: PatchTST配置
        """
        super().__init__()

        self.patch_length = config['patch_length']
        self.stride = config['stride']
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.n_layers = config['n_layers']
        self.d_ff = config['d_ff']
        self.dropout = config['dropout']

        # 从输入数据推断特征数量
        # 这里假设输入包含：flight_level, ground_speed, track_angle, vertical_rate, selected_altitude
        self.n_features = 5

        # Patch嵌入层
        self.patch_embedding = PatchTSTEmbedding(
            patch_length=self.patch_length,
            stride=self.stride,
            d_model=self.d_model,
            n_features=self.n_features
        )

        # Transformer编码器
        self.encoder = PatchTSTEncoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout
        )

        # 输出投影层（用于后续的社交编码器）
        self.output_projection = nn.Linear(self.d_model, self.d_model)

        # 层归一化
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        前向传播

        Args:
            x: 输入时序数据 [batch_size, n_aircrafts, seq_len, n_features]

        Returns:
            encoded_temporal: 编码后的时序特征 [batch_size, n_aircrafts, n_patches, d_model]
            n_patches: patch数量
        """
        batch_size, n_aircrafts, seq_len, n_features = x.shape

        # 重塑输入以处理每架飞机
        x = x.view(-1, seq_len, n_features)  # [batch_size * n_aircrafts, seq_len, n_features]

        # Patch嵌入
        embedded_patches, n_patches = self.patch_embedding(x)
        # embedded_patches: [batch_size * n_aircrafts, n_patches, d_model]

        # Transformer编码
        encoded_patches = self.encoder(embedded_patches)
        # encoded_patches: [batch_size * n_aircrafts, n_patches, d_model]

        # 输出投影
        encoded_patches = self.output_projection(encoded_patches)
        encoded_patches = self.layer_norm(encoded_patches)

        # 重塑回原始批次形状
        encoded_temporal = encoded_patches.view(
            batch_size, n_aircrafts, n_patches, self.d_model
        )

        return encoded_temporal, n_patches


if __name__ == "__main__":
    # 测试PatchTST
    batch_size = 2
    n_aircrafts = 5
    seq_len = 120  # 10分钟历史
    n_features = 5  # flight_level, ground_speed, track_angle, vertical_rate, selected_altitude

    # 创建测试数据
    x = torch.randn(batch_size, n_aircrafts, seq_len, n_features)

    # PatchTST配置
    config = {
        'patch_length': 16,
        'stride': 8,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1
    }

    # 创建时序编码器
    temporal_encoder = TemporalEncoder(config)

    # 前向传播
    encoded_temporal, n_patches = temporal_encoder(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {encoded_temporal.shape}")
    print(f"Patch数量: {n_patches}")
    print("PatchTST测试通过！")