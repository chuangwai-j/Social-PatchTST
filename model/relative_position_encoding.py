"""
相对位置编码（RPE）模块
用于在Social Transformer中编码飞机间的物理距离信息
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class RelativePositionEncoding(nn.Module):
    """
    相对位置编码模块
    将飞机间的物理距离编码为注意力偏置
    """

    def __init__(self, d_model: int, max_distance: float = 100.0, distance_bins: int = 20):
        """
        初始化相对位置编码

        Args:
            d_model: 模型维度
            max_distance: 最大考虑距离（海里）
            distance_bins: 距离分箱数
        """
        super().__init__()
        self.d_model = d_model
        self.max_distance = max_distance
        self.distance_bins = distance_bins
        self.bin_size = max_distance / distance_bins

        # 距离嵌入层
        self.distance_embedding = nn.Embedding(distance_bins + 1, d_model)

        # 可学习的距离权重
        self.distance_weights = nn.Parameter(torch.ones(distance_bins + 1))

        # 注册距离缓冲区
        self.register_buffer('distance_bins_tensor', torch.arange(distance_bins + 1))

    def forward(self, distance_matrix: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        计算相对位置编码

        Args:
            distance_matrix: 飞机间距离矩阵 [batch_size, n_aircrafts, n_aircrafts]
            temperature: 注意力温度参数

        Returns:
            相对位置编码偏置 [batch_size, n_heads, n_aircrafts, n_aircrafts]
        """
        batch_size, n_aircrafts, _ = distance_matrix.shape

        # 将连续距离分箱
        distance_binned = torch.clamp(
            (distance_matrix / self.bin_size).long(),
            0, self.distance_bins
        )  # [batch_size, n_aircrafts, n_aircrafts]

        # 获取距离嵌入
        distance_embeddings = self.distance_embedding(distance_binned)  # [batch_size, n_aircrafts, n_aircrafts, d_model]

        # 应用距离权重（温度缩放）
        distance_weights = self.distance_weights[distance_binned] / temperature  # [batch_size, n_aircrafts, n_aircrafts]

        # 计算点积得到注意力偏置
        # 这里我们简化为直接使用距离权重作为偏置
        attention_bias = distance_weights.unsqueeze(1)  # [batch_size, 1, n_aircrafts, n_aircrafts]

        return attention_bias

    def get_interaction_mask(self, distance_matrix: torch.Tensor, threshold: float = 10.0) -> torch.Tensor:
        """
        生成交互掩码（只考虑距离阈值内的飞机）

        Args:
            distance_matrix: 距离矩阵
            threshold: 交互距离阈值

        Returns:
            交互掩码 [batch_size, n_aircrafts, n_aircrafts]
        """
        # 距离阈值内的飞机可以交互
        interaction_mask = (distance_matrix <= threshold).float()
        return interaction_mask


class MultiHeadAttentionWithRPE(nn.Module):
    """
    带相对位置编码的多头注意力
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 rpe_config: Optional[dict] = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # 相对位置编码
        if rpe_config is not None:
            self.rpe = RelativePositionEncoding(
                d_model=d_model,
                max_distance=rpe_config.get('max_distance', 100.0),
                distance_bins=rpe_config.get('distance_bins', 20)
            )
        else:
            self.rpe = None

    def forward(self, x: torch.Tensor, distance_matrix: torch.Tensor,
                mask: Optional[torch.Tensor] = None, temperature: float = 1.0) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, n_aircrafts, d_model]
            distance_matrix: 距离矩阵 [batch_size, n_aircrafts, n_aircrafts]
            mask: 注意力掩码 [batch_size, n_aircrafts, n_aircrafts]
            temperature: 注意力温度参数

        Returns:
            输出张量 [batch_size, n_aircrafts, d_model]
        """
        batch_size, n_aircrafts, _ = x.shape

        # 计算Q, K, V
        Q = self.q_linear(x).view(batch_size, n_aircrafts, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, n_aircrafts, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, n_aircrafts, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [batch_size, n_heads, n_aircrafts, n_aircrafts]

        # 添加相对位置编码偏置
        if self.rpe is not None:
            rpe_bias = self.rpe(distance_matrix, temperature)
            # rpe_bias: [batch_size, 1, n_aircrafts, n_aircrafts]
            scores = scores + rpe_bias

        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, n_aircrafts, n_aircrafts]
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        output = torch.matmul(attention_weights, V)
        # output: [batch_size, n_heads, n_aircrafts, d_k]

        # 重塑输出
        output = output.transpose(1, 2).contiguous().view(
            batch_size, n_aircrafts, self.d_model
        )

        # 最终线性变换
        output = self.out_linear(output)

        return output


class SocialTransformerBlock(nn.Module):
    """
    社交Transformer块
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 rpe_config: Optional[dict] = None):
        super().__init__()
        self.attention = MultiHeadAttentionWithRPE(
            d_model=d_model, n_heads=n_heads, dropout=dropout, rpe_config=rpe_config
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, distance_matrix: torch.Tensor,
                mask: Optional[torch.Tensor] = None, temperature: float = 1.0) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, n_aircrafts, d_model]
            distance_matrix: 距离矩阵 [batch_size, n_aircrafts, n_aircrafts]
            mask: 注意力掩码
            temperature: 注意力温度参数

        Returns:
            输出张量 [batch_size, n_aircrafts, d_model]
        """
        # 多头注意力 + 残差连接 + 层归一化
        attn_output = self.attention(x, distance_matrix, mask, temperature)
        x = self.norm1(x + attn_output)

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)

        return x


if __name__ == "__main__":
    # 测试相对位置编码
    batch_size = 2
    n_aircrafts = 5
    d_model = 512

    # 创建测试数据
    x = torch.randn(batch_size, n_aircrafts, d_model)
    distance_matrix = torch.rand(batch_size, n_aircrafts, n_aircrafts) * 50  # 0-50海里

    # 测试相对位置编码
    rpe_config = {
        'max_distance': 100.0,
        'distance_bins': 20
    }

    attention_layer = MultiHeadAttentionWithRPE(
        d_model=d_model, n_heads=8, dropout=0.1, rpe_config=rpe_config
    )

    output = attention_layer(x, distance_matrix)
    print(f"输入形状: {x.shape}")
    print(f"距离矩阵形状: {distance_matrix.shape}")
    print(f"输出形状: {output.shape}")
    print("相对位置编码测试通过！")