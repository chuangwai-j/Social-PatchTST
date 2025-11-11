"""
Social Transformer模块
用于学习多架飞机之间的社交交互
"""

import torch
import torch.nn as nn
from .relative_position_encoding import SocialTransformerBlock


class SocialEncoder(nn.Module):
    """
    社交编码器
    学习多架飞机之间的交互关系
    """

    def __init__(self, config: dict):
        """
        初始化社交编码器

        Args:
            config: Social Transformer配置
        """
        super().__init__()

        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.n_layers = config['n_layers']
        self.d_ff = config['d_ff']
        self.dropout = config['dropout']
        self.max_aircrafts = config.get('max_aircrafts', 50)

        # RPE配置
        self.rpe_config = config.get('rpe', {})
        self.interaction_threshold = config.get('interaction_threshold', 10.0)

        # 输入投影层（如果输入维度与d_model不同）
        self.input_projection = nn.Linear(self.d_model, self.d_model)

        # 社交Transformer层
        self.social_layers = nn.ModuleList([
            SocialTransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                rpe_config=self.rpe_config
            )
            for _ in range(self.n_layers)
        ])

        # 输出投影层
        self.output_projection = nn.Linear(self.d_model, self.d_model)

        # 层归一化
        self.layer_norm = nn.LayerNorm(self.d_model)

        # 可学习的飞机身份嵌入（用于区分不同飞机）
        self.aircraft_embedding = nn.Embedding(10000, self.d_model)  # 假设最多10000架���机

    def create_aircraft_masks(self, n_aircrafts: int, aircraft_ids: list) -> torch.Tensor:
        """
        创建飞机掩码（用于padding）

        Args:
            n_aircrafts: 实际飞机数量
            aircraft_ids: 飞机ID列表

        Returns:
            飞机掩码 [batch_size, n_aircrafts, n_aircrafts]
        """
        batch_size = len(aircraft_ids) // n_aircrafts if aircraft_ids else 1

        # 创建有效飞机掩码
        mask = torch.ones(batch_size, n_aircrafts, n_aircrafts)

        # 这里可以添加基于飞机ID的掩码逻辑
        # 例如，过滤掉无效的飞机ID

        return mask

    def forward(self, x: torch.Tensor, distance_matrix: torch.Tensor,
                aircraft_ids: list = None, temperature: float = 1.0) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 时序编码器的输出 [batch_size, n_aircrafts, n_patches, d_model]
            distance_matrix: 飞机间距离矩阵 [batch_size, n_aircrafts, n_aircrafts]
            aircraft_ids: 飞机ID列表
            temperature: 注意力温度参数

        Returns:
            社交感知后的特征 [batch_size, n_aircrafts, n_patches, d_model]
        """
        batch_size, n_aircrafts, n_patches, d_model = x.shape

        # 输入投影
        x = self.input_projection(x)

        # 重塑为 [batch_size, n_patches, n_aircrafts, d_model]
        # 这样可以将每个时间步的飞机视为一个序列
        x = x.permute(0, 2, 1, 3)  # [batch_size, n_patches, n_aircrafts, d_model]
        x = x.contiguous().view(-1, n_aircrafts, d_model)  # [batch_size * n_patches, n_aircrafts, d_model]

        # 创建交互掩码
        if self.rpe_config.get('enabled', True):
            # 扩展距离矩阵到所有patch
            distance_matrix_expanded = distance_matrix.unsqueeze(1)
            distance_matrix_expanded = distance_matrix_expanded.expand(
                -1, n_patches, -1, -1
            ).contiguous().view(-1, n_aircrafts, n_aircrafts)

            # 创建交互掩码
            from .relative_position_encoding import RelativePositionEncoding
            rpe = RelativePositionEncoding(self.d_model, **self.rpe_config)
            interaction_mask = rpe.get_interaction_mask(
                distance_matrix_expanded, self.interaction_threshold
            )
        else:
            interaction_mask = None

        # 通过社交Transformer层
        for layer in self.social_layers:
            x = layer(
                x, distance_matrix_expanded, interaction_mask, temperature
            )

        # 输出投影
        x = self.output_projection(x)
        x = self.layer_norm(x)

        # 重塑回原始形状
        x = x.view(batch_size, n_patches, n_aircrafts, self.d_model)
        x = x.permute(0, 2, 1, 3)  # [batch_size, n_aircrafts, n_patches, d_model]

        return x


class MindistAwareAttention(nn.Module):
    """
    最小距离感知注意力模块
    专门用于处理mindist约束
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # 安全距离嵌入（学习不同安全等级的重要性）
        self.safety_distance_embedding = nn.Embedding(10, d_model)  # 10个安全等级

    def forward(self, x: torch.Tensor, distance_matrix: torch.Tensor,
                safety_threshold: float = 5.0) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, n_aircrafts, d_model]
            distance_matrix: 距离矩阵 [batch_size, n_aircrafts, n_aircrafts]
            safety_threshold: 安全距离阈值（海里）

        Returns:
            安全感知的输出特征
        """
        batch_size, n_aircrafts, d_model = x.shape

        # 计算Q, K, V
        Q = self.q_linear(x).view(batch_size, n_aircrafts, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, n_aircrafts, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, n_aircrafts, self.n_heads, self.d_k).transpose(1, 2)

        # 基础注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 安全距离偏置
        safety_levels = torch.clamp(
            (distance_matrix / safety_threshold).long(), 0, 9
        )  # [batch_size, n_aircrafts, n_aircrafts]

        safety_embeddings = self.safety_distance_embedding(safety_levels)
        # safety_embeddings: [batch_size, n_aircrafts, n_aircrafts, d_model]

        # 将安全嵌入投影到注意力分数
        safety_bias = torch.einsum('bijd,hdk->bhijk', safety_embeddings, self.out_linear.weight[:self.d_model].view(self.n_heads, self.d_k, self.d_model))
        safety_bias = safety_bias.sum(-1) / math.sqrt(self.d_model)

        # 添加安全偏置
        scores = scores + safety_bias

        # 计算注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        output = torch.matmul(attention_weights, V)

        # 重塑输出
        output = output.transpose(1, 2).contiguous().view(
            batch_size, n_aircrafts, self.d_model
        )

        output = self.out_linear(output)

        return output


if __name__ == "__main__":
    # 测试Social Transformer
    batch_size = 2
    n_aircrafts = 5
    n_patches = 13  # 根据patch_length=16, stride=8, seq_len=120计算得出
    d_model = 512

    # 创建测试数据
    x = torch.randn(batch_size, n_aircrafts, n_patches, d_model)
    distance_matrix = torch.rand(batch_size, n_aircrafts, n_aircrafts) * 20  # 0-20海里

    # Social Transformer配置
    config = {
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 2048,
        'dropout': 0.1,
        'max_aircrafts': 50,
        'rpe': {
            'enabled': True,
            'max_distance': 100.0,
            'distance_bins': 20,
            'temperature': 1.0
        },
        'interaction_threshold': 10.0
    }

    # 创建社交编码器
    social_encoder = SocialEncoder(config)

    # 前向传播
    social_output = social_encoder(x, distance_matrix)

    print(f"输入形状: {x.shape}")
    print(f"距离矩阵形状: {distance_matrix.shape}")
    print(f"输出形状: {social_output.shape}")
    print("Social Transformer测试通过！")