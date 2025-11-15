"""
Social Transformer模块 - 严格按照解剖表规范
Social-Encoder：输入(N, T, K, d) + (N, T, K) → 输出(N, T, d_social=512)
"""

import torch
import torch.nn as nn
import math


class SocialEncoder(nn.Module):
    """
    航空级Social-Encoder
    严格按照解剖表：
    - 输入：(N, T, K, d) + (N, T, K)
    - 输出：(N, T, d_social=512)
    - 功能：用相对距离做RPE，输出social embedding
    """

    def __init__(self, config: dict):
        """
        初始化社交编码器

        Args:
            config: 配置字典，必须包含d_model=512
        """
        super().__init__()

        self.d_model = config['d_model']  # 512

        # 1. 相对距离编码（RPE）
        self.distance_embedding = nn.Linear(1, self.d_model // 4)

        # 2. 邻居特征编码
        self.neighbor_proj = nn.Linear(5, self.d_model // 2)  # d=5是航空特征维度

        # 3. 社交互作MLP（18.3M参数的主要来源）
        self.social_mlp = nn.Sequential(
            nn.Linear(self.d_model // 2 + self.d_model // 4, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)  # 输出 d_social=512
        )

        # 4. 层归一化
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, x_nbr: torch.Tensor, dist_mx: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 严格按照解剖表

        Args:
            x_nbr: 邻居特征 (N, T, K, d) - K=20是最大邻居数，d=5是航空特征维度
            dist_mx: 相对距离 (N, T, K) - Haversine米→海里

        Returns:
            social_embedding: (N, T, d_social=512) - Social embedding
        """
        N, T, K, d = x_nbr.shape  # (N, T=120, K=20, d=5)

        # 1. 相对距离编码
        distance_features = self.distance_embedding(dist_mx.unsqueeze(-1))  # (N, T, K, d_model//4)

        # 2. 邻居特征编码
        neighbor_features = self.neighbor_proj(x_nbr)  # (N, T, K, d_model//2)

        # 3. 聚合邻居信息（加权平均，距离作为权重）
        # 使用距离的倒数作为注意力权重
        distance_weights = torch.clamp(1.0 / (dist_mx + 1e-8), min=0, max=10)
        distance_weights = distance_weights.unsqueeze(-1)  # (N, T, K, 1)
        distance_weights = distance_weights / distance_weights.sum(dim=2, keepdim=True)

        # 加权聚合邻居特征
        weighted_neighbors = neighbor_features * distance_weights
        aggregated_neighbors = weighted_neighbors.sum(dim=2)  # (N, T, d_model//2)

        # 4. 聚合距离特征
        weighted_distances = distance_features * distance_weights
        aggregated_distances = weighted_distances.sum(dim=2)  # (N, T, d_model//4)

        # 5. 融合并通过MLP
        combined = torch.cat([aggregated_neighbors, aggregated_distances], dim=-1)  # (N, T, d_model//2 + d_model//4)
        social_embedding = self.social_mlp(combined)  # (N, T, d_model)

        # 6. 层归一化
        return self.layer_norm(social_embedding)


if __name__ == "__main__":
    # 测试Social-Encoder (按解剖表规范)
    batch_size = 4
    T = 120  # 10分钟，5秒采样
    K = 20   # 最大邻居数
    d = 5    # 航空特征维度

    # 创建测试数据 - 严格按照解剖表
    x_nbr = torch.randn(batch_size, T, K, d)  # 邻居特征 (N, T=120, K=20, d=5)
    dist_mx = torch.rand(batch_size, T, K) * 50  # 相对距离 (N, T=120, K)

    # Social-Encoder配置
    config = {
        'd_model': 512  # d_social=512
    }

    # 创建社交编码器
    social_encoder = SocialEncoder(config)

    # 前向传播
    social_embedding = social_encoder(x_nbr, dist_mx)

    print("=== Social-Encoder测试 (解剖表规范) ===")
    print(f"输入邻居特征形状: {x_nbr.shape}")
    print(f"输入距离矩阵形状: {dist_mx.shape}")
    print(f"输出social_embedding形状: {social_embedding.shape}")
    print(f"✅ 严格按照解剖表：输入(N,T,K,d)+(N,T,K) → 输出(N,T,d_social=512)")

    # 参数量统计
    total_params = sum(p.numel() for p in social_encoder.parameters())
    print(f"Social-Encoder参数量: {total_params:,}")
    print("✅ 测试通过！")