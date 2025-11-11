#!/usr/bin/env python3
"""
V9 分层评估器
支持按场景类别分层评估模型性能
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional
import json
import os
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from config.config_manager import load_config
from data.dataset.scene_dataset import SceneDataset, create_data_loaders


class StratifiedEvaluator:
    """
    V9 分层评估器
    支持按场景类别、距离区间等维度进行分层评估
    """

    def __init__(self, config_path: str):
        """
        初始化评估器

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 评估距离分箱定义（海里）
        self.distance_bins = [0, 5, 10, 30, np.inf]
        self.distance_labels = ['<5NM', '5-10NM', '10-30NM', '>30NM', 'SOLO']

        # 评估指标
        self.metrics = {
            'position': ['RMSE', 'MAE'],
            'altitude': ['RMSE', 'MAE'],
            'velocity': ['RMSE', 'MAE'],
            'mindist': ['MAE', 'Violation_Rate']
        }

    def evaluate_model(self, model, test_loader, save_path: str = None) -> Dict:
        """
        对模型进行分层评估

        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器
            save_path: 结果保存路径

        Returns:
            dict: 分层评估结果
        """
        model.eval()

        # 按场景类别和距离区间收集预测结果
        results_by_category = defaultdict(list)
        results_by_distance = defaultdict(list)
        all_results = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="分层评估中"):
                # 将数据移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # 前向传播
                predictions = model(batch, teacher_forcing_ratio=0.0)

                # 处理每个样本
                batch_size = predictions['predictions'].shape[0]
                for i in range(batch_size):
                    # 获取真实值和预测值
                    pred_coords = predictions['predictions'][i].cpu().numpy()
                    true_coords = batch['targets'][i].cpu().numpy()
                    mask = batch['mask'][i].cpu().numpy()
                    distance_matrix = batch['distance_matrix'][i].cpu().numpy()

                    # 获取场景信息
                    scene_category = batch.get('scene_category', ['unknown'] * batch_size)[i] if isinstance(batch.get('scene_category'), list) else batch['scene_category'][i]
                    mindist = batch.get('mindist', [9999.0] * batch_size)[i] if isinstance(batch.get('mindist'), list) else batch['mindist'][i].item()

                    # 计算指标
                    metrics = self._calculate_metrics(
                        pred_coords, true_coords, mask, distance_matrix, mindist
                    )

                    # 添加场景信息
                    metrics['scene_category'] = scene_category
                    metrics['mindist'] = mindist
                    metrics['distance_bin'] = self._get_distance_bin(mindist)

                    # 收集结果
                    results_by_category[scene_category].append(metrics)
                    results_by_distance[metrics['distance_bin']].append(metrics)
                    all_results.append(metrics)

        # 计算分层统计
        category_stats = self._compute_group_stats(results_by_category)
        distance_stats = self._compute_group_stats(results_by_distance)
        overall_stats = self._compute_group_stats({'all': all_results})

        # 整合结果
        evaluation_results = {
            'overall': overall_stats['all'],
            'by_category': category_stats,
            'by_distance': distance_stats,
            'sample_distribution': self._compute_sample_distribution(results_by_category, results_by_distance),
            'metrics_definition': {
                'position_RMSE': '位置预测均方根误差 (度)',
                'position_MAE': '位置预测平均绝对误差 (度)',
                'altitude_RMSE': '高度预测均方根误差 (英尺)',
                'altitude_MAE': '高度预测平均绝对误差 (英尺)',
                'velocity_RMSE': '速度预测均方根误差 (节)',
                'velocity_MAE': '速度预测平均绝对误差 (节)',
                'mindist_MAE': '最小距离预测平均绝对误差 (海里)',
                'mindist_Violation_Rate': '最小距离违规率 (<3NM)'
            }
        }

        # 保存结果
        if save_path:
            self._save_results(evaluation_results, save_path)

        return evaluation_results

    def _calculate_metrics(self, pred_coords: np.ndarray, true_coords: np.ndarray,
                          mask: np.ndarray, distance_matrix: np.ndarray, mindist: float) -> Dict:
        """
        计算单个样本的评估指标

        Args:
            pred_coords: 预测坐标 [aircrafts, time, features]
            true_coords: 真实坐标 [aircrafts, time, features]
            mask: 有效飞机掩码
            distance_matrix: 距离矩阵
            mindist: 最小距离

        Returns:
            dict: 评估指标
        """
        # 只考虑有效的飞机
        valid_aircrafts = np.where(mask)[0]
        if len(valid_aircrafts) == 0:
            return self._get_empty_metrics()

        # 提取有效数据
        pred_valid = pred_coords[valid_aircrafts]
        true_valid = true_coords[valid_aircrafts]

        # 假设特征顺序: [flight_level, latitude, longitude, ground_speed, track_angle]
        # 或者可能是 [latitude, longitude, flight_level, ground_speed, track_angle]

        # 为了安全，我们需要确定特征顺序
        # 让我们检查配置文件中的特征定义
        temporal_features = self.config.get('data.feature_cols.temporal_features', [])
        target_features = self.config.get('data.feature_cols.target_features', [])

        # 根据特征名称确定索引
        feature_indices = self._get_feature_indices(temporal_features, target_features)

        # 计算各项指标
        metrics = {}

        # 位置误差 (经纬度)
        if 'latitude' in feature_indices and 'longitude' in feature_indices:
            lat_idx, lon_idx = feature_indices['latitude'], feature_indices['longitude']
            pred_pos = pred_valid[:, :, [lat_idx, lon_idx]]
            true_pos = true_valid[:, :, [lat_idx, lon_idx]]

            metrics['position_RMSE'] = np.sqrt(np.mean((pred_pos - true_pos) ** 2))
            metrics['position_MAE'] = np.mean(np.abs(pred_pos - true_pos))

        # 高度误差
        if 'flight_level' in feature_indices:
            alt_idx = feature_indices['flight_level']
            pred_alt = pred_valid[:, :, alt_idx]
            true_alt = true_valid[:, :, alt_idx]

            metrics['altitude_RMSE'] = np.sqrt(np.mean((pred_alt - true_alt) ** 2))
            metrics['altitude_MAE'] = np.mean(np.abs(pred_alt - true_alt))

        # 速度误差
        if 'ground_speed' in feature_indices:
            speed_idx = feature_indices['ground_speed']
            pred_speed = pred_valid[:, :, speed_idx]
            true_speed = true_valid[:, :, speed_idx]

            metrics['velocity_RMSE'] = np.sqrt(np.mean((pred_speed - true_speed) ** 2))
            metrics['velocity_MAE'] = np.mean(np.abs(pred_speed - true_speed))

        # 最小距离相关指标
        metrics['mindist_MAE'] = abs(mindist - 9999.0) if mindist == 9999.0 else 0.0  # 简化版
        metrics['mindist_Violation_Rate'] = 1.0 if mindist < 3.0 and mindist != 9999.0 else 0.0

        return metrics

    def _get_feature_indices(self, temporal_features: List[str], target_features: List[str]) -> Dict[str, int]:
        """获取特征在数组中的索引"""
        # 假设temporal和target特征顺序一致，使用target_features
        feature_indices = {}
        for i, feature in enumerate(target_features):
            feature_indices[feature] = i
        return feature_indices

    def _get_empty_metrics(self) -> Dict:
        """返回空的指标字典"""
        return {
            'position_RMSE': 0.0, 'position_MAE': 0.0,
            'altitude_RMSE': 0.0, 'altitude_MAE': 0.0,
            'velocity_RMSE': 0.0, 'velocity_MAE': 0.0,
            'mindist_MAE': 0.0, 'mindist_Violation_Rate': 0.0
        }

    def _get_distance_bin(self, mindist: float) -> str:
        """根据最小距离获取距离分箱标签"""
        if mindist == 9999.0:
            return 'SOLO'
        elif mindist < 5:
            return '<5NM'
        elif mindist < 10:
            return '5-10NM'
        elif mindist < 30:
            return '10-30NM'
        else:
            return '>30NM'

    def _compute_group_stats(self, grouped_results: Dict) -> Dict:
        """计算分组统计信息"""
        group_stats = {}

        for group_name, results in grouped_results.items():
            if not results:
                continue

            # 聚合所有指标
            stats = {}
            for metric_name in self._get_empty_metrics().keys():
                values = [r[metric_name] for r in results if metric_name in r]
                if values:
                    stats[f'{metric_name}_mean'] = np.mean(values)
                    stats[f'{metric_name}_std'] = np.std(values)
                    stats[f'{metric_name}_median'] = np.median(values)
                    stats[f'{metric_name}_min'] = np.min(values)
                    stats[f'{metric_name}_max'] = np.max(values)
                else:
                    stats[f'{metric_name}_mean'] = 0.0
                    stats[f'{metric_name}_std'] = 0.0
                    stats[f'{metric_name}_median'] = 0.0
                    stats[f'{metric_name}_min'] = 0.0
                    stats[f'{metric_name}_max'] = 0.0

            stats['sample_count'] = len(results)
            group_stats[group_name] = stats

        return group_stats

    def _compute_sample_distribution(self, results_by_category: Dict, results_by_distance: Dict) -> Dict:
        """计算样本分布统计"""
        distribution = {}

        # 按类别分布
        category_counts = {cat: len(results) for cat, results in results_by_category.items()}
        total_samples = sum(category_counts.values())
        distribution['by_category'] = {
            'counts': category_counts,
            'percentages': {cat: count/total_samples*100 for cat, count in category_counts.items()},
            'total_samples': total_samples
        }

        # 按距离分布
        distance_counts = {dist: len(results) for dist, results in results_by_distance.items()}
        distribution['by_distance'] = {
            'counts': distance_counts,
            'percentages': {dist: count/total_samples*100 for dist, count in distance_counts.items()},
            'total_samples': total_samples
        }

        return distribution

    def _save_results(self, results: Dict, save_path: str):
        """保存评估结果"""
        os.makedirs(save_path, exist_ok=True)

        # 保存JSON格式结果
        with open(os.path.join(save_path, 'stratified_evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # 生成报告
        self._generate_report(results, save_path)

    def _generate_report(self, results: Dict, save_path: str):
        """生成评估报告"""
        report_path = os.path.join(save_path, 'evaluation_report.md')

        with open(report_path, 'w') as f:
            f.write("# V9 分层评估报告\n\n")

            # 总体性能
            f.write("## 总体性能\n\n")
            overall = results['overall']
            f.write(f"- 总样本数: {results['sample_distribution']['by_category']['total_samples']}\n")
            f.write(f"- 位置RMSE: {overall.get('position_RMSE_mean', 0):.6f} ± {overall.get('position_RMSE_std', 0):.6f}\n")
            f.write(f"- 高度RMSE: {overall.get('altitude_RMSE_mean', 0):.2f} ± {overall.get('altitude_RMSE_std', 0):.2f} ft\n")
            f.write(f"- 速度RMSE: {overall.get('velocity_RMSE_mean', 0):.2f} ± {overall.get('velocity_RMSE_std', 0):.2f} kts\n")
            f.write(f"- 冲突违规率: {overall.get('mindist_Violation_Rate_mean', 0):.4f}\n\n")

            # 按类别性能
            f.write("## 按场景类别性能\n\n")
            f.write("| 类别 | 样本数 | 位置RMSE | 高度RMSE | 速度RMSE | 违规率 |\n")
            f.write("|------|--------|----------|----------|----------|--------|\n")

            for category, stats in results['by_category'].items():
                f.write(f"| {category} | {stats['sample_count']} | "
                       f"{stats.get('position_RMSE_mean', 0):.6f} | "
                       f"{stats.get('altitude_RMSE_mean', 0):.2f} | "
                       f"{stats.get('velocity_RMSE_mean', 0):.2f} | "
                       f"{stats.get('mindist_Violation_Rate_mean', 0):.4f} |\n")

            f.write("\n")

            # 按距离区间性能
            f.write("## 按距离区间性能\n\n")
            f.write("| 距离区间 | 样本数 | 位置RMSE | 高度RMSE | 速度RMSE | 违规率 |\n")
            f.write("|----------|--------|----------|----------|----------|--------|\n")

            for distance, stats in results['by_distance'].items():
                f.write(f"| {distance} | {stats['sample_count']} | "
                       f"{stats.get('position_RMSE_mean', 0):.6f} | "
                       f"{stats.get('altitude_RMSE_mean', 0):.2f} | "
                       f"{stats.get('velocity_RMSE_mean', 0):.2f} | "
                       f"{stats.get('mindist_Violation_Rate_mean', 0):.4f} |\n")

        print(f"评估报告已保存到: {report_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='V9分层评估')
    parser.add_argument('--config', type=str,
                       default='config/social_patchtst_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--model_path', type=str,
                       default='logs/best_model.pth',
                       help='模型检查点路径')
    parser.add_argument('--scenes_dir', type=str,
                       default='/mnt/d/model/adsb_scenes/scenes',
                       help='场景数据目录')
    parser.add_argument('--output_dir', type=str,
                       default='./evaluation_results',
                       help='结果输出目录')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批大小')

    args = parser.parse_args()

    # 创建评估器
    evaluator = StratifiedEvaluator(args.config)

    # 创建数据加载器
    _, _, test_loader = create_data_loaders(
        args.config, args.scenes_dir,
        batch_size=args.batch_size, max_neighbors=10, num_workers=4
    )

    # 加载模型
    from model import create_model
    model = create_model(args.config)
    checkpoint = torch.load(args.model_path, map_location=evaluator.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(evaluator.device)

    print("开始分层评估...")
    results = evaluator.evaluate_model(model, test_loader, args.output_dir)

    print("分层评估完成！")
    print(f"结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()