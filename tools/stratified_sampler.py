#!/usr/bin/env python3
"""
V9 场景分层采样脚本
从完整的场景数据集中科学采样，确保数据多样性
"""

import os
import json
import argparse
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import tqdm


def load_scene_metadata(scenes_dir: str) -> List[Dict]:
    """
    加载所有场景的元数据

    Args:
        scenes_dir: 场景目录路径

    Returns:
        场景元数据列表
    """
    scenes = []
    scene_dirs = [d for d in os.listdir(scenes_dir)
                  if os.path.isdir(os.path.join(scenes_dir, d))]

    for scene_dir in scene_dirs:
        scene_path = os.path.join(scenes_dir, scene_dir)
        metadata_path = os.path.join(scene_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                metadata['scene_dir'] = scene_path
                scenes.append(metadata)

    return scenes


def stratified_sampling(scenes: List[Dict],
                         total_target: int = 250000,
                         solo_ratio: float = 0.4,
                         low_risk_ratio: float = 0.3,
                         high_risk_ratio: float = 0.3) -> Dict[str, List[str]]:
    """
    分层采样策略

    Args:
        scenes: 所有场景元数据
        total_target: 目标总场景数
        solo_ratio: 独自飞行场景比例
        low_risk_ratio: 低风险场景比例
        high_risk_ratio: 高风险场景比例

    Returns:
        分层采样结果
    """
    # 分类场景
    solo_scenes = []      # mindist = 9999 (独自飞行)
    low_risk_scenes = []   # 30-50 NM
    high_risk_scenes = []  # < 30 NM

    print("分��场景...")
    for scene in tqdm.tqdm(scenes, desc="分析场景"):
        mindist = scene['mindist_nm']

        if mindist == 9999.0:
            solo_scenes.append(scene)
        elif mindist < 30.0:
            high_risk_scenes.append(scene)
        else:
            low_risk_scenes.append(scene)

    print(f"场景分类完成:")
    print(f"  独自飞行场景: {len(solo_scenes):,} ({len(solo_scenes)/len(scenes)*100:.1f}%)")
    print(f"  低风险场景 (30-50NM): {len(low_risk_scenes):,} ({len(low_risk_scenes)/len(scenes)*100:.1f}%)")
    print(f"  高风险场景 (<30NM): {len(high_risk_scenes):,} ({len(high_risk_scenes)/len(scenes)*100:.1f}%)")

    # 计算各类别目标数量
    solo_target = int(total_target * solo_ratio)
    low_risk_target = int(total_target * low_risk_ratio)
    high_risk_target = int(total_target * high_risk_ratio)

    print(f"\n采样目标 (总计 {total_target:,}):")
    print(f"  轨迹预测基础 (独自飞行): {solo_target:,}")
    print(f"  低风险交互 (30-50NM): {low_risk_target:,}")
    print(f"  高风险交互 (<30NM): {high_risk_target:,}")

    # 执行采样
    result = {}

    # 采样独自飞行场景
    if len(solo_scenes) >= solo_target:
        result['solo'] = random.sample(solo_scenes, solo_target)
        print(f"  ✓ 随机采样 {solo_target:,} 个独自飞行场景")
    else:
        result['solo'] = solo_scenes  # 全部使用
        print(f"  ⚠️  独自飞行场景不足，使用全部 {len(solo_scenes):,} 个")

    # 采样低风险场景
    if len(low_risk_scenes) >= low_risk_target:
        result['low_risk'] = random.sample(low_risk_scenes, low_risk_target)
        print(f"  ✓ 随机采样 {low_risk_target:,} 个低风险场景")
    else:
        result['low_risk'] = low_risk_scenes
        print(f"  ⚠️  低风险场景不足，使用全部 {len(low_risk_scenes):,} 个")

    # 采样高风险场景
    if len(high_risk_scenes) >= high_risk_target:
        result['high_risk'] = random.sample(high_risk_scenes, high_risk_target)
        print(f"  ✓ 随机采样 {high_risk_target:,} 个高风险场景")
    else:
        result['high_risk'] = high_risk_scenes
        print(f"  ⚠️  高风险场景不足，使用全部 {len(high_risk_scenes):,} 个")

    total_selected = sum(len(v) for v in result.values())
    print(f"\n实际采样总数: {total_selected:,}")

    return result


def create_sampled_dataset(sampled_scenes: Dict[str, List[Dict]],
                          output_dir: str) -> str:
    """
    创建采样后的数据集目录

    Args:
        sampled_scenes: 分层采样结果
        output_dir: 输出目录

    Returns:
        输出目录路径
    """
    output_path = os.path.join(output_dir, "scenes_sampled_250k")
    os.makedirs(output_path, exist_ok=True)

    print(f"\n创建采样数据集: {output_path}")

    # 复制场景
    total_copied = 0
    for category, scenes in sampled_scenes.items():
        print(f"复制{category}场景...")

        for scene in tqdm.tqdm(scenes, desc=f"复制{category}"):
            scene_dir = scene['scene_dir']
            scene_id = os.path.basename(scene_dir)
            new_scene_dir = os.path.join(output_path, scene_id)

            # 复制整个场景目录
            shutil.copytree(scene_dir, new_scene_dir, dirs_exist_ok=True)
            total_copied += 1

    print(f"\n✅ 数据集创建完成！")
    print(f"输出路径: {output_path}")
    print(f"包含场景数: {total_copied:,}")

    return output_path


def analyze_sampling_quality(sampled_dir: str):
    """
    分析采样数据集的质量

    Args:
        sampled_dir: 采样数据集目录
    """
    print(f"\n=== 采样数据集质量分析 ===")

    # 加载采样后的场景元数据
    sampled_scenes = load_scene_metadata(sampled_dir)

    # 统计mindist分布
    mindist_values = [scene['mindist_nm'] for scene in sampled_scenes]

    mindist_stats = {
        'min': min(mindist_values),
        'max': max(mindist_values),
        'mean': sum(mindist_values) / len(mindist_values),
        'median': sorted(mindist_values)[len(mindist_values) // 2]
    }

    print(f"Mindist统计 (总场景数: {len(sampled_scenes):,}):")
    print(f"  最小距离: {mindist_stats['min']:.1f} NM")
    print(f"  最大距离: {mindist_stats['max']:.1f} NM")
    print(f"  平均距离: {mindist_stats['mean']:.1f} NM")
    print(f"  中位数距离: {mindist_stats['median']:.1f} NM")

    # 统计邻居数量分布
    neighbor_counts = [scene['n_neighbors'] for scene in sampled_scenes]
    neighbor_stats = {
        'mean': sum(neighbor_counts) / len(neighbor_counts),
        'max': max(neighbor_counts),
        'zero_count': sum(1 for n in neighbor_counts if n == 0)
    }

    print(f"\n邻居数量统计:")
    print(f"  平均邻居数: {neighbor_stats['mean']:.1f}")
    print(f"  最大邻居数: {neighbor_stats['max']}")
    print(f"  无邻居场景: {neighbor_stats['zero_count']} ({neighbor_stats['zero_count']/len(sampled_scenes)*100:.1f}%)")

    return mindist_stats, neighbor_stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='V9场景分层采样脚本')
    parser.add_argument('--scenes-dir', type=str,
                       default='/mnt/d/model/adsb_scenes/scenes',
                       help='原始场景数据目录')
    parser.add_argument('--output-dir', type=str,
                       default='/mnt/d/model/adsb_scenes',
                       help='输出目录')
    parser.add_argument('--total-target', type=int, default=250000,
                       help='目标总场景数')
    parser.add_argument('--solo-ratio', type=float, default=0.4,
                       help='独自飞行场景比例 (0.0-1.0)')
    parser.add_argument('--low-risk-ratio', type=float, default=0.3,
                       help='低风险场景比例 (0.0-1.0)')
    parser.add_argument('--high-risk-ratio', type=float, default=0.3,
                       help='高风险场景比例 (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--analyze-only', action='store_true',
                       help='只分析现有数据，不进行采样')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    # 检查输入目录
    if not os.path.exists(args.scenes_dir):
        print(f"错误：场景目录不存在: {args.scenes_dir}")
        return

    print("=== V9 场景分层采样 ===")
    print(f"输入目录: {args.scenes_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"目标场景数: {args.total_target:,}")
    print(f"采样比例: 独自飞行={args.solo_ratio:.1f}, 低风险={args.low_risk_ratio:.1f}, 高风险={args.high_risk_ratio:.1f}")

    # 加载所有场景元数据
    print("\n加载场景元数据...")
    all_scenes = load_scene_metadata(args.scenes_dir)

    if not all_scenes:
        print("错误：没有找到有效的场景数据")
        return

    print(f"找到 {len(all_scenes):,} 个场景")

    if args.analyze_only:
        # 只分析现有数据
        print("\n跳过采样，仅分析现有数据...")
        mindist_stats, neighbor_stats = analyze_sampling_quality(args.scenes_dir)
        return

    # 执行分层采样
    print(f"\n开始分层采样...")
    sampled_scenes = stratified_sampling(
        all_scenes,
        total_target=args.total_target,
        solo_ratio=args.solo_ratio,
        low_risk_ratio=args.low_risk_ratio,
        high_risk_ratio=args.high_risk_ratio
    )

    # 创建采样数据集
    output_path = create_sampled_dataset(sampled_scenes, args.output_dir)

    # 分析采样质量
    mindist_stats, neighbor_stats = analyze_sampling_quality(output_path)

    print("\n🎯 V9 分层采样完成！")
    print("现在可以使用这个高质量的数据集进行训练，确保模型既能预测正常轨迹，又能处理紧急避让。")


if __name__ == "__main__":
    main()