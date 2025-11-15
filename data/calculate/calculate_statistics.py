#!/usr/bin/env python3
"""
计算训练集全局统计量（主角+配角）
只使用训练集数据，避免数据泄露
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import time
from tqdm import tqdm

def process_features(df):
    """处理特征：提取基本特征并计算速度向量"""
    lat = df['latitude'].values
    lon = df['longitude'].values
    flight_level = df['flight_level'].values
    ground_speed = df['ground_speed'].values
    track_angle = df['track_angle'].values
    vertical_rate = df['vertical_rate'].values
    selected_altitude = df['selected_altitude'].values

    # 转换速度向量 (m/s)
    track_rad = np.deg2rad(track_angle)
    vx = ground_speed * np.sin(track_rad)
    vy = ground_speed * np.cos(track_rad)

    # 组合主要特征 [lat, lon, flight_level, vx, vy]
    processed_features = np.column_stack([
        lat, lon, flight_level, vx, vy
    ])

    # 额外特征 [vertical_rate, selected_altitude]
    extra_features = np.column_stack([
        vertical_rate, selected_altitude
    ])

    return processed_features, extra_features

def calculate_training_statistics(train_paths_file, max_scenes=None, save_path="train_statistics.json"):
    """
    计算训练集全局统计量（只统计主角数据）

    Args:
        train_paths_file: 训练路径文件
        max_scenes: 最大场景数限制，None表示全部
        save_path: 统计量保存路径
    """
    print("🚀 开始计算训练集全局统计量...")
    print(f"   路径文件: {train_paths_file}")
    print(f"   保存路径: {save_path}")
    print("   注意：只统计主角(ego)数据，不包括配角")
    print("   统计特征：主要5个 + 额外2个(vertical_rate, selected_altitude)")

    # 初始化统计变量 - 主要特征（5个）
    feature_sum = np.zeros(5)  # 5个特征的累计和
    feature_sq_sum = np.zeros(5)  # 平方累计和

    # 初始化统计变量 - 额外特征（2个）
    extra_sum = np.zeros(2)  # vertical_rate, selected_altitude的累计和
    extra_sq_sum = np.zeros(2)  # 平方累计和

    total_count = 0
    valid_scenes = 0
    failed_scenes = 0

    # 读取训练场景路径
    scene_paths = []
    with open(train_paths_file, 'r') as f:
        for line in f:
            scene_path = line.strip()
            if scene_path:
                scene_paths.append(scene_path)

    if max_scenes:
        scene_paths = scene_paths[:max_scenes]
        print(f"   限制场景数: {max_scenes}")

    print(f"   总场景数: {len(scene_paths)}")

    # 逐个场景处理
    start_time = time.time()

    for i, scene_path in enumerate(tqdm(scene_paths, desc="处理场景")):
        try:
            # 只处理主角数据
            ego_path = os.path.join(scene_path, "ego.csv")
            if not os.path.exists(ego_path):
                failed_scenes += 1
                continue

            ego_df = pd.read_csv(ego_path)
            main_features, extra_features = process_features(ego_df)  # [seq_len, 5] 和 [seq_len, 2]

            # 累计主要特征统计量（只统计主角）
            feature_sum += np.sum(main_features, axis=0)
            feature_sq_sum += np.sum(main_features ** 2, axis=0)

            # 累计额外特征统计量
            extra_sum += np.sum(extra_features, axis=0)
            extra_sq_sum += np.sum(extra_features ** 2, axis=0)

            total_count += main_features.shape[0]
            valid_scenes += 1

        except Exception as e:
            failed_scenes += 1
            if failed_scenes <= 10:  # 只显示前10个错误
                print(f"   ⚠️  场景 {i+1} 处理失败: {e}")
            continue

    end_time = time.time()

    # 计算全局均值和标准差
    if total_count > 0:
        # 主要特征统计量
        main_mean = feature_sum / total_count
        main_variance = (feature_sq_sum / total_count) - (main_mean ** 2)
        main_std = np.sqrt(np.maximum(main_variance, 1e-8))  # 避免负数或零

        # 额外特征统计量
        extra_mean = extra_sum / total_count
        extra_variance = (extra_sq_sum / total_count) - (extra_mean ** 2)
        extra_std = np.sqrt(np.maximum(extra_variance, 1e-8))  # 避免负数或零

        # 组织统计量
        statistics = {
            "data_info": {
                "total_scenes": len(scene_paths),
                "valid_scenes": valid_scenes,
                "failed_scenes": failed_scenes,
                "total_data_points": int(total_count),
                "processing_time_seconds": end_time - start_time
            },
            "main_features": {
                "feature_names": ["latitude", "longitude", "flight_level", "vx", "vy"],
                "mean": main_mean.tolist(),
                "std": main_std.tolist()
            },
            "extra_features": {
                "feature_names": ["vertical_rate", "selected_altitude"],
                "mean": extra_mean.tolist(),
                "std": extra_std.tolist()
            },
            "all_features": {
                "feature_names": ["latitude", "longitude", "flight_level", "vx", "vy", "vertical_rate", "selected_altitude"],
                "mean": np.concatenate([main_mean, extra_mean]).tolist(),
                "std": np.concatenate([main_std, extra_std]).tolist()
            }
        }

        # 保存统计量
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:  # 只有目录路径不为空时才创建
                os.makedirs(save_dir, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(statistics, f, indent=2)

            print(f"\n💾 统计量已保存到: {save_path}")

        print(f"\n✅ 计算完成!")
        print(f"   有效场景: {valid_scenes:,} / {len(scene_paths):,}")
        print(f"   总数据点: {total_count:,}")
        print(f"   处理时间: {end_time - start_time:.1f} 秒")
        print(f"   平均每场景: {(end_time - start_time) / valid_scenes:.3f} 秒")

        print(f"\n📊 主要特征统计结果:")
        main_feature_names = ["latitude(纬度)", "longitude(经度)", "flight_level(高度)", "vx(东向速度)", "vy(北向速度)"]
        for i, name in enumerate(main_feature_names):
            print(f"   {name:12}: 均值={main_mean[i]:8.3f}, 标准差={main_std[i]:8.3f}")

        print(f"\n📊 额外特征统计结果:")
        extra_feature_names = ["vertical_rate(垂直速度)", "selected_altitude(选中高度)"]
        for i, name in enumerate(extra_feature_names):
            print(f"   {name:16}: 均值={extra_mean[i]:8.3f}, 标准差={extra_std[i]:8.3f}")

        return statistics
    else:
        print(f"❌ 没有有效数据！")
        return None

def test_statistics(saved_stats_path, test_sample_size=10):
    """测试统计量是否正确"""
    print(f"\n🧪 测试统计量...")

    with open(saved_stats_path, 'r') as f:
        stats = json.load(f)

    # 使用主要特征进行测试
    main_mean = np.array(stats['main_features']['mean'])
    main_std = np.array(stats['main_features']['std'])
    extra_mean = np.array(stats['extra_features']['mean'])
    extra_std = np.array(stats['extra_features']['std'])

    # 加载一个测试场景
    train_paths_file = "/mnt/f/adsb/scenes_picked/train_paths.txt"
    with open(train_paths_file, 'r') as f:
        scene_paths = [line.strip() for line in f if line.strip()]

    # 随机采样几个场景测试
    import random
    test_paths = random.sample(scene_paths, min(test_sample_size, len(scene_paths)))

    for i, scene_path in enumerate(test_paths[:3]):  # 只显示前3个
        try:
            ego_path = os.path.join(scene_path, "ego.csv")
            ego_df = pd.read_csv(ego_path)
            main_features, extra_features = process_features(ego_df)

            # 标准化
            main_normalized = (main_features - main_mean) / main_std
            extra_normalized = (extra_features - extra_mean) / extra_std

            print(f"   场景 {i+1}:")
            print(f"     主要特征范围: [{main_features.min():.2f}, {main_features.max():.2f}] -> "
                  f"标准化范围: [{main_normalized.min():.2f}, {main_normalized.max():.2f}]")
            print(f"     额外特征范围: [{extra_features.min():.2f}, {extra_features.max():.2f}] -> "
                  f"标准化范围: [{extra_normalized.min():.2f}, {extra_normalized.max():.2f}]")

        except Exception as e:
            print(f"   场景 {i+1}: 测试失败 - {e}")

def main():
    """主函数"""
    # 训练路径文件
    train_paths_file = "/mnt/f/adsb/scenes_picked/train_paths.txt"

    # 先测试10个场景，验证保存是否正常
    print("🧪 先测试10个场景...")
    stats = calculate_training_statistics(
        train_paths_file,
        max_scenes=10,  # 先测试10个场景
        save_path="train_statistics_test_10.json"
    )

    if stats:
        print(f"\n✅ 测试成功！")
        print(f"   10个场景统计量已保存到: train_statistics_test_10.json")
        print(f"   现在可以开始计算全部175,000个场景了")

        # 询问是否继续计算全部
        user_input = input("\n是否继续计算全部175,000个场景？(y/n): ")
        if user_input.lower() == 'y':
            print("\n🚀 开始计算全部训练集统计量...")
            print("   这将统计175,000个场景的主角(ego)数据")
            print("   预计需要约57分钟")

            stats = calculate_training_statistics(
                train_paths_file,
                max_scenes=None,  # None表示计算全部场景
                save_path="train_statistics_ego_only.json"
            )

            if stats:
                # 测试统计量
                test_statistics("train_statistics_ego_only.json")
                print(f"\n💾 全量统计量已保存到: train_statistics_ego_only.json")
                print(f"   可以在scene_dataset.py中加载并使用这些统计量进行标准化")
        else:
            print("已取消全量计算")
    else:
        print("❌ 测试失败，请检查错误")

if __name__ == "__main__":
    main()