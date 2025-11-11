#!/usr/bin/env python3
"""
ADS-B 轨迹数据提取工具 (V7-Social - 场景生成器)
- 专为 Social-PatchTST 模型设计
- 废弃 V6 (groupby) 逻辑，采用"世界状态"和"基于场景"的提取
- 使用 240 点（20分钟）滑动窗口提取"Ego"和"Neighbors"
- 并行处理以加速
"""

import pandas as pd
import numpy as np
import os
import glob
import argparse
from datetime import datetime, timedelta
import warnings
import random
from tqdm import tqdm
from typing import List, Tuple, Optional
import multiprocessing
import uuid
import functools
import collections

warnings.filterwarnings('ignore')


# ==================== 配置参数 ====================

class Config:
    """配置类 - V7-Social 方案"""

    def __init__(self):
        # 数据路径
        self.INPUT_DIR = "/mnt/d/adsb"
        self.OUTPUT_DIR = "/mnt/d/model/adsb_scenes_v7"  # 【新】V7 场景输出目录

        # 处理参数
        self.MAX_FILES = 2000
        self.RESAMPLE_RATE = "5S"
        self.MIN_TIME_GAP_SECONDS = 180  # 轨迹中断阈值

        # 【V7 窗口参数 - 采纳您的建议】
        self.SEC_PER_POINT = 5
        self.HISTORY_POINTS = 120  # 10分钟历史
        self.FUTURE_POINTS = 120   # 10分钟未来
        self.MIN_TRACK_POINTS = self.HISTORY_POINTS + self.FUTURE_POINTS  # 240点 (20分钟)

        # 【V7 滑动窗口参数】
        # 步长：每 50 秒（10个点）生成一个新场景
        self.SLIDING_WINDOW_STRIDE_POINTS = 10

        # 【V6 黄金数据阈值 - 已废弃】
        # (我们不再做分类，而是做预测)

        # 列定义 (不变)
        self.COLUMN_ORDER = [
            "target_address", "callsign", "timestamp",
            "latitude", "longitude", "geometric_altitude", "flight_level",
            "ground_speed", "track_angle", "vertical_rate", "selected_altitude",
            "lnav_mode", "aircraft_type"
        ]
        self.NUMERIC_COLS = [
            "latitude", "longitude", "geometric_altitude", "flight_level",
            "ground_speed", "track_angle", "vertical_rate", "selected_altitude"
        ]
        self.CATEGORICAL_COLS = ["callsign", "lnav_mode", "aircraft_type"]


# ==================== 核心功能函数 ====================

def resample_aircraft_trajectory(group, config):
    """
    对单架飞机的轨迹进行重采样 (此函数不变，依然重要)
    """
    if len(group) < 2:
        return pd.DataFrame()
    group = group.drop_duplicates(subset=['timestamp'], keep='last')
    if len(group) < 2:
        return pd.DataFrame()
    base_time = datetime(2025, 1, 1)
    timestamps = [base_time + timedelta(seconds=float(ts)) for ts in group['timestamp']]
    group = group.copy()
    group['datetime'] = timestamps
    group = group.set_index('datetime').sort_index()
    resampled_numeric = group[config.NUMERIC_COLS].resample(config.RESAMPLE_RATE).interpolate(method='linear')
    resampled_categorical = group[config.CATEGORICAL_COLS].resample(config.RESAMPLE_RATE).interpolate(method='pad')
    resampled_group = pd.concat([resampled_numeric, resampled_categorical], axis=1)
    target_address = group['target_address'].iloc[0]
    resampled_group['target_address'] = target_address
    resampled_group['timestamp'] = (resampled_group.index - base_time).total_seconds()
    resampled_group = resampled_group.fillna(method='bfill').dropna()
    resampled_group = resampled_group.reset_index(drop=True)
    resampled_group = resampled_group[config.COLUMN_ORDER]
    return resampled_group


# ==================== V7-Social 并行工作函数 ====================

def generate_scenes_from_file(filepath, config):
    """
    【V7 核心逻辑】
    处理单个文件，提取所有 "Ego-Neighbors" 场景
    """
    scenes_generated_count = 0
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return 0

        # --- 1. 构建"世界状态" ---
        # (与V6不同) 我们不 groupby，我们重采样文件中的 *所有* 飞机
        required_cols = ['target_address', 'callsign', 'timestamp'] + config.NUMERIC_COLS + config.CATEGORICAL_COLS
        if not all(col in df.columns for col in required_cols):
            return 0

        resampled_trajectories = []
        for target_address, group in df.groupby('target_address'):
            resampled_track = resample_aircraft_trajectory(group, config)
            if not resampled_track.empty:
                resampled_trajectories.append(resampled_track)

        if not resampled_trajectories:
            return 0

        # world_state_df 包含了此文件中所有飞机的、5秒间隔的、连续的轨迹数据
        world_state_df = pd.concat(resampled_trajectories, ignore_index=True).sort_values(by='timestamp')
        if world_state_df.empty:
            return 0

        # --- 2. 识别"Ego"飞机的长轨迹段 ---
        # 在这里，我们才 groupby 来识别 *连续* 的轨迹

        # 按飞机和时间排序
        world_state_df = world_state_df.sort_values(by=['target_address', 'timestamp'])
        # 识别轨迹中断
        world_state_df['time_gap'] = world_state_df.groupby('target_address')['timestamp'].diff()
        world_state_df['segment_id'] = (world_state_df['time_gap'] > config.MIN_TIME_GAP_SECONDS).cumsum()

        # 遍历 *所有* 连续轨迹段
        for (target_address, segment_id), segment in world_state_df.groupby(['target_address', 'segment_id']):

            # --- 3. 应用"滑动窗口" ---
            # 如果这个连续轨迹段足够长，我们就可以在上面"滑动"240点的窗口
            if len(segment) >= config.MIN_TRACK_POINTS:

                # 在这个长轨迹段上滑动
                for i in range(0, len(segment) - config.MIN_TRACK_POINTS + 1, config.SLIDING_WINDOW_STRIDE_POINTS):

                    ego_track = segment.iloc[i : i + config.MIN_TRACK_POINTS]

                    # 确保窗口是完整的240点
                    if len(ego_track) != config.MIN_TRACK_POINTS:
                        continue

                    t_start = ego_track['timestamp'].min()
                    t_end = ego_track['timestamp'].max()
                    ego_id = ego_track['target_address'].iloc[0]

                    # --- 4. 注入"Social"信息 (查找邻居) ---
                    # 返回"世界状态"，查找在 *同一时间窗口* 内的所有 *其他* 飞机

                    neighbors_df = world_state_df[
                        (world_state_df['timestamp'] >= t_start) &
                        (world_state_df['timestamp'] <= t_end) &
                        (world_state_df['target_address'] != ego_id)
                    ]

                    # --- 5. 清洗和保存"场景" ---
                    # 我们只保留那些 *完整* 存在于此 240 点窗口的邻居
                    complete_neighbors = []
                    for neighbor_id, neighbor_track in neighbors_df.groupby('target_address'):
                        if len(neighbor_track) == config.MIN_TRACK_POINTS:
                            complete_neighbors.append(neighbor_track)

                    # 【重要】我们只保存有"交互"的场景，即至少有1个邻居
                    if not complete_neighbors:
                        continue

                    # 创建场景目录
                    scene_id = str(uuid.uuid4())
                    scene_dir = os.path.join(config.OUTPUT_DIR, "scenes", scene_id)
                    os.makedirs(scene_dir, exist_ok=True)

                    # 保存 Ego 轨迹
                    ego_track.to_csv(os.path.join(scene_dir, "ego.csv"), index=False)

                    # 保存所有邻居的轨迹
                    final_neighbors_df = pd.concat(complete_neighbors, ignore_index=True)
                    final_neighbors_df.to_csv(os.path.join(scene_dir, "neighbors.csv"), index=False)

                    scenes_generated_count += 1

    except Exception as e:
        print(f"  处理文件 {os.path.basename(filepath)} 时出错: {e}")
        pass

    return scenes_generated_count


# ==================== 主处理函数 (并行版) ====================

def process_adsb_data(config):
    """
    主处理函数 (V7 - 并行场景生成器)
    """
    print("=== ADS-B 场景数据提取 - V7-Social (240点) ===")
    print(f"最小轨迹长度: {config.MIN_TRACK_POINTS} 点 ({config.MIN_TRACK_POINTS * config.SEC_PER_POINT / 60:.0f} 分钟)")
    print(f"滑动窗口步长: {config.SLIDING_WINDOW_STRIDE_POINTS} 点 ({config.SLIDING_WINDOW_STRIDE_POINTS * config.SEC_PER_POINT} 秒)")
    print(f"处理文件数: {config.MAX_FILES}")

    # --- 1. 创建输出目录结构 ---
    # 我们只需要一个总的 'scenes' 目录
    scenes_output_dir = os.path.join(config.OUTPUT_DIR, "scenes")
    os.makedirs(scenes_output_dir, exist_ok=True)
    print(f"场景将保存到: {scenes_output_dir}")

    # --- 2. 获取所有数据文件 ---
    all_files = sorted(glob.glob(os.path.join(config.INPUT_DIR, "*.csv")))
    if not all_files:
        print(f"错误：在 {config.INPUT_DIR} 中未找到任何 .csv 文件。")
        return

    print(f"找到 {len(all_files)} 个数据文件")
    files_to_process = all_files[:config.MAX_FILES]
    print(f"处理 {len(files_to_process)} 个文件...")

    # --- 3. 设置并行池 ---
    num_cores = multiprocessing.cpu_count()
    print(f"使用 {num_cores} 个CPU核心并行处理...")

    # "固定" config 参数
    task_processor = functools.partial(generate_scenes_from_file, config=config)

    total_scenes = 0

    with multiprocessing.Pool(num_cores) as pool:
        for scenes_count in tqdm(pool.imap_unordered(task_processor, files_to_process),
                                 total=len(files_to_process), desc="并行处理文件"):
            total_scenes += scenes_count

    # --- 5. 打印最终报告 ---
    print("\n\n--- ✅ 全部处理完毕 (V7-Social) ---")
    print(f"数据已保存到: {scenes_output_dir}")
    print("\n=== 最终数据集统计 ===")
    print(f"总计生成场景数: {total_scenes:,} 个")
    print("每个场景包含一个 'ego.csv' (240点) 和一个 'neighbors.csv' (N*240点)")
    print(f"\n🎯 V7-Social 场景数据生成完毕！")
    print(f"💡 提示：您的 Social-PatchTST 模型现在可以读取这些场景目录进行训练了。")


# ==================== 命令行接口 ====================

def main():
    """
    主函数 - 支持命令行参数
    """
    parser = argparse.ArgumentParser(description='ADS-B 场景数据提取工具 (V7-Social)')
    parser.add_argument('--input-dir', default='/mnt/d/adsb', help='输入数据目录')
    parser.add_argument('--output-dir', default='/mnt/d/model/adsb_scenes_v7', help='输出最终场景的根目录')
    parser.add_argument('--max-files', type=int, default=2000, help='最大处理文件数量')
    parser.add_argument('--stride', type=int, default=10, help='滑动窗口步长 (点数, 默认10点 = 50秒)')

    args = parser.parse_args()

    # 创建配置对象
    config = Config()

    # 应用命令行参数
    config.INPUT_DIR = args.input_dir
    config.OUTPUT_DIR = args.output_dir
    config.MAX_FILES = args.max_files
    config.SLIDING_WINDOW_STRIDE_POINTS = args.stride

    # 重新计算相关参数
    config.SEC_PER_POINT = int(config.RESAMPLE_RATE[:-1]) if config.RESAMPLE_RATE.endswith('S') else 5
    config.HISTORY_POINTS = 120
    config.FUTURE_POINTS = 120
    config.MIN_TRACK_POINTS = config.HISTORY_POINTS + config.FUTURE_POINTS

    # 开始处理
    process_adsb_data(config)


# ==================== 程序入口 ====================

if __name__ == "__main__":
    main()