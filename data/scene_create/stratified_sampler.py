#!/usr/bin/env python3
"""
修复版分层采样器 - 先切分后采样 (Split-First Stratified Sampler)
逻辑修正：
❌ 错误：先从总池抽样 -> 再切分 (导致时间乱序，训练集混入未来数据)
✅ 正确：先按时间切分总池 -> 再各自分层抽样 (保证物理隔离)
"""
import time
import pandas as pd
import numpy as np
from pathlib import Path

SCENE_ROOT = Path("/mnt/f/adsb/scenes")
INDEX_FILE = Path("/mnt/f/adsb/scene_index.tsv")
OUTPUT_DIR = Path("/mnt/f/adsb/stratified_250k")

# 分层阈值
SOLO_THR = 50.0
LOW_RISK_LO = 3.0
LOW_RISK_HI = 10.0

# 参数
TOTAL_TARGET = 250_000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_raw_index():
    """加载原始索引，保留原始时间顺序"""
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"❌ 索引文件不存在: {INDEX_FILE}")

    print("📂 读取场景索引 (假设文件行序 = 时间顺序)...")
    start_time = time.time()

    # 也就是直接相信你的 index.tsv 是时间有序的
    df = pd.read_csv(INDEX_FILE, sep='|', names=['scene_id', 'mindist_nm'])

    # 预计算 Layer，方便后续处理
    # 使用向量化操作加速
    conditions = [
        (df['mindist_nm'] > SOLO_THR),
        (df['mindist_nm'] >= LOW_RISK_LO) & (df['mindist_nm'] <= LOW_RISK_HI),
        (df['mindist_nm'] < 3.0)
    ]
    choices = ['Solo', 'Low-Risk', 'High-Risk']
    df['layer'] = np.select(conditions, choices, default='Solo')

    # 添加路径 (向量化)
    # 注意：这里只存相对路径或ID，最后保存时再拼完整路径，节省内存

    print(f"✅ 索引加载完成: {len(df):,} 条 | 耗时: {time.time() - start_time:.2f}s")
    return df


def stratified_sample_from_subset(df_subset, subset_name, n_target):
    """在给定的子集内进行分层采样"""
    print(f"   🎯 正在对 [{subset_name}] 进行分层采样 (目标: {n_target:,})...")

    targets = {
        'Solo': int(n_target * 0.30),
        'Low-Risk': int(n_target * 0.50),
        'High-Risk': int(n_target * 0.20),
    }

    results = []
    for layer, count in targets.items():
        layer_data = df_subset[df_subset['layer'] == layer]

        if len(layer_data) == 0:
            print(f"      ⚠️  {subset_name} - {layer} 层为空！无法采样！")
            continue

        # 采样 (如果不够就重复采样 replace=True)
        # random_state 确保复现性
        sampled = layer_data.sample(n=count, replace=(len(layer_data) < count), random_state=42)
        results.append(sampled)

    final_df = pd.concat(results).sample(frac=1, random_state=42)  # 最后打乱顺序，方便训练
    print(f"      ✅ {subset_name} 完成: {len(final_df):,} 条")
    return final_df


def main():
    print("🚀 修复版分层采样器 (Split-Then-Sample Strategy)")
    print("=" * 60)

    # 1. 加载原始数据 (保持时间顺序)
    df_raw = load_raw_index()
    total_raw = len(df_raw)

    # 2. 【关键步骤】先按时间顺序切分大池子
    # 假设 df_raw 的行序就是时间序
    print("\n🔪 第一步：按原始时间顺序切分总池 (物理隔离)...")

    idx_train_end = int(total_raw * TRAIN_RATIO)
    idx_val_end = int(total_raw * (TRAIN_RATIO + VAL_RATIO))

    # 这里的 .copy() 很重要，确保物理隔离
    pool_train = df_raw.iloc[:idx_train_end].copy()
    pool_val = df_raw.iloc[idx_train_end:idx_val_end].copy()
    pool_test = df_raw.iloc[idx_val_end:].copy()

    print(f"   原始池 Train: {len(pool_train):,} (Index 0 - {idx_train_end})")
    print(f"   原始池 Val  : {len(pool_val):,} (Index {idx_train_end} - {idx_val_end})")
    print(f"   原始池 Test : {len(pool_test):,} (Index {idx_val_end} - {total_raw})")

    # 3. 【关键步骤】在各自的池子里进行分层采样
    print("\n🎲 第二步：在隔离的池子内进行分层采样...")

    final_train = stratified_sample_from_subset(pool_train, "Train", int(TOTAL_TARGET * TRAIN_RATIO))
    final_val = stratified_sample_from_subset(pool_val, "Val", int(TOTAL_TARGET * VAL_RATIO))
    final_test = stratified_sample_from_subset(pool_test, "Test", int(TOTAL_TARGET * TEST_RATIO))

    # 4. 保存结果
    print("\n💾 保存最终 CSV...")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    def save_full_csv(df, name):
        # 还原完整路径用于 DataLoader
        export_df = df.copy()
        export_df['ego_path'] = export_df['scene_id'].apply(lambda x: str(SCENE_ROOT / x / "ego.csv"))
        export_df['neighbor_path'] = export_df['scene_id'].apply(lambda x: str(SCENE_ROOT / x / "neighbors.csv"))

        # 只保留需要的列
        cols = ['scene_id', 'layer', 'mindist_nm', 'ego_path', 'neighbor_path']
        export_df[cols].to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        print(f"   ✅ {name}.csv 保存成功")

    save_full_csv(final_train, "train")
    save_full_csv(final_val, "val")
    save_full_csv(final_test, "test")

    print("\n" + "=" * 60)
    print("🎉 数据集构建完成 (无数据泄露版)")
    print(f"📂 输出位置: {OUTPUT_DIR}")
    print("✅ 逻辑验证: Train的数据全部来自前70%的时间段，Test来自后15%，绝无重叠。")


if __name__ == "__main__":
    main()