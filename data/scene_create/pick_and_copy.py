#!/usr/bin/env python3
"""
WD Black SN770 专属极速复制工具
原理：单进程 + 64线程并发 + 零进程开销
"""
import os
import shutil
import time
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ================= 配置 =================
# 你的 CSV 目录
CSV_DIR = Path("/mnt/f/adsb/stratified_250k")
# 目标目录
DST_ROOT = Path("/mnt/f/adsb/scenes_picked")
# 线程数：WD Black SN770 队列深度极大，建议开 32-64
NUM_THREADS = 64


# =======================================

def copy_worker(args):
    src, dst = args
    # 如果目标已存在，跳过（实现断点续传）
    if dst.exists():
        return
    try:
        # 纯粹的 I/O 操作，Python 对此优化极好
        shutil.copytree(src, dst, dirs_exist_ok=True)
    except Exception as e:
        pass  # 忽略个别错误，不卡断


def process_split(split):
    csv_path = CSV_DIR / f"{split}.csv"
    if not csv_path.exists(): return

    # 读取源路径
    print(f"📖 解析 {split} 索引...")
    df = pd.read_csv(csv_path)

    tasks = []
    dst_split_dir = DST_ROOT / split
    dst_split_dir.mkdir(parents=True, exist_ok=True)

    # 预计算路径，避免在多线程中做字符串操作
    print(f"⚡ 构建任务队列...")
    for _, row in df.iterrows():
        scene_id = row['scene_id']

        # 解析源路径 (从 CSV 里的 ego_path 反推文件夹)
        # 假设 CSV 里 ego_path 是绝对路径 /mnt/f/adsb/scenes/UUID/ego.csv
        # 如果只有 scene_id，则手动拼接
        if 'ego_path' in row:
            src_path = Path(row['ego_path']).parent
        else:
            # 兜底逻辑
            src_path = Path("/mnt/f/adsb/scenes") / scene_id

        dst_path = dst_split_dir / scene_id
        tasks.append((src_path, dst_path))

    print(f"🔥开始复制 {split} (并发: {NUM_THREADS})...")
    start_t = time.time()

    # 使用 ThreadPoolExecutor 榨干 IOPS
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # tqdm 只负责 UI，不涉及磁盘 I/O，非常轻量
        list(tqdm(executor.map(copy_worker, tasks), total=len(tasks), unit="scene", unit_scale=True))

    end_t = time.time()
    duration = end_t - start_t
    speed = len(tasks) / duration if duration > 0 else 0
    print(f"✅ {split} 完成 | 耗时: {duration:.1f}s | 速度: {speed:.1f} 场景/秒")


def main():
    print("🚀 复制引擎启动")


    print(f"🎯 源索引: {CSV_DIR}")
    print(f"📂 目标: {DST_ROOT}")
    print("-" * 50)

    total_start = time.time()

    for split in ['train', 'val', 'test']:
        process_split(split)

    # 生成最终的文件列表 (用于 AutoDL 验证)
    print("-" * 50)
    print("📝 生成文件列表索引...")
    for split in ['train', 'val', 'test']:
        list_file = DST_ROOT / f"{split}_paths.txt"
        try:
            scenes = os.listdir(DST_ROOT / split)
            with open(list_file, 'w') as f:
                f.write('\n'.join(scenes))
        except:
            pass

    print(f"\n🎉 全部完成！总耗时: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()