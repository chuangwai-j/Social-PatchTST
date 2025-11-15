#!/usr/bin/env python3
"""
优化版分层采样器 - 读取预生成索引
30 % Solo | 50 % Low-Risk | 20 % High-Risk
使用索引文件，避免目录扫描，秒级完成
"""
import time, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

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

def load_scenes_from_index():
    """从索引文件加载场景数据"""
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"❌ 索引文件不存在: {INDEX_FILE}\n请先运行: bash data/scene_create/generate_index.sh")

    print("📂 读取场景索引...")
    start_time = time.time()

    # 读取索引文件
    df_index = pd.read_csv(INDEX_FILE, sep='|', names=['scene_id', 'mindist_nm'])
    load_time = time.time() - start_time

    print(f"✅ 索引载入完成：{len(df_index):,} 条 | 耗时: {load_time:.2f}秒")

    # 转换为完整记录
    print("🔄 转换为训练数据格式...")
    convert_start = time.time()

    records = []
    for _, row in df_index.iterrows():
        mindist = float(row.mindist_nm)
        scene_id = row.scene_id

        # 分层逻辑
        if mindist > SOLO_THR:
            layer = 'Solo'
        elif LOW_RISK_LO <= mindist <= LOW_RISK_HI:
            layer = 'Low-Risk'
        elif mindist < 3.0:
            layer = 'High-Risk'
        else:
            layer = 'Solo'  # 兜底

        records.append({
            'scene_id': scene_id,
            'layer': layer,
            'mindist_nm': mindist,
            'ego_path': str(SCENE_ROOT / scene_id / "ego.csv"),
            'neighbor_path': str(SCENE_ROOT / scene_id / "neighbors.csv"),
        })

    convert_time = time.time() - convert_start
    print(f"✅ 数据转换完成：{len(records):,} 条 | 耗时: {convert_time:.2f}秒")

    return records, load_time, convert_time

def main():
    print("🚀 优化版分层采样器启动 (使用索引文件)")
    print("="*60)

    # 记录总开始时间
    total_start_time = time.time()

    # 1. 加载数据
    records, load_time, convert_time = load_scenes_from_index()
    df_all = pd.DataFrame(records)

    # 2. 检查分层分布
    print(f"\n📊 各层分布:")
    layer_distribution = df_all['layer'].value_counts()
    for layer, count in layer_distribution.items():
        percentage = count / len(df_all) * 100
        print(f"  {layer}: {count:,} 条 ({percentage:.1f}%)")

    # 3. 分层采样
    print(f"\n🎯 开始分层采样...")
    sample_start = time.time()

    def sample_layer(g, n):
        return g.sample(n=n, replace=len(g) < n, random_state=42)

    layer_targets = {
        'Solo': int(TOTAL_TARGET * 0.30),
        'Low-Risk': int(TOTAL_TARGET * 0.50),
        'High-Risk': int(TOTAL_TARGET * 0.20),
    }

    print(f"目标采样: Solo {layer_targets['Solo']:,} | Low-Risk {layer_targets['Low-Risk']:,} | High-Risk {layer_targets['High-Risk']:,}")

    sampled = (df_all.groupby('layer', group_keys=False)
                     .apply(lambda g: sample_layer(g, layer_targets[g.name])))

    sample_time = time.time() - sample_start
    print(f"✅ 采样完成：{len(sampled):,} 条 | 耗时: {sample_time:.2f}秒")

    # 4. 划分数据集
    print(f"\n🎯 划分训练/验证/测试集...")
    split_start = time.time()

    train, temp = train_test_split(sampled, stratify=sampled['layer'],
                                   train_size=TRAIN_RATIO, random_state=42)
    val, test = train_test_split(temp, stratify=temp['layer'],
                                 train_size=VAL_RATIO/(VAL_RATIO+TEST_RATIO), random_state=42)

    split_time = time.time() - split_start
    print(f"✅ 数据划分完成 | 耗时: {split_time:.2f}秒")

    # 5. 输出CSV
    print(f"\n💾 保存CSV文件...")
    output_start = time.time()

    OUTPUT_DIR.mkdir(exist_ok=True)
    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    output_time = time.time() - output_start
    total_time = time.time() - total_start_time

    # 6. 统计报告
    print(f"\n" + "="*60)
    print(f"🎉 优化版 25 万条分层采样完成")
    print(f"="*60)

    for name, df in zip(('Train', 'Val', 'Test'), (train, val, test)):
        layer_counts = df['layer'].value_counts()
        print(f"{name:6s}: {len(df):,} 条 | 分层比例: ", end="")
        for layer in ['Solo', 'Low-Risk', 'High-Risk']:
            count = layer_counts.get(layer, 0)
            pct = count / len(df) * 100
            print(f"{layer} {pct:.0f}% ", end="")
        print()

    print(f"\n⏱️  性能统计:")
    print(f"   索引加载: {load_time:.2f}秒")
    print(f"   数据转换: {convert_time:.2f}秒")
    print(f"   分层采样: {sample_time:.2f}秒")
    print(f"   数据划分: {split_time:.2f}秒")
    print(f"   CSV输出: {output_time:.2f}秒")
    print(f"   总耗时: {total_time:.2f}秒")

    print(f"\n📂 输出目录: {OUTPUT_DIR}")
    print(f"✅ 数据已就绪，可直接开始训练！")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e)
        print(f"\n💡 解决方案:")
        print(f"   1. 先运行: bash data/scene_create/generate_index.sh")
        print(f"   2. 然后运行: python data/scene_create/stratified_sampler.py")
    except Exception as e:
        print(f"❌ 执行失败: {e}")