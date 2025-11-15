#!/bin/bash
# pick_and_copy_ultra_fast.sh —— NVMe SSD优化版超高速场景复制
# 充分发挥SN770/SN850等高性能NVMe固态硬盘的全部性能
set -euo pipefail

# 配置路径
SRC_BASE="/mnt/f/adsb/scenes"
DST="/mnt/f/adsb/scenes_picked_ultra"   # NVMe SSD目标目录

# 性能优化参数
CORES=$(nproc)
PARALLEL_JOBS=$((CORES * 3))  # 3倍核心数，充分利用SSD并发能力
IO_BLOCK_SIZE="1M"            # 大块传输
RSYNC_BUFFER_SIZE="64M"       # rsync缓冲区大小

# SSD优化设置函数
optimize_ssd_performance() {
    echo "⚡ 优化SSD性能设置..."

    # 检查是否有NVMe设备
    if lsblk -d -o NAME,ROTA | grep -q "0$" && lsblk | grep -q nvme; then
        echo "   检测到NVMe SSD，应用优化设置..."

        # 设置I/O调度器为none（NVMe SSD最佳）
        for nvme in /sys/block/nvme*; do
            if [ -w "$nvme/queue/scheduler" ]; then
                echo none > "$nvme/queue/scheduler" 2>/dev/null || true
            fi
        done

        # 增加队列深度
        for nvme in /sys/block/nvme*; do
            if [ -w "$nvme/queue/nr_requests" ]; then
                echo 1024 > "$nvme/queue/nr_requests" 2>/dev/null || true
            fi
        done

        echo "   ✅ SSD优化完成"
    else
        echo "   ⚠️  未检测到NVMe SSD，使用默认设置"
    fi
}

# 创建输出目录
mkdir -p "$DST"/{train,val,test}

# 性能优化
optimize_ssd_performance

echo "🚀 NVMe SSD超高速模式：$(printf '%d' $PARALLEL_JOBS)个并发任务，1MB块传输..."
echo "📊 目标：充分发挥SN770/SN850等高性能SSD的全部I/O能力"

total_start=$(date +%s)
start_time=$(date +%s.%N)

# 处理每个数据集
for split in train val test; do
    echo "📂 处理 ${split} 集合..."
    split_start=$(date +%s.%N)

    # 检查CSV文件是否存在
    csv_file="/mnt/f/adsb/stratified_250k_fixed/${split}.csv"
    if [ ! -f "$csv_file" ]; then
        echo "   ❌ CSV文件不存在: $csv_file"
        continue
    fi

    # 从CSV提取场景目录路径（去掉最后的/ego.csv），去重
    tail -n +2 "$csv_file" | \
        cut -d, -f4 | \
        sed 's|/ego.csv||' | \
        sort -u > "$DST/${split}_paths.txt"

    scene_count=$(wc -l < "$DST/${split}_paths.txt")
    echo "   找到 ${scene_count} 个唯一场景"

    if [ $scene_count -eq 0 ]; then
        echo "   ⚠️  ${split} 集合为空，跳过"
        continue
    fi

    # 使用rsync进行超高速并行复制
    echo "   🔥 开始高速复制 ${split} 场景（${PARALLEL_JOBS}并发）..."

    # 创建临时进度文件
    progress_file="$DST/${split}_progress.tmp"
    echo "0" > "$progress_file"

    # 导出变量供子shell使用
    export split DST progress_file

    # 使用rsync + xargs实现高速并行复制
    cat "$DST/${split}_paths.txt" | \
        xargs -n 1 -P $PARALLEL_JOBS -I {} bash -c '
            scene_path="{}"
            scene_name=$(basename "$scene_path")
            target_dir="$DST/$split"

            # 检查源目录是否存在
            if [ ! -d "$scene_path" ]; then
                echo "❌ 不存在: $scene_path" >&2
                exit 1
            fi

            # 使用rsync替代cp，提供���好的性能和进度显示
            rsync -a \
                  --info=progress2 \
                  --no-inc-recursive \
                  --inplace \
                  --whole-file \
                  --block-size="$IO_BLOCK_SIZE" \
                  --buffer-size="$RSYNC_BUFFER_SIZE" \
                  "$scene_path/" \
                  "$target_dir/$scene_name/"

            # 更新进度
            atomic_inc() {
                local file="$1"
                local current
                current=$(cat "$file" 2>/dev/null || echo "0")
                echo $((current + 1)) > "$file.tmp"
                mv "$file.tmp" "$file"
            }
            atomic_inc "$progress_file"

            echo "✅ $scene_name"
        '

    # 统计复制的场景数量
    copied_count=$(ls "$DST/$split" 2>/dev/null | wc -l)
    split_end=$(date +%s.%N)
    split_time=$(echo "$split_end - $split_start" | bc -l)

    # 计算速度
    if [ "$(echo "$split_time > 0" | bc -l)" -eq 1 ]; then
        scenes_per_sec=$(echo "scale=1; $copied_count / $split_time" | bc -l)
        speed_info="(${scenes_per_sec} 场景/秒)"
    else
        speed_info="(超高速!)"
    fi

    echo "   🎉 ${split}: ${copied_count}/${scene_count} 个场景已复制 $speed_info"
    echo "   ⏱️  耗时: $(echo "$split_time" | bc -l) 秒"
done

total_end=$(date +%s.%N)
total_time=$(echo "$total_end - $start_time" | bc -l)

# 最终统计
echo ""
echo "🎉 NVMe SSD超高速复制完成！"
echo "============================================================"

# 清理进度文件
rm -f "$DST"/*_progress.tmp "$DST"/*_paths.txt

total_copied=0
total_size=0

echo "📊 详细统计："
for s in train val test; do
    count=$(ls "$DST/$s" 2>/dev/null | wc -l)
    size=$(du -sb "$DST/$s" 2>/dev/null | cut -f1 || echo "0")
    total_copied=$((total_copied + count))
    total_size=$((total_size + size))

    # 格式化显示大小
    if [ $size -gt $((1024*1024*1024)) ]; then
        size_gb=$(echo "scale=2; $size / (1024^3)" | bc -l)
        size_str="${size_gb} GB"
    elif [ $size -gt $((1024*1024)) ]; then
        size_mb=$(echo "scale=2; $size / (1024^2)" | bc -l)
        size_str="${size_mb} MB"
    else
        size_kb=$(echo "scale=2; $size / 1024" | bc -l)
        size_str="${size_kb} KB"
    fi

    printf "%-6s : %d 个场景 (%s)\n" "$s" "$count" "$size_str"
done

# 计算总体性能
echo "   总计: $total_copied 个场景"

# 格式化总大小
if [ $total_size -gt $((1024*1024*1024)) ]; then
    total_gb=$(echo "scale=2; $total_size / (1024^3)" | bc -l)
    size_str="(${total_gb} GB)"
elif [ $total_size -gt $((1024*1024)) ]; then
    total_mb=$(echo "scale=2; $total_size / (1024^2)" | bc -l)
    size_str="(${total_mb} MB)"
else
    total_kb=$(echo "scale=2; $total_size / 1024" | bc -l)
    size_str="(${total_kb} KB)"
fi

echo "💾 总数据量: $size_str"
echo "⏱️  总耗时: $(echo "$total_time" | bc -l) 秒"

# 计算性能指标
if [ "$(echo "$total_time > 0" | bc -l)" -eq 1 ]; then
    scenes_per_sec=$(echo "scale=1; $total_copied / $total_time" | bc -l)
    if [ $total_size -gt 0 ]; then
        mb_per_sec=$(echo "scale=1; $total_size / (1024*1024) / $total_time" | bc -l)
        echo "⚡ 复制速度: ${scenes_per_sec} 场景/秒, ${mb_per_sec} MB/秒"
    else
        echo "⚡ 复制速度: ${scenes_per_sec} 场景/秒"
    fi
fi

# SSD性能评估
if [ $total_size -gt 0 ] && [ "$(echo "$total_time > 0" | bc -l)" -eq 1 ]; then
    gb_per_sec=$(echo "scale=2; $total_size / (1024^3) / $total_time" | bc -l)
    echo "🔥 NVMe SSD性能: ${gb_per_sec} GB/秒"

    # 性能评级
    if (( $(echo "$gb_per_sec >= 2.0" | bc -l) )); then
        echo "   评级: 🔥🔥🔥 顶级性能 (发挥SSD >80% 能力)"
    elif (( $(echo "$gb_per_sec >= 1.0" | bc -l) )); then
        echo "   评级: 🔥🔥 优秀性能 (发挥SSD 60-80% 能力)"
    elif (( $(echo "$gb_per_sec >= 0.5" | bc -l) )); then
        echo "   评级: 🔥 良好性能 (发挥SSD 40-60% 能力)"
    else
        echo "   评级: ⚠️  有待优化 (发挥SSD <40% 能力)"
    fi
fi

# 生成场景列表
echo ""
echo "📋 生成场景索引文件..."
ls "$DST/train" > "$DST/train_scenes.txt"
ls "$DST/val" > "$DST/val_scenes.txt"
ls "$DST/test" > "$DST/test_scenes.txt"

echo "✅ 场景列表已生成"
echo ""
echo "🎯 数据集准备完成！可以开始训练了！"
echo "💡 提示: 如果需要进一步提升性能，请确保："
echo "   - 使用NVMe SSD (推荐SN770/SN850或同级产品)"
echo "   - 系统电源模式设置为高性能"
echo "   - 关闭不必要的后台程序"
echo ""
echo "🚀 Social-PatchTST训练准备就绪！"