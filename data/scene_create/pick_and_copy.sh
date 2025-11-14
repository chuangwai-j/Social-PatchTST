#!/bin/bash
# pick_and_copy_fast.sh —— 直接使用CSV路径快速复制场景文件夹
set -euo pipefail

# 配置路径
DST="/mnt/f/adsb/scenes_picked"   # 固态新目录

# 创建输出目录
mkdir -p "$DST"/{train,val,test}

echo "🚀 超快模式：直接从CSV提取场景路径并复制..."

total_start=$(date +%s)

# 处理每个数据集
for split in train val test; do
    echo "📂 处理 ${split} 集合..."

    # 从CSV提取场景目录路径（去掉最后的/ego.csv），去重
    tail -n +2 "/mnt/f/adsb/stratified_250k/${split}.csv" | \
        cut -d, -f4 | \
        sed 's|/ego.csv||' | \
        sort -u > "$DST/${split}_paths.txt"

    scene_count=$(wc -l < "$DST/${split}_paths.txt")
    echo "   找到 ${scene_count} 个唯一场景"

    # 并行复制场景文件夹
    echo "   开始复制 ${split} 场景..."
    cat "$DST/${split}_paths.txt" | \
        xargs -n 1 -P "$(nproc)" -I {} sh -c '
            scene_path="{}"
            scene_name=$(basename "$scene_path")
            target_dir="'$DST/$split'"
            if [ -d "$scene_path" ]; then
                cp -r "$scene_path" "$target_dir/"
                echo "✅ $scene_name"
            else
                echo "❌ 不存在: $scene_path" >&2
            fi
        '

    # 统计复制的场景数量
    copied_count=$(ls "$DST/$split" 2>/dev/null | wc -l)
    echo "   ✅ ${split}: ${copied_count}/${scene_count} 个场景已复制"
done

total_end=$(date +%s)
total_time=$((total_end - total_start))

# 最终统计
echo ""
echo "🎉 复制完成！总耗时: ${total_time} 秒"
total_copied=0
for s in train val test; do
    count=$(ls "$DST/$s" 2>/dev/null | wc -l)
    total_copied=$((total_copied + count))
    printf "%-6s : %d 个场景\n" "$s" "$count"
done
echo "   总计: $total_copied 个场景"

echo "💾 占用空间: $(du -sh "$DST" 2>/dev/null | cut -f1)"

# 生成场景列表
ls "$DST/train" > "$DST/train_scenes.txt"
ls "$DST/val" > "$DST/val_scenes.txt"
ls "$DST/test" > "$DST/test_scenes.txt"

echo "✅ 场景列表已生成"
echo "🎯 可以开始训练了！"