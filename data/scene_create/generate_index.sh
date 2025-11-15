#!/usr/bin/env bash
# generate_index.sh  -  125 万场景 → 30 MB 索引，10 分钟级
# 基于Kimi的优化方案：find+xargs+jq 流水线，绕过Python GIL和系统调用瓶颈

set -e  # 出错立即退出

cd /mnt/f/adsb

echo "🔍 开始生成索引..."
echo "目标：125万场景 → 30MB索引文件"
echo "方法：find + xargs + jq 系统级流水线"
echo "预计耗时：6-8分钟（SN7100）"
echo

start=$SECONDS

# 使用find进行inode顺序读取，xargs多进程jq解析
# 这是最优的I/O模式：顺序读取 + 多进程解析 + 流水线处理
echo "⚡ 启动系统级流水线..."
find scenes -name metadata.json -print0 |
  xargs -0 -P $(nproc) -I{} sh -c '
    d=$(dirname {})
    id=$(basename "$d")
    dist=$(jq -r .mindist_nm "{}" 2>/dev/null || echo 9999)
    printf "%s|%.3f\n" "$id" "$dist"
  ' > scene_index.tsv

# 检查结果
elapsed=$((SECONDS-start))
lines=$(wc -l < scene_index.tsv)
size=$(du -h scene_index.tsv | cut -f1)

echo
echo "✅ 索引生成完成！"
echo "有效行数: $lines"
echo "文件大小: $size"
echo "总耗时: $elapsed 秒"
echo

# 性能统计
if [ $elapsed -gt 0 ]; then
    scenes_per_sec=$((lines / elapsed))
    echo "📊 性能统计:"
    echo "   处理速度: $scenes_per_sec 场景/秒"
    echo "   索引位置: /mnt/f/adsb/scene_index.tsv"
    echo
fi

# 验证数据完整性
echo "🔍 数据验证..."
head -5 scene_index.tsv
echo "..."
tail -5 scene_index.tsv

echo
echo "🎉 索引已就绪！现在可以运行优化版采样脚本："