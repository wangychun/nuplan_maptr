#!/bin/bash
# 查看 nuplan_maptrv2 最新训练进度
# 用法: bash /data2/wyc/nuplan_maptrv2/tools/show_progress.sh

LOG=$(ls -t /data2/wyc/nuplan_maptrv2/work_dirs/*/*.log 2>/dev/null | head -1)
if [ -z "$LOG" ]; then
    echo "未找到训练日志（work_dirs/*/*.log）"
    exit 1
fi
echo "日志: $LOG"
echo "=== 最近 1 条进度 ==="
tail -1 "$LOG" | grep -oE 'Epoch \[[0-9]+\]\[[0-9]+/[0-9]+\].*' | cut -c1-170
echo "=== 最近 3 条 loss ==="
grep -oE 'Epoch \[[0-9]+\]\[[0-9]+/[0-9]+\].*loss: [0-9.]+' "$LOG" | tail -3 | sed -E 's/, data_time[^,]*,/,/; s/, memory[^,]*,/,/' | cut -c1-150
echo "=== 已保存 checkpoint ==="
CKPT_DIR=$(dirname "$LOG")
ls -la "$CKPT_DIR"/*.pth 2>/dev/null || echo '暂无 checkpoint（当前 epoch 结束后保存）'
