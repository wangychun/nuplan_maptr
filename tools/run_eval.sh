#!/usr/bin/env bash
# 一键评测 MapTRV2 训练 checkpoint（nuPlan 模型 -> nuPlan / nuScenes）
#
# 用法:
#   bash tools/run_eval.sh work_dirs/full/epoch_7.pth
#   bash tools/run_eval.sh work_dirs/full/latest.pth [--skip-nuplan] [--skip-nusc]
#
# 输出:
#   reports/eval_nuplan/chamfer_<tag>.txt
#   reports/eval_nusc/chamfer_<tag>.txt
#   reports/vis_nuplan_cam/ 与 reports/vis_nusc_cam/（cam 投影，z=-1.6 贴合地面）
#
# 说明:
#   - nuScenes 评测用原始 (x,y) 坐标（不 swap），并传新 map-ann 让 _format_gt 重新生成
#   - nuScenes 评测必须用 scale=0.5 配置（maptrv2_nusc_official_eval_3cls_align.py）：
#     官方 nuScenes 模型训练分辨率 800×450（scale=0.5）；用 scale=0.6(960×540) 会得到假象 mAP=0
#   - 评测前会自动删除旧 map_ann json，避免读到坐标错误的缓存
#   - 环境: maptr env + PYTHONPATH=MapTRV2 + cwd=项目根
set -euo pipefail

CKPT="${1:?用法: bash tools/run_eval.sh <checkpoint.pth> [--skip-nuplan] [--skip-nusc]}"
shift || true
SKIP_NUPLAN=0; SKIP_NUSC=0
for a in "$@"; do
  case "$a" in
    --skip-nuplan) SKIP_NUPLAN=1 ;;
    --skip-nusc) SKIP_NUSC=1 ;;
    *) echo "未知参数: $a"; exit 1 ;;
  esac
done

ROOT=/data2/wyc/nuplan_maptrv2
PY=/home/bicv01/miniforge3/envs/maptr/bin/python
export PATH=/home/bicv01/miniforge3/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/home/bicv01/miniforge3/envs/maptr/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$ROOT/MapTRV2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$ROOT"

TAG=$(basename "$CKPT" .pth)
echo "==== 评测 checkpoint: $CKPT (tag=$TAG) ===="
[ -f "$CKPT" ] || { echo "checkpoint 不存在: $CKPT"; exit 1; }

# ---------- nuPlan 同域 ----------
if [ "$SKIP_NUPLAN" = "0" ]; then
  echo ""
  echo "==== [1/3] nuPlan 同域评测 ===="
  mkdir -p reports/eval_nuplan
  rm -f data/infos/nuplan_map_anns_full_val_sub_$TAG.json
  $PY tools/test_evaluate.py \
    MapTRV2/projects/configs/maptrv2/maptrv2_nuplan_full.py \
    "$CKPT" \
    --ann-file data/infos/nuplan_val_eval_sub.pkl \
    --map-ann data/infos/nuplan_map_anns_full_val_sub_$TAG.json \
    2>&1 | tee reports/eval_nuplan/chamfer_$TAG.txt | grep -E "\| (divider|ped_crossing|boundary|mAP)|NuscMap_chamfer/mAP" | tail -6
fi

# ---------- nuScenes 跨域 ----------
if [ "$SKIP_NUSC" = "0" ]; then
  echo ""
  echo "==== [2/3] nuScenes 跨域评测（原始坐标不 swap，scale=0.5）===="
  mkdir -p reports/eval_nusc
  rm -f data/infos/nuscenes_map_anns_eval_sub_orig_$TAG.json
  $PY tools/test_evaluate.py \
    MapTRV2/projects/configs/maptrv2/maptrv2_nusc_official_eval_3cls_align.py \
    "$CKPT" \
    --ann-file data/infos/nuscenes_map_infos_eval_sub_orig_path.pkl \
    --map-ann data/infos/nuscenes_map_anns_eval_sub_orig_$TAG.json \
    2>&1 | tee reports/eval_nusc/chamfer_$TAG.txt | grep -E "\| (divider|ped_crossing|boundary|mAP)|NuscMap_chamfer/mAP" | tail -6
fi

# ---------- cam 可视化 ----------
echo ""
echo "==== [3/3] cam 投影可视化（z=-1.6 贴合地面）===="
$PY tools/visualize_nuplan_pred_cam.py \
  MapTRV2/projects/configs/maptrv2/maptrv2_nuplan_full.py \
  "$CKPT" \
  --show-dir reports/vis_nuplan_cam \
  --num-samples 4 --score-thresh 0.3 \
  --ann-file data/infos/nuplan_val_eval_sub.pkl 2>&1 | tail -1
$PY tools/visualize_nuplan_pred_cam.py \
  MapTRV2/projects/configs/maptrv2/maptrv2_nusc_official_eval_3cls_align.py \
  "$CKPT" \
  --show-dir reports/vis_nusc_cam \
  --num-samples 4 --score-thresh 0.3 \
  --ann-file data/infos/nuscenes_map_infos_eval_sub_orig_path.pkl 2>&1 | tail -1

echo ""
echo "==== 完成！结果: ===="
echo "  nuPlan:  reports/eval_nuplan/chamfer_$TAG.txt"
echo "  nuScenes: reports/eval_nusc/chamfer_$TAG.txt"
