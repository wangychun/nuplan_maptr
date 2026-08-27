# MapTRV2 三数据集评测汇总（epoch_5）

> checkpoint：`work_dirs/full/epoch_5.pth`（nuPlan 全量训练，6 相机 / 3 类 / num_vec=50）
> 评测协议：Chamfer AP（与官方一致）
> 日期：2026-08-26

## 结果总表

| 数据集 | 样本 | 图像尺寸 | divider AP | ped AP | boundary AP | **mAP** |
|---|---|---|---|---|---|---|
| **nuPlan**（val 子集） | 8 | 960×540 | 0.4568 | 0.0000 | 0.0289 | **0.1619** |
| **nuScenes**（eval 子集） | 8 | 960×540（对齐） | 0.0000 | 0.0000 | 0.0024 | **0.0008** |
| **NVIDIA** | 3 | 鱼眼 | —（无 GT，仅可视化） | — | — | — |

> nuScenes 图像原始 1600×900，评测用 scale=0.6 对齐到 960×540（与 nuPlan 训练一致）。

## 结论

- **nuPlan（同域）**：mAP 0.1619，divider AP 0.4568——模型在自己训练分布上效果合理。
- **nuScenes（跨域）**：mAP ~0.0008，几乎为 0——**真实跨域泛化失败**。
  - 预测分数普遍 0.14~0.29（无 >0.3 的高置信预测），模型对 nuScenes 极不确定。
  - 已排除坐标系差异：用**官方 nuScenes 模型**（maptrv2_nusc_r50_24ep_w_centerline.pth）
    在同样 8 样本上同域评测，mAP **0.558**（divider 0.562 / boundary 0.667 / centerline 0.667），
    证明评测链路正常、nuScenes 原始 (x,y) 坐标正确（**无需 x/y swap**）。
  - 根因：nuScenes 与 nuPlan 相机几何/场景差异大 → 真实泛化差。
- **NVIDIA**：无地图 GT，只能可视化。

> ⚠️ 坐标系结论（重要）：nuScenes info 的 annotation 坐标是**正确**的，
> **不要做 x/y swap**。早前误做的 swap 会把 GT 旋转 90°（与模型预测空间差 90°），
> 导致 mAP 变成 0，且 cam 投影出现"boundary 横着/飘在空中"的错误。
> 详情见 `eval_nusc/official_model_samedomain_8sub.txt`。

## 目录结构

```
reports/
├── eval_nuplan/     nuPlan Chamfer 评测结果
├── eval_nusc/       nuScenes Chamfer 评测结果
├── eval_nvidia/     NVIDIA 说明（无 GT）
├── vis_nuplan_bev/  nuPlan BEV 预测 vs GT（6 张）
├── vis_nuplan_cam/  nuPlan 相机投影（24 张）
├── vis_nusc_bev/    nuScenes BEV 预测 vs GT（6 张）
├── vis_nusc_cam/    nuScenes 相机投影（16 张）
├── vis_nvidia_bev/  NVIDIA BEV 预测（3 张）
└── vis_nvidia_cam/  NVIDIA 相机投影（18 张）
```

## 评测命令

```bash
# nuPlan
python tools/test_evaluate.py MapTRV2/projects/configs/maptrv2/maptrv2_nuplan_full.py \
  work_dirs/full/epoch_5.pth \
  --ann-file data/infos/nuplan_val_eval_sub.pkl \
  --map-ann data/infos/nuplan_map_anns_full_val_sub.json

# nuScenes（尺寸对齐 config，原始坐标 + 绝对路径）
# 注意：info 必须用原始 (x,y) 坐标（不要 swap），并删除旧 map_ann json 让其重新生成
python tools/test_evaluate.py MapTRV2/projects/configs/maptrv2/maptrv2_nusc_eval_align.py \
  work_dirs/full/epoch_5.pth \
  --ann-file data/infos/nuscenes_map_infos_eval_sub_orig_path.pkl \
  --map-ann data/infos/nuscenes_map_anns_eval_sub_orig.json

# 官方 nuScenes 模型同域验证（链路正确性基准）
python tools/test_evaluate.py MapTRV2/projects/configs/maptrv2/maptrv2_nusc_official_eval_8sub.py \
  MapTRV2/ckpts/maptrv2_nusc_r50_24ep_w_centerline.pth \
  --ann-file data/infos/nuscenes_map_infos_eval_sub_orig_path.pkl \
  --map-ann data/infos/nuscenes_official_8sub.json
```
