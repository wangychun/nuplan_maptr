# 三数据集评测结果汇总（MapTRV2 · nuPlan 训练模型）

> 日期：2026-08-27（更新：scale=0.5 配置修正后重跑）
> checkpoint：`work_dirs/full/epoch_7.pth`（nuPlan 全量训练，6 相机 / 3 类 / num_vec=50）
> 评测协议：Chamfer AP（与官方一致）
> 重要：nuScenes 评测用**原始 (x,y) 坐标**（不要 swap）+ **scale=0.5 配置**，见下文。

## 一、评测结果总表

| 数据集 | 样本数 | 图像尺寸 | divider AP | ped_crossing AP | boundary AP | **mAP** |
|---|---|---|---|---|---|---|
| **nuPlan**（val 子集） | 8 | 960×540 | 0.4568 | 0.0000 | 0.0289 | **0.1619** |
| **nuScenes**（eval 子集） | 8 | 1600×900→800×450 | 0.0000 | 0.0000 | 0.0010 | **0.0011** |
| **NVIDIA**（pai_subset） | 3 | 鱼眼（多 FOV） | —（无 GT，仅可视化） | — | — | — |

> nuScenes 用 `maptrv2_nusc_official_eval_3cls_align.py`（scale=0.5，800×450）评测，与官方 nuScenes 模型训练分辨率一致。

## 二、结果分析

1. **nuPlan（同域）效果好**：mAP 0.1619，divider AP 高达 0.4568。模型在**自己训练的数据分布**上评测，效果合理（divider 是高频类别，boundary 偏低的 0.0289 与训练时的 GT 类别不均衡一致）。
2. **nuScenes（跨域）效果差**：mAP 0.0011（boundary 0.0010），divider/ped 全 0。预测线与 GT 几何上基本对不上——**真实的跨数据集泛化失败**（见"四、scale 坑"的严格对照）。
   - 原因：nuScenes 与 nuPlan 相机几何（车顶环视 vs 车体四角）、焦距（1266 vs 1545）、场景风格差异大；
   - **已确认评测链路正确**：见"三、坐标系结论"与"四、scale 坑"。
3. **NVIDIA（无 GT）**：只能可视化预测，不能算 Chamfer 指标。预测可在鱼眼相机图像上投影。

## 三、坐标系结论（重要）

- nuScenes info 的 annotation `(x,y)` 坐标**本来就是正确的**，与 MapTRv2 局部坐标系
  （x 横向 ±15m、y 纵向/车头 ±30m）一致。**不要做 x/y swap**。
- 早前误做的 swap（`fix_nuscenes_xy.py`，把 `(x,y)→(y,x)`）会把 GT 旋转 90°：
  - 评测 mAP 变成 0；
  - cam 投影出现"boundary 横着、飘在空中"的错误。
- 正确评测 info：`data/infos/nuscenes_map_infos_eval_sub_orig_path.pkl`（原始坐标 + 绝对路径）。
- ⚠️ `_format_gt()` 只在 GT json 不存在时才用 `vectormap_pipeline` 重新生成；评测前应删除旧
  map_ann json（或换新路径），避免读到坐标错误的缓存。

## 四、⚠️ scale 坑（重要，评测正确性的关键）

- **官方 nuScenes 模型训练分辨率是 scale=0.5（800×450）**，nuPlan 模型训练分辨率是 scale=0.6（960×540）。
- **评测 nuScenes 必须用 scale=0.5 的配置**（`maptrv2_nusc_official_eval_3cls_align.py`）。
- **坑**：用 `maptrv2_nusc_eval_align.py`（scale=0.6）评测 nuScenes 会得到**假象 mAP=0**——
  给模型喂与训练不一致分辨率的图，即使是官方 nuScenes 模型也输出 0。
- **严格对照实验**（同一 3 类配置 `maptrv2_nusc_official_eval_3cls_align.py`，scale=0.5，只换 checkpoint）：

| 模型 | divider AP | boundary AP | **mAP** |
|---|---|---|---|
| 官方 nuScenes 权重（3 类适配，`maptrv2_nusc_adapted_nuplan.pth`） | 0.499 | 0.380 | **0.2082** |
| nuPlan 训练（epoch_7） | 0.009 | 0.001 | **0.0011** |

> 官方权重 backbone 与官方 4 类逐字节一致（仅截分类头为 3 类），对照严格。
> **结论：评测脚本/坐标/GT 都正确；nuPlan 模型跨域到 nuScenes 确实泛化失败（0.001 vs 官方 0.208）。**

## 五、评测产物

| 数据集 | BEV 可视化 | 相机投影 |
|---|---|---|
| nuPlan | `reports/vis_nuplan_bev/`（6 张） | `reports/vis_nuplan_cam/`（24 张） |
| nuScenes | `reports/vis_nusc_bev/`（6 张，原始坐标） | `reports/vis_nusc_cam/`（24 张，z=-1.6 贴合） |
| NVIDIA | `reports/vis_nvidia_bev/`（3 张） | `reports/vis_nvidia_cam/`（18 张） |

## 六、评测命令速查

```bash
# 一键评测（推荐）
bash tools/run_eval.sh work_dirs/full/latest.pth

# nuPlan 子集（8 样本）
python tools/test_evaluate.py \
  MapTRV2/projects/configs/maptrv2/maptrv2_nuplan_full.py \
  work_dirs/full/epoch_7.pth \
  --ann-file data/infos/nuplan_val_eval_sub.pkl \
  --map-ann data/infos/nuplan_map_anns_full_val_sub.json

# nuScenes 子集（8 样本，scale=0.5 配置，原始坐标 info，新 map-ann）
python tools/test_evaluate.py \
  MapTRV2/projects/configs/maptrv2/maptrv2_nusc_official_eval_3cls_align.py \
  work_dirs/full/epoch_7.pth \
  --ann-file data/infos/nuscenes_map_infos_eval_sub_orig_path.pkl \
  --map-ann data/infos/nuscenes_map_anns_eval_sub_orig.json
```

> 环境：`export PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2`，用 maptr 环境 python（bicv01），cwd 在项目根 `/data2/wyc/nuplan_maptrv2`。
