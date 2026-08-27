# nuPlan val 全量（有 cam 覆盖）评估操作手册

> 目的：对 nuPlan **val 集中所有有相机图像覆盖的 log**（225 个，共 101,046 帧样本）做全量 Chamfer 评估，
> 替代之前只有 8 个样本的子集评估，得到更可靠、覆盖全部有图像 log 的同域指标。
> 背景：val_set 传感器分卷不完整，24 个分卷只覆盖 **225/1381** 个 val log，其余 log 无图像无法评估。
> 本手册只评估这 225 个有 cam 数据的 log，即"有 cam 数据的全量"。

---

## 0. 规模与资源预估

| 项 | 数值 | 说明 |
|---|---|---|
| 有 cam 覆盖的 log 数 | 225 / 1381 | 全量 val log 的约 16% |
| 覆盖的帧样本（全帧） | 101,046 | 全量 val info 为 677,119 |
| 推荐解压帧采样（stride=10） | 约 10,100 样本 | 与 train 解压策略一致，覆盖全部 225 log |
| 解压图像体积（stride=10，6 路） | 约 130~180 GB | 已解压 1087 log 约 1TB（平均约 0.9GB/log） |
| 单卡评估时间（stride=10） | 约 1~3 小时 | 视 A100 单卡吞吐与 GT 生成时间 |

> 若磁盘与时间充足，可用 stride=1 全帧（101,046 样本，约 1.3TB 图像，10~30 小时）；
> 否则推荐 stride=10（与训练数据同密度，指标有代表性）。

---

## 1. 前置条件

- 环境：maptr env（bicv01 或 xiaoxuan 的 `/home/xiaoxuan/miniconda3/envs/maptr`），或云服务器等价环境；
  数据在 `/data2/wyc`（NFS 共享，云服务器可直接读）。
- 环境变量（每条命令前都要有，或写成脚本）：
  ```bash
  export PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2
  export CUDA_VISIBLE_DEVICES=0
  ```
- 依赖数据：`reports/sensor_blobs_index_full_val.json`（val 传感器索引）、
  `data/infos/nuplan_map_infos_full_val.pkl`（全量 val info，已绝对路径）。

---

## 2. 步骤一：生成全量 eval info + log 名单

用 `tools/make_val_eval_subset.py`（已支持 `--frame-stride` 与 `--max-samples 0`=全量）：

```bash
PY=/data2/30033/nuplan-devkit/miniconda3/envs/nuplan/bin/python
"$PY" /data2/wyc/nuplan_maptrv2/tools/make_val_eval_subset.py \
  --infos /data2/wyc/nuplan_maptrv2/data/infos/nuplan_map_infos_full_val.pkl \
  --index /data2/wyc/nuplan_maptrv2/reports/sensor_blobs_index_full_val.json \
  --frame-stride 10 \
  --max-samples 0 \
  --out-info /data2/wyc/nuplan_maptrv2/data/infos/nuplan_val_full_cam.pkl \
  --out-logs /data2/wyc/nuplan_maptrv2/data/infos/nuplan_val_full_cam_logs.txt
```

预期输出：`索引覆盖 log 数: 225`、`抽到样本: ~10105, 涉及 log: 225`。

> `--frame-stride` 必须与第 2 步解压的 `--frame-stride` 一致（都用 10），保证每个样本的图像都解压到。
> 想全帧就用 `--frame-stride 1`（约 101,046 样本，需配合 stride=1 解压）。

---

## 3. 步骤二：解压 225 个 log 的图像（本机后台，断线不影响）

```bash
cd /data2/wyc/nuplan_maptrv2
nohup /data2/30033/nuplan-devkit/miniconda3/envs/nuplan/bin/python tools/extract_sensor_images.py \
  --blobs-root /data2/han/nuplan/archives/nuplan-v1.1/sensor_blobs/val_set \
  --index reports/sensor_blobs_index_full_val.json \
  --db-dir raw/nuplan_full/dbs \
  --logs data/infos/nuplan_val_full_cam_logs.txt \
  --out raw/nuplan_full/sensor_blobs \
  --frame-stride 10 \
  --channels CAM_F0 CAM_R0 CAM_L0 CAM_B0 CAM_L2 CAM_R2 \
  --skip-existing \
  > work_dirs/extract_val_full_cam.log 2>&1 &
```

- 进度：`tail -f work_dirs/extract_val_full_cam.log`，看到 `done: extracted N, skipped M -> ...` 即完成。
- 预计新增约 130~180 GB；先确认磁盘余量：`df -h /data2/wyc`。
- 磁盘不足时可把 `nuplan_val_full_cam_logs.txt` 拆成几份分批解压（每份一次跑，`--skip-existing` 可续传）。
- `--channels` 只解压 info 用到的 6 路（默认 8 路会多解压 2 路，浪费约 25% 空间）。

---

## 4. 步骤三：全量评估（chamfer 指标）

```bash
cd /data2/wyc/nuplan_maptrv2/MapTRV2
export PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2
export CUDA_VISIBLE_DEVICES=0

python /data2/wyc/nuplan_maptrv2/tools/test_evaluate.py \
  projects/configs/maptrv2/maptrv2_nuplan_full.py \
  /data2/wyc/nuplan_maptrv2/work_dirs/full/latest.pth \
  --ann-file /data2/wyc/nuplan_maptrv2/data/infos/nuplan_val_full_cam.pkl \
  --map-ann /data2/wyc/nuplan_maptrv2/data/infos/nuplan_map_anns_full_cam.json \
  --max-samples 0
```

- `--max-samples 0` = 全量（脚本默认就是 0）。
- 首次会自动从 info 生成 GT `map_ann`（约 1 万样本，GT 生成需要一些时间）；传新路径避免读到旧缓存。
- 输出：`NuscMap_chamfer/mAP`、`divider/ped_crossing/boundary AP`。
- 建议后台跑并留档：
  ```bash
  nohup python /data2/wyc/nuplan_maptrv2/tools/test_evaluate.py \
    projects/configs/maptrv2/maptrv2_nuplan_full.py \
    /data2/wyc/nuplan_maptrv2/work_dirs/full/latest.pth \
    --ann-file /data2/wyc/nuplan_maptrv2/data/infos/nuplan_val_full_cam.pkl \
    --map-ann /data2/wyc/nuplan_maptrv2/data/infos/nuplan_map_anns_full_cam.json \
    --max-samples 0 \
    > /data2/wyc/nuplan_maptrv2/reports/eval_nuplan/chamfer_full_cam.txt 2>&1 &
  ```

---

## 5. 步骤四：可视化（可选，抽样几张）

```bash
cd /data2/wyc/nuplan_maptrv2/MapTRV2
export PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2
export CUDA_VISIBLE_DEVICES=0

# BEV 对比图（GT 红/预测绿）
python /data2/wyc/nuplan_maptrv2/tools/visualize_nuplan_pred.py \
  projects/configs/maptrv2/maptrv2_nuplan_full.py \
  /data2/wyc/nuplan_maptrv2/work_dirs/full/latest.pth \
  --show-dir /data2/wyc/nuplan_maptrv2/reports/vis_nuplan_full_cam_bev \
  --num-samples 6 --score-thresh 0.3 \
  --ann-file /data2/wyc/nuplan_maptrv2/data/infos/nuplan_val_full_cam.pkl

# 相机投影叠加（默认 6 路全画，自动映射 nuScenes 名 -> nuPlan 磁盘目录）
python /data2/wyc/nuplan_maptrv2/tools/visualize_nuplan_pred_cam.py \
  projects/configs/maptrv2/maptrv2_nuplan_full.py \
  /data2/wyc/nuplan_maptrv2/work_dirs/full/latest.pth \
  --show-dir /data2/wyc/nuplan_maptrv2/reports/vis_nuplan_full_cam_cam \
  --num-samples 4 --score-thresh 0.3 \
  --ann-file /data2/wyc/nuplan_maptrv2/data/infos/nuplan_val_full_cam.pkl
```

---

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| `FileNotFoundError: img file does not exist` | 解压 stride 与 info 的 `--frame-stride` 不一致，或解压未完成；确保都用 10 且看到 done |
| `no archive for <log>/CAM_*` | 该 log 不在索引覆盖内（不在解压名单），重跑步骤一/二确认名单 |
| 磁盘空间不足 | 分批解压、`--channels` 只留 6 路，或改 `--frame-stride 20` 降低图像量 |
| 评估时间过长 | stride=10 单卡约 1~3h；想更快可把 225 log 按比例分到多卡，每卡一份子 info 并行跑 |

---

## 7. 产物

- `data/infos/nuplan_val_full_cam.pkl`：全量有 cam 的 eval info（约 1 万样本，stride=10）
- `data/infos/nuplan_val_full_cam_logs.txt`：225 个有 cam 的 log 名单（供解压）
- `reports/eval_nuplan/chamfer_full_cam.txt`：全量 Chamfer 指标
- `reports/vis_nuplan_full_cam_bev/`、`reports/vis_nuplan_full_cam_cam/`：可视化
