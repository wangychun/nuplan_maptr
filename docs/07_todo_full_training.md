# 完整 nuPlan 训练 —— 保姆级待办（更新版）

> 目标：用完整 nuPlan 训练集（约 1085 个 train log）重新训练 MapTRV2，替代 mini（64 log）基线。  
> 数据格式目标（2026-08-24 起）：nuPlan info 对齐 nuScenes 在 MapTR 中使用的格式（参考 `docs/09_ref.md`），可直接复用官方 `CustomNuScenesOfflineLocalMapDataset`，不再依赖自定义 dataset。  
> 铁律：`/data2/han` 下的数据只读，绝不写、不改、不删；所有解压产物、索引、info、checkpoint 一律放 `/data2/wyc/nuplan_maptrv2/`。

---

## 📌 当前进度（2026-08-24 更新）

| 步骤 | 状态 | 说明 |
|---|---|---|
| 步骤 1 解压 DB | ✅ 已完成 | `raw/nuplan_full/dbs/` = **14506 个 .db**（train 13125 + val 1381） |
| 步骤 2 log 名单 | ✅ 已完成 | `full_train_logs.txt` = 1085 行；`full_val_logs.txt` = 1381 行（已从 val.zip 提取，与 train 0 重叠） |
| 步骤 3 扫描索引 | ✅ 已完成 | `sensor_blobs_index_full_train.json`（70 分卷，43 camera 够用）+ `sensor_blobs_index_full_val.json` 均已生成 |
| 步骤 4 解压图像 | ✅ 已完成 | train + val 图像已全部解压（约 498 万张） |
| 步骤 5 生成 info | ✅ 已完成（已修正为 6 路） | train/val info 已生成，并用 `tools/reduce_infos_cams.py` 修正为 **6 路 nuScenes 命名**（见 6.3），可直接用于 `num_cams=6` 配置 |
| 步骤 6 校验 info | ⬜ 未开始 | — |
| 步骤 7 改配置+训练 | ⚠️ 配置已改好 | `maptrv2_nuplan_full.py` 已改为官方 `CustomNuScenesOfflineLocalMapDataset` + 6 路（`num_cams=6`），待 6.3 重跑 info 后启动训练 |

> ✅ val 名单已修正（2026-08-18）：`full_val_logs.txt` = 1381 行（从 `nuplan-v1.1_val.zip` 的目录提取，与 train 名单 0 重叠）。不要用 `ls raw/nuplan_full/dbs/` 生成——那个目录是 train+val 混合的，ls 会得到 13125 行。正确方法：
> ```bash
> unzip -l /data2/han/nuplan/archives/nuplan-v1.1/nuplan-v1.1_val.zip "*.db" \
>   | awk '/\.db$/ {n=$4; sub(/^.*\//,"",n); print n}' \
>   | sed 's/\.db$//' | sort -u > configs/splits/full_val_logs.txt
> ```

> 🔧 重要提醒（更新）：
> - camera-only 训练不需要 lidar：MapTRV2 仅使用环视相机图像，训练/生成 info 不依赖 lidar。因此步骤 3 无需等待 lidar 分卷迁移完成，现有索引中的 43 个 camera 分卷已足够，可直接进行步骤 4。
> - val DB 必须解压：生成 val info 前需解压 val 的 1381 个 .db（见步骤 1 补充命令）。解压后 `ls raw/nuplan_full/dbs/*.db | wc -l` 应约 14506。

---

## 0. 数据总览（先看明白再动手）

### 0.1 只读源数据（在 `/data2/han`，只能读）

```
/data2/han/nuplan/archives/nuplan-v1.1/
├── nuplan-maps-v1.0.zip                      # 地图（已解压，无需再动）
├── nuplan-v1.1_train_boston.zip              # 波士顿 DB 包（38G）
├── nuplan-v1.1_train_pittsburgh.zip          # 匹兹堡 DB 包（30G）
├── nuplan-v1.1_train_singapore.zip           # 新加坡 DB 包（35G）
├── nuplan-v1.1_train_vegas_1.zip ~ _6.zip    # 拉斯维加斯 DB 包（6 个，共约 900G）
├── nuplan-v1.1_val.zip                       # 验证集 DB 包（97G）
└── sensor_blobs/
    ├── train_set/    # 完整训练集传感器分卷：43 个 camera zip + 41 个 lidar zip + 1 个 log 清单（lidar 分卷仍在陆续到位，以实际为准）
    │   └── public_set_train_sensor.txt       # 官方 train log 清单（1085 个 log）
    ├── val_set/      # 验证集传感器分卷（24 个 zip = 12 camera + 12 lidar）
    └── mini_set/     # 之前的 mini（不用管）
```

### 0.2 产物目录（在 `/data2/wyc/nuplan_maptrv2`，全部新建）

```
raw/nuplan_full/dbs/          # 解压出的完整 DB（train 13125 + val 1381 ≈ 14506）
raw/nuplan_full/sensor_blobs/ # 解压出的完整图像（约 1.5~3T，按 stride 而定）
configs/splits/full_train_logs.txt  # 完整 train log 名单（1085）
configs/splits/full_val_logs.txt    # 完整 val log 名单（1381）
reports/sensor_blobs_index_full_train.json  # train 分卷索引（当前 70 个，camera 43 个可用）
data/infos/nuplan_map_infos_full_train.pkl   # 完整 train info（nuScenes 对齐格式，顶层 {'infos'}）
data/infos/nuplan_map_infos_full_val.pkl     # 完整 val info（nuScenes 对齐格式）
work_dirs/full/               # 完整训练输出
```

### 0.3 关键事实

- 地图只有 4 个 location（sg-one-north / us-ma-boston / us-nv-las-vegas-strip / us-pa-pittsburgh-hazelwood），已拷贝到 `raw/nuplan_full/maps/maps/`（共 1.4G），完整训练直接用它，不需要重新解压地图。
- 完整训练集约 1085 个 log（mini 是 64 个）。
- 13125 个 DB vs 1085 个 train log：DB 包含全部场景（`dbs/` 约 13125 个 log），但官方只对 benchmark 划定的 1085 个 train log 发布传感器数据（camera/lidar，见 `public_set_train_sensor.txt`）。MapTRV2 训练必须有图像，所以 `full_train_logs.txt` = 1085 是官方 train 子集（正确，不是漏）；其余 ~12000 个无传感器数据的 log 训练用不上，也不需要为它们解压图像。
- MapTRV2 是 camera-only 方法：训练只需要相机图像，lidar 数据不是必需。因此即使 lidar 分卷缺失，也不影响主线。

---

## 1. 前置准备（每次开新终端都要做）

```bash
# 1.1 激活训练环境（Python 3.8 + torch1.9）
# source /home/bicv01/miniforge3/etc/profile.d/conda.sh
conda activate maptr

# 1.2 进入项目根目录
cd /data2/wyc/nuplan_maptrv2

# 1.3 设置环境变量（必须！否则 GKT 找不到 libc10 / nohup 报错）
export PATH=/home/bicv01/miniforge3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/home/bicv01/miniforge3/envs/maptr/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH

# 1.4 确认磁盘空间（/data2 至少要有 3~4T 富余）
df -h /data2/wyc
```

> 验证：`python -c "import torch, mmdet3d; print(torch.__version__, mmdet3d.__version__)"` 能正常输出即 OK。

---

## 2. 步骤 1：解压完整 DB

DB 是训练的前提（关键帧、ego 位姿、相机标定都在 DB 里）。`build_nuplan_infos.py` 用 `db_dir.glob("*.db")` 找 DB，所以必须把 .db 扁平化放到同一个目录。

### 2.1 train DB（已全部解压，无需重复）

当前 `raw/nuplan_full/dbs/` 已有 13125 个 train .db，无需再动。  
如果磁盘空间紧张，可考虑删除那些不在 `full_train_logs.txt` 和 `full_val_logs.txt` 中的多余 DB（约 1.2 万个），但非必须，且操作前务必确认名单，谨慎执行。

### 2.2 val DB（未解压，必须执行）

```bash
cd /data2/wyc/nuplan_maptrv2

# 解压 val DB（约 97G，1381 个 .db），用 nohup 后台执行
nohup unzip -j /data2/han/nuplan/archives/nuplan-v1.1/nuplan-v1.1_val.zip "*.db" -d raw/nuplan_full/dbs \
  > work_dirs/unzip_val.log 2>&1 &

tail -f work_dirs/unzip_val.log
```

> 说明：
> - `-j` 平铺到 `raw/nuplan_full/dbs/` 目录，与 train DB 混放；train/val 名单不重叠，不影响 `build_nuplan_infos.py` 按 `--logs` 过滤。
> - 解压完成后，总 DB 数应为 14506（13125 train + 1381 val）。

检查点：
```bash
ls raw/nuplan_full/dbs/*.db | wc -l   # 输出应约 14506
```

---

## 3. 步骤 2：生成 train/val 的 log 名单（已完成）

```bash
cd /data2/wyc/nuplan_maptrv2
mkdir -p configs/splits

# train 名单：直接用官方清单（过滤掉 "File group" 分隔行）
grep -v "File group" /data2/han/nuplan/archives/nuplan-v1.1/sensor_blobs/train_set/public_set_train_sensor.txt \
  > configs/splits/full_train_logs.txt

# val 名单：⚠️ 不要从混合的 dbs/ 目录 ls！直接从 val.zip 目录提取
unzip -l /data2/han/nuplan/archives/nuplan-v1.1/nuplan-v1.1_val.zip "*.db" \
  | awk '/\.db$/ {n=$4; sub(/^.*\//,"",n); print n}' \
  | sed 's/\.db$//' | sort -u > configs/splits/full_val_logs.txt
```

检查点：
- `wc -l configs/splits/full_train_logs.txt` 应为 1085
- `wc -l configs/splits/full_val_logs.txt` 应为 1381

---

## 4. 步骤 3：扫描 train 传感器分卷，建立索引（已部分完成）

当前已有索引 `reports/sensor_blobs_index_full_train.json`，包含 70 个分卷（43 camera + 27 lidar）。  
MapTRV2 只用到 camera 分卷，当前 43 个 camera 分卷已齐全，完全够用，无需等待 lidar 迁移完成。  
可以直接继续步骤 4。

### 4.1 如果需要补全索引（可选，等 lidar 迁移完成后）

```bash
cd /data2/wyc/nuplan_maptrv2

# 等源目录 lidar 分卷数量稳定后（ls *.zip | wc -l 达 84），可重跑生成完整索引
python tools/scan_sensor_blobs.py \
  --blobs-root /data2/han/nuplan/archives/nuplan-v1.1/sensor_blobs/train_set \
  --db-dir raw/nuplan_full/dbs \
  --out reports/sensor_blobs_index_full_train.json \
  --jobs 8 \
  --skip-db-check
```

> 说明：
> - `--skip-db-check`（full 数据集务必加）：`dbs/` 里 13125 个 DB 含大量未发布传感器数据的 log，DB 引用校验会报海量 `[MISSING]`，且可能因损坏 DB 崩溃。跳过校验只生成归档索引，最快最稳。
> - 索引会先落盘再校验：即使中途失败，`reports/sensor_blobs_index_full_train.json` 也已有 `archives`。
> - 若只训练 MapTRV2，无需等待 lidar；现有索引已可支持步骤 4 的相机图像提取。

### 4.2 生成 val 索引（需要执行，因为步骤 4 会用到 val 图像）

```bash
cd /data2/wyc/nuplan_maptrv2

python tools/scan_sensor_blobs.py \
  --blobs-root /data2/han/nuplan/archives/nuplan-v1.1/sensor_blobs/val_set \
  --db-dir raw/nuplan_full/dbs \
  --out reports/sensor_blobs_index_full_val.json \
  --jobs 8 \
  --skip-db-check
```

检查点：
- `reports/sensor_blobs_index_full_train.json` 已存在（含至少 43 个 camera 分卷）
- `reports/sensor_blobs_index_full_val.json` 生成成功

---

## 5. 步骤 4：解压相机图像（train + val）

这是最占空间、最耗时的一步。完整 1085 个 train log + 1381 个 val log × 8 路相机图像约 1.5~3T。

### 5.1 train 图像

```bash
cd /data2/wyc/nuplan_maptrv2

nohup python tools/extract_sensor_images.py \
  --blobs-root /data2/han/nuplan/archives/nuplan-v1.1/sensor_blobs/train_set \
  --index reports/sensor_blobs_index_full_train.json \
  --db-dir raw/nuplan_full/dbs \
  --logs configs/splits/full_train_logs.txt \
  --out raw/nuplan_full/sensor_blobs \
  --frame-stride 10 \
  --skip-existing \
  > work_dirs/extract_full_train.log 2>&1 &

tail -f work_dirs/extract_full_train.log
```

### 5.2 val 图像

```bash
cd /data2/wyc/nuplan_maptrv2

nohup python tools/extract_sensor_images.py \
  --blobs-root /data2/han/nuplan/archives/nuplan-v1.1/sensor_blobs/val_set \
  --index reports/sensor_blobs_index_full_val.json \
  --db-dir raw/nuplan_full/dbs \
  --logs configs/splits/full_val_logs.txt \
  --out raw/nuplan_full/sensor_blobs \
  --frame-stride 10 \
  --skip-existing \
  > work_dirs/extract_full_val.log 2>&1 &

tail -f work_dirs/extract_full_val.log
```

> 参数说明：
> - `--frame-stride 10`：每 10 帧解压 1 帧（对应 1Hz 采样），可大幅省空间。想用 2Hz 就改成 5（对应 info 的 `--stride 5`）。此值必须和步骤 5 的 `--stride` 保持一致。
> - `--skip-existing`：断点续传，已解压的跳过，重复执行安全。
> - 注意：val 图像与 train 图像会解压到同一个 `raw/nuplan_full/sensor_blobs`，但数据目录结构按 log 名区分，不会冲突。

检查点：`tail -f work_dirs/extract_full_train.log` 和 `tail -f work_dirs/extract_full_val.log` 看进度；`find raw/nuplan_full/sensor_blobs -name "*.jpg" | wc -l` 应持续增长。

---

## 6. 步骤 5：生成 info pkl（核心转换）—— 对齐 nuScenes 格式

> 目标（2026-08-24 起）：让 nuPlan 生成的 info 对齐 nuScenes 在 MapTR 中使用的格式（参考 `docs/09_ref.md`），即与官方 `tools/maptrv2/custom_nusc_map_converter.py` 产出的 `nuscenes_map_infos_temporal_*.pkl` 字段一致。这样可直接用官方 `CustomNuScenesOfflineLocalMapDataset` 加载（`mmdet3d` 的 `load_annotations` 读顶层 `data['infos']`），无需再依赖自定义 `NuPlanMapDataset`。
>
> ⚠️ 重要：本步骤必须用修复后的 `tools/nuplan_maptrv2/nuplan_map.py`（2026-08-17 已修复 boundary 闭合环 bug，详见 `06_validation_report.md` 2.4 节）。且 `build_nuplan_infos.py` 需要 nuplan-devkit 环境（maptr 环境缺 pyproj/nuplan 模块）：
> ```bash
> export PYTHONPATH=/data2/30033/nuplan-devkit
> /data2/30033/nuplan-devkit/miniconda3/envs/nuplan/bin/python tools/build_nuplan_infos.py --format nuscenes ...
> ```

### 6.0 输出格式（nuScenes 对齐）

`build_nuplan_infos.py --format nuscenes`（默认）输出：

```python
{
  "infos": [                    # 顶层必须是 infos（官方 load_annotations 读 data['infos']）
    {
      "lidar_path": str,        # MapTRV2 不用 lidar，占位即可
      "token": str,             # 唯一
      "prev"/"next": str,       # 同 log 内前后帧 token（脚本自动连接）
      "can_bus": np.zeros(18),  # get_data_info 只覆写 [:7] 与 [-2:]
      "frame_idx": int,
      "sweeps": [],             # 空
      "map_location": str,      # nuPlan map_version（如 sg-one-north）
      "scene_token": str,       # logfile
      "lidar2ego_translation": [0,0,0],      # lidar 视为与 ego 重合
      "lidar2ego_rotation": [1,0,0,0],       # wxyz
      "ego2global_translation": [x,y,z],     # 3
      "ego2global_rotation": [w,x,y,z],      # wxyz 四元数（nuPlan qw,qx,qy,qz 即 wxyz）
      "timestamp": int,
      "cams": {                              # 每路相机（键名已对齐 nuScenes）
        "CAM_FRONT": {                       # nuPlan CAM_F0 -> CAM_FRONT（映射见下方）
          "data_path": str,                 # 解压图像路径（仍指向 nuPlan 原始图像）
          "type": "camera",
          "sample_data_token": str,
          "sensor2ego_translation": [3],    # 相机在 ego 坐标
          "sensor2ego_rotation": [w,x,y,z],
          "ego2global_translation": [3],    # 相机在全局
          "ego2global_rotation": [w,x,y,z],
          "sensor2lidar_rotation": 3x3,     # == sensor2ego（lidar=ego）
          "sensor2lidar_translation": [3],
          "timestamp": int,
          "cam_intrinsic": 3x3,
        }, ...
      },
      "annotation": {                        # 与官方 VectorizedLocalMap 直接兼容
        "divider": [np.ndarray(N,2)], "ped_crossing": [...], "boundary": [...]
      },
    }, ...
  ]
}
```

关键对齐点（对照官方 `custom_nusc_map_converter.py` / `nuscenes_offlinemap_dataset.py`）：- **相机 6 路（本方案）**：用 `--channels CAM_F0 CAM_B0 CAM_L0 CAM_R0 CAM_L2 CAM_R2` 生成，info 只含这 6 路（对应 nuScenes 六视角），与配置 `num_cams=6` 一致。- 四元数顺序 wxyz（nuPlan `qw,qx,qy,qz` 即 wxyz，直接映射；相机旋转用 `rotmat_to_quat` 转换）。
- `lidar2ego` = 单位变换、`sensor2lidar` = `sensor2ego`：MapTRV2 是 camera-only，lidar 与 ego 重合。
- `can_bus` 用 `zeros(18)`（官方 server scene 的 fallback 就是 18 维；`get_data_info` 运行时会覆写 `[:7]` 与 `[-2:]`）。
- `annotation` 保持 `{divider, ped_crossing, boundary}`——官方 `VectorizedLocalMap.gen_vectorized_samples` 就是按这 3 个键读。

### 6.1 train info

```bash
cd /data2/wyc/nuplan_maptrv2

export PYTHONPATH=/data2/30033/nuplan-devkit
nohup /data2/30033/nuplan-devkit/miniconda3/envs/nuplan/bin/python tools/build_nuplan_infos.py \
  --db-dir raw/nuplan_full/dbs \
  --map-root raw/nuplan_full/maps/maps \
  --logs configs/splits/full_train_logs.txt \
  --out data/infos/nuplan_map_infos_full_train.pkl \
  --pc-range -15 -30 -10 15 30 10 \
  --stride 10 \
  --img-root raw/nuplan_full/sensor_blobs \
  --format nuscenes \
  --channels CAM_F0 CAM_B0 CAM_L0 CAM_R0 CAM_L2 CAM_R2 \
  > work_dirs/build_full_train.log 2>&1 &

tail -f work_dirs/build_full_train.log
```

### 6.2 val info（先确认 val 索引已生成、val DB 已解压）

> ⚠️ 别忘了 val 前置（对应之前 `FileNotFoundError: reports/sensor_blobs_index_full_val.json`）：
> 1. val 索引：确认 `reports/sensor_blobs_index_full_val.json` 已生成（见步骤 3 的 4.2）；若还没有，先跑 4.2 的扫描命令。
> 2. val DB：`raw/nuplan_full/dbs/` 需包含 val 的 1381 个 .db（见步骤 1 补充命令），解压后 `ls raw/nuplan_full/dbs/*.db | wc -l` 应约 14506。

```bash
cd /data2/wyc/nuplan_maptrv2

export PYTHONPATH=/data2/30033/nuplan-devkit
nohup /data2/30033/nuplan-devkit/miniconda3/envs/nuplan/bin/python tools/build_nuplan_infos.py \
  --db-dir raw/nuplan_full/dbs \
  --map-root raw/nuplan_full/maps/maps \
  --logs configs/splits/full_val_logs.txt \
  --out data/infos/nuplan_map_infos_full_val.pkl \
  --pc-range -15 -30 -10 15 30 10 \
  --stride 10 \
  --img-root raw/nuplan_full/sensor_blobs \
  --format nuscenes \
  --channels CAM_F0 CAM_B0 CAM_L0 CAM_R0 CAM_L2 CAM_R2 \
  > work_dirs/build_full_val.log 2>&1 &

tail -f work_dirs/build_full_val.log
```

> 说明：
> - `--format nuscenes` 是默认值（也可显式写）；旧 AV2 风格用 `--format av2`（仅供旧 `NuPlanMapDataset` 流程，不再推荐）。
> - **`--channels`（6 路方案）**：传 nuPlan 通道名 `CAM_F0 CAM_B0 CAM_L0 CAM_R0 CAM_L2 CAM_R2`；脚本会自动映射为 nuScenes 六视角命名（`CAM_FRONT` 等）并按官方顺序写入 `info['cams']`。info 只含这 6 路，**必须与配置里 `num_cams=6` 一致**；若不加该参数，info 会含全部 8 路（未映射部分保留原 nuPlan 名），与 `num_cams=6` 不匹配。
> - `--stride 10` 必须与步骤 4 的 `--frame-stride` 一致。
> - 这一步读 DB + 地图 + 图像路径，生成 info，1085 个 train log 和 1381 个 val log 可能要跑数小时（日志会逐个 log 打印 `samples so far=...`）。
> - 如果只想先小规模试跑，可加 `--limit 200`（每个 log 最多 200 样本）或用部分 log 名单。

检查点：
- `tail -f work_dirs/build_full_train.log` 最后一行出现 `saved N samples ...`；`tail -f work_dirs/build_full_val.log` 同理。
- 快速验证结构（应是 nuScenes 格式）：`python -c "import pickle; d=pickle.load(open('data/infos/nuplan_map_infos_full_train.pkl','rb')); print(list(d.keys()), len(d['infos']))"` 输出 `['infos'] <数量>`。

### 6.3 把已生成的 8 路 info 修正为 6 路（推荐，秒级完成，无需重跑）

> 之前生成的 `nuplan_map_infos_full_{train,val}.pkl` 未带 `--channels`，是 **8 路**（键为 `CAM_F0` 等），与配置 `num_cams=6` 不匹配。**无需重新 build**：`tools/reduce_infos_cams.py` 做纯后处理（保留 6 路 + 键名映射 nuScenes + 官方顺序重排），结果与用 `--channels` 重新 build **完全等价**（build 对每路相机独立生成，与是否生成其他路无关）。
>
> train 和 val **可以同时跑**（两个独立进程，读不同文件、写不同文件，互不干扰）。

```bash
cd /data2/wyc/nuplan_maptrv2
export PYTHONPATH=/data2/30033/nuplan-devkit
PY=/data2/30033/nuplan-devkit/miniconda3/envs/nuplan/bin/python

# train + val 并行修正（默认覆盖原文件；可用 --out 指定新文件）
nohup "$PY" tools/reduce_infos_cams.py --infos data/infos/nuplan_map_infos_full_train.pkl \
  > work_dirs/reduce_train_6cam.log 2>&1 &
nohup "$PY" tools/reduce_infos_cams.py --infos data/infos/nuplan_map_infos_full_val.pkl \
  > work_dirs/reduce_val_6cam.log 2>&1 &
wait
cat work_dirs/reduce_train_6cam.log work_dirs/reduce_val_6cam.log
```

> 只有在想**彻底重建**时才用 6 路 `--channels` 重新跑 6.1/6.2 的 build 命令（慢，数小时；不推荐）。

修正完成后校验（应报每样本 **6 路**相机、nuScenes 命名）：

```bash
python tools/validate_nuplan_infos.py \
  --infos data/infos/nuplan_map_infos_full_train.pkl \
  --pc-range -15 -30 -10 15 30 10
```

---

## 7. 步骤 6：校验 info

```bash
cd /data2/wyc/nuplan_maptrv2

python tools/validate_nuplan_infos.py \
  --infos data/infos/nuplan_map_infos_full_train.pkl \
  --pc-range -15 -30 -10 15 30 10
```

> 校验项：**6 路相机（nuScenes 命名）**是否齐全、图像路径是否存在、类别统计、token 唯一。有报错按提示排查（通常是某 log 图像没解压全）。

---

## 8. 步骤 7：改训练配置并启动

### 8.1 复制配置

```bash
cd /data2/wyc/nuplan_maptrv2/MapTRV2
cp projects/configs/maptrv2/maptrv2_nuplan_mini.py projects/configs/maptrv2/maptrv2_nuplan_full.py
```

### 8.2 修改配置里这几处（info 已对齐 nuScenes，改用官方 dataset）

打开 `maptrv2_nuplan_full.py`，改：

```python
# 关键：info 已是 nuScenes 风格（顶层 {'infos'}），直接用官方 dataset，不再用自定义 NuPlanMapDataset
dataset_type = 'CustomNuScenesOfflineLocalMapDataset'   # 官方类，load_annotations 读 data['infos']

map_classes = ['divider', 'ped_crossing', 'boundary']   # 3 类（与 info['annotation'] 键一致）

# 相机方案（已选定方案 2：6 路，对应 nuScenes 六视角）
# 模型 transformer 里：
num_cams = 6     # 与 info 的 6 路（--channels CAM_F0 CAM_B0 CAM_L0 CAM_R0 CAM_L2 CAM_R2）一致

data_root = '/data2/wyc/nuplan_maptrv2/data/infos/'
# train 的 ann_file 改成完整 info
ann_file=data_root + 'nuplan_map_infos_full_train.pkl',
# val 的 ann_file 与 map_ann_file 改成完整 val
ann_file=data_root + 'nuplan_map_infos_full_val.pkl',
map_ann_file=data_root + 'nuplan_map_anns_full_val.json',   # 评测 GT，官方 _format_gt 会自动生成；仅训练不评测可暂时注释
# 训练轮数（可选，完整数据建议加到 24）
total_epochs = 24
```

> 相机数量（已选**方案 2：6 路**）：nuPlan 共 8 路相机，本方案只保留 **6 路**（`CAM_F0 CAM_B0 CAM_L0 CAM_R0 CAM_L2 CAM_R2`），`build_nuplan_infos.py` 会自动把 `info['cams']` 键名映射为 nuScenes 六视角命名（`CAM_FRONT` 等，含官方顺序），模型 `MapTRPerceptionTransformer.num_cams` 保持 **6**。
> ⚠️ **一致性要求**：info 必须用步骤 5 的 `--channels`（6 路）生成，使 info 只含这 6 路；若 info 是 8 路而 `num_cams=6` 会维度不匹配报错。当前若已用默认 8 路生成过 info，需用 6 路 `--channels` 重新生成。
>
> 关于 `map_ann_file`：官方 `CustomNuScenesOfflineLocalMapDataset._format_gt` 会在评测时基于 `info['annotation']` 自动生成 GT json，无需手动造（旧 `NuPlanMapDataset` 的生成脚本不再需要）。仅训练不评测时用 `--no-validate` 即可。

> 超参调整建议：完整数据量是 mini 的约 17 倍，直接沿用 mini 的学习率、warmup 可能不稳定。建议先小规模试跑（如 1~2 个 epoch），观察 loss 是否正常；必要时调整 `lr`、`warmup_iters`。

### 8.3 启动 8 卡训练

> 环境：用 **`/home/xiaoxuan/miniconda3/envs/maptr`**（已修复为兼容组合：**numpy 1.22.4 + numba 0.48.0 + llvmlite 0.31.0**，匹配 mmdet3d 0.17.2；版本问题见常见坑）。
> ⚠️ **必须设 `PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2`**，否则 worker 报 `No module named 'projects'`。

**方式一（推荐）：一键脚本**

```bash
bash /data2/wyc/nuplan_maptrv2/MapTRV2/run_full_train.sh
```

**方式二：完整命令（export 与 nohup 一起复制，缺一不可）**

```bash
cd /data2/wyc/nuplan_maptrv2/MapTRV2
export PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2
export LD_LIBRARY_PATH=/home/xiaoxuan/miniconda3/envs/maptr/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup /home/xiaoxuan/miniconda3/envs/maptr/bin/python -m torch.distributed.launch \
  --nproc_per_node=8 --master_port=28509 \
  tools/train.py projects/configs/maptrv2/maptrv2_nuplan_full.py \
  --launcher pytorch \
  --work-dir /data2/wyc/nuplan_maptrv2/work_dirs/full \
  --no-validate \
  > /data2/wyc/nuplan_maptrv2/work_dirs/full/train_full.log 2>&1 &
```

> 提示：**不要** `export PATH=/home/xiaoxuan/miniconda3/bin:...`（会把 base 的 python 放最前，之后若敲 `python` 会命中 base 而报 `No module named 'torch'`）；命令里用 maptr 的绝对路径 python 即可。

检查点：`tail -f work_dirs/full/train_full.log`，等出现 `Epoch [1][...]` 的 loss 即正常。

---

## 9. 分阶段建议（新手强烈建议先这样做）

完整数据 TB 级、步骤 4/5 耗时很长，不要一上来就全量：

1. 先用 1 个 location（如 boston 或 singapore）跑通 2~7 步，确认无报错。
2. 再逐步补上其余 location（vegas 最大放最后）。
3. 每个 location 的 log 名单单独维护（如 `full_train_boston.txt`），方便分批生成 info、合并。

---

## 10. 常见坑速查

| 现象 | 原因 / 解决 |
|---|---|
| `nohup: failed to run command 'env': Input/output error` | PATH 里有坏盘 `/data` 路径；用第 1 节的干净 PATH + 绝对路径 python |
| `ImportError: libc10.so` | 没设 `LD_LIBRARY_PATH`；按第 1 节设置 |
| `FormatCode() got unexpected keyword 'verify'` | yapf 版本过新；装 `yapf==0.31.0` |
| `AssertionError`（shuffle 时） | dataset 缺 `flag`；可视化脚本已内置处理，训练不影响 |
| info 里图像路径不存在 | 步骤 4 解压不全或 `--frame-stride` 与 `--stride` 不一致 |
| 磁盘写满 | 图像/DB 巨大；分 location 做，及时清理中间 zip |
| `dbs/` 中 DB 损坏导致脚本崩溃 | 使用 `--skip-db-check` 跳过 DB 校验 |
| 扫描索引时 tar 格式报错 | sensor blobs 实际是 tar 格式，脚本已支持自动识别；确保使用最新脚本 |
| 训练 loss 异常 | 完整数据量下学习率/warmup 需要调整；先小规模试跑 |
| 训练启动报 `AttributeError: module 'numpy' has no attribute 'long'/'int'` | numpy 1.24 太新，与旧 numba/networkx 不兼容；统一装 **`numpy==1.22.4`**（`np.long/np.int/np.float` 在 1.22 仍可用） |
| 训练启动报 `ModuleNotFoundError: No module named 'numba.errors'` | numba 太新（0.57）与 mmdet3d 0.17.2 不兼容；降回 **`numba==0.48.0`**（配 `llvmlite==0.31.0`） |
| 用官方 `CustomNuScenesOfflineLocalMapDataset` 报 KeyError（缺 `ego2global_*`/`can_bus`/`lidar2ego_*` 等） | info 不是 nuScenes 格式；用 `--format nuscenes` 重新生成（旧 AV2 风格只配 `NuPlanMapDataset`） |
| `FileNotFoundError: reports/sensor_blobs_index_full_val.json` | val 解压前漏跑 val 索引扫描；执行步骤 3 的 4.2 命令生成 |

---

## 11. 规模与耗时估算（参考）

| 步骤 | 数据量 | 预计耗时 |
|---|---|---|
| 解压 DB（train+val） | ~250G+ | 数小时（vegas 更长） |
| 解压图像（stride=10） | ~1.5T | 可能 1~2 天 |
| 生成 info | pkl 可能数 GB | 数小时 |
| 训练（24 epoch） | 取决于样本量 | 数天（8 卡） |

> 如果时间/磁盘有限，优先用更大 `--stride`（如 20 = 0.5Hz）或先训 1~2 个 location。