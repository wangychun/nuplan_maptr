# 云服务器上评测 MapTRV2 训练模型（nuPlan / nuScenes / NVIDIA）

> 目的：用**当前训练中的 checkpoint**（`work_dirs/full/latest.pth`）在**别的云服务器**上，对
> nuPlan、nuScenes、NVIDIA 三个数据集各抽**少量样本**做评测（chamfer 指标）与可视化。
> 数据在本机 `/data2/wyc`（与 `/data2/nuscenes`），云服务器通过 **NFS/共享盘**直接读写。
> 本教程假设云服务器能挂载 `/data2/wyc` 与 `/data2/nuscenes`；若只能挂 `/data2/wyc`，
> 请把 `/data2/nuscenes` 也同步挂载，或用 2.2 的小样本方案把 nuScenes 数据拷到 `/data2/wyc` 下。

---

## 0. 数据与产物路径速查（都在本机，云服务器直接读）

| 项 | 路径 |
|---|---|
| MapTRV2 代码 | `/data2/wyc/nuplan_maptrv2/MapTRV2` |
| 自定义工具脚本 | `/data2/wyc/nuplan_maptrv2/tools/` |
| 训练 checkpoint | `/data2/wyc/nuplan_maptrv2/work_dirs/full/latest.pth`（→ epoch_N.pth） |
| nuPlan 全量 val info | `/data2/wyc/nuplan_maptrv2/data/infos/nuplan_map_infos_full_val.pkl`（67 万样本） |
| nuPlan 训练配置 | `/data2/wyc/nuplan_maptrv2/MapTRV2/projects/configs/maptrv2/maptrv2_nuplan_full.py` |
| nuScenes 原始数据 | `/data2/nuscenes/`（samples/sweeps/v1.0-trainval/maps） |
| nuScenes 现成 info（官方生成） | `/data2/nuscenes/nuscenes_map_infos_temporal_val.pkl` |
| nuScenes 评测 GT | `/data2/nuscenes/nuscenes_map_anns_val.json` |
| NVIDIA 原始数据 | `/data2/wyc/nuplan_maptrv2/MapTRV2/data/nvidia/raw1/pai_subset/`（7 路相机 mp4 + 标定 + egomotion） |

模型说明：当前 checkpoint 是 **nuPlan 训练的 MapTRV2**：6 相机、3 类地图
（divider / ped_crossing / boundary）、num_vec=50。因为 info 已对齐 nuScenes 格式，
它可以直接加载 nuScenes（同样 6 相机、3 类）数据；NVIDIA 数据转换后也可推理（无地图 GT，只能可视化）。

---

## 1. 云服务器环境准备（一次性）

云服务器需要能跑 MapTRV2 的 Python 环境（torch + mmcv-full + mmdet3d + 本项目插件）。
参考本机 maptr 环境（Python 3.8 / torch 1.9.1 / mmcv-full 1.4.0 / mmdet3d 0.17.2，numpy 1.22.4 + numba 0.48.0）。

```bash
# 在云服务器上创建环境（版本与本机一致最稳）
conda create -n maptr python=3.8 -y
conda activate maptr
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0 mmsegmentation==0.14.1 mmdet3d==0.17.2
pip install "numpy==1.22.4" "numba==0.48.0" llvmlite==0.31.0
pip install pyquaternion shapely nuscenes-devkit opencv-python pillow matplotlib pandas pyarrow
```

> 若云服务器环境与本机不同（如新 torch），只要 `import mmdet3d, projects.mmdet3d_plugin` 不报错即可。
> 验证（在 MapTRV2 目录下）：
> ```bash
> cd /data2/wyc/nuplan_maptrv2/MapTRV2 && PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2 python -c "import mmdet3d, projects.mmdet3d_plugin; print('OK')"
> ```

**运行时的固定环境变量**（每条命令前都要有，或写成脚本）：
```bash
export PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2
export CUDA_VISIBLE_DEVICES=0
```

---

## 2. 准备评测数据（三个数据集）

### 2.1 nuPlan（前置：确保 val 图像已解压）

> ⚠️ **val 图像必须已解压**。之前 3.1 报 `FileNotFoundError: .../raw/nuplan_full/sensor_blobs/2021.06.07.../CAM_F0/xxx.jpg`
> 就是因为 **val 的 1381 个 log 图像从未解压**（train 解压过、val 没跑）。
> 检查：`ls raw/nuplan_full/sensor_blobs/ | wc -l` 应约 **2466**（train 1085 + val 1381）；如果只有 1085，说明 val 未解压。

**在本机后台启动 val 图像解压**（数据在 `/data2/wyc`，云服务器 NFS 直接读到；断线不影响）：

```bash
cd /data2/wyc/nuplan_maptrv2
nohup /data2/30033/nuplan-devkit/miniconda3/envs/nuplan/bin/python tools/extract_sensor_images.py \
  --blobs-root /data2/han/nuplan/archives/nuplan-v1.1/sensor_blobs/val_set \
  --index reports/sensor_blobs_index_full_val.json \
  --db-dir raw/nuplan_full/dbs \
  --logs configs/splits/full_val_logs.txt \
  --out raw/nuplan_full/sensor_blobs \
  --frame-stride 10 \
  --skip-existing \
  > work_dirs/extract_full_val.log 2>&1 &
```

- 进度：`tail -f work_dirs/extract_full_val.log`，看到 `done: extracted N, skipped 0 -> raw/nuplan_full/sensor_blobs` 即完成。
- 验证：`ls raw/nuplan_full/sensor_blobs/ | wc -l` 到约 2466。
- 完成后再到云服务器跑第 3 节 nuPlan 评测。

直接用全量 val info，评测时用 `--max-samples` 抽少量即可，见第 3 节。

### 2.1b 若 val 传感器分卷不完整（只覆盖部分 log）→ 用子集方案

> 当前 val_set 只有 24 个分卷，覆盖 **225/1381** 个 val log，其余 log **无图像**，
> 全量解压/全量 eval 会报 `no archive for <log>/CAM_*`。只需要少量样本时用子集方案：

```bash
# 1) 生成“有图像覆盖”的 eval 子集 info + log 名单（秒级，本机或云服务器均可）
PY=/data2/30033/nuplan-devkit/miniconda3/envs/nuplan/bin/python
"$PY" /data2/wyc/nuplan_maptrv2/tools/make_val_eval_subset.py \
  --infos /data2/wyc/nuplan_maptrv2/data/infos/nuplan_map_infos_full_val.pkl \
  --index /data2/wyc/nuplan_maptrv2/reports/sensor_blobs_index_full_val.json \
  --max-samples 8 \
  --out-info /data2/wyc/nuplan_maptrv2/data/infos/nuplan_val_eval_sub.pkl \
  --out-logs /data2/wyc/nuplan_maptrv2/data/infos/nuplan_val_eval_logs.txt

# 2) 解压名单里的 log 图像（本机后台；名单只有 1~2 个 log，很快）
cd /data2/wyc/nuplan_maptrv2
nohup /data2/30033/nuplan-devkit/miniconda3/envs/nuplan/bin/python tools/extract_sensor_images.py \
  --blobs-root /data2/han/nuplan/archives/nuplan-v1.1/sensor_blobs/val_set \
  --index reports/sensor_blobs_index_full_val.json \
  --db-dir raw/nuplan_full/dbs \
  --logs data/infos/nuplan_val_eval_logs.txt \
  --out raw/nuplan_full/sensor_blobs \
  --frame-stride 10 --skip-existing \
  > work_dirs/extract_full_val_sub.log 2>&1 &

# 3) 云服务器用子集 info 评测（map-ann 传一个新路径，脚本会自动生成对应 GT）
python tools/test_evaluate.py projects/configs/maptrv2/maptrv2_nuplan_full.py \
  /data2/wyc/nuplan_maptrv2/work_dirs/full/latest.pth \
  --ann-file /data2/wyc/nuplan_maptrv2/data/infos/nuplan_val_eval_sub.pkl \
  --map-ann /data2/wyc/nuplan_maptrv2/data/infos/nuplan_map_anns_full_val_sub.json \
  --max-samples 8
```

> ✅ **本机已生成**：`data/infos/nuplan_val_eval_sub.pkl`（8 样本，涉及 log `2021.06.07.11.59.52_veh-35_02283_02464`）与 `data/infos/nuplan_val_eval_logs.txt`。只需跑第 2 步解压即可。

### 2.2 nuScenes：修正 info 路径 + 抽小子集

> ✅ **本机已生成**：`/data2/wyc/nuplan_maptrv2/data/infos/nuscenes_map_infos_eval_sub.pkl`（**8 个样本**，data_path 已绝对化）。云服务器直接用它即可；如需改样本数再执行下面命令。

nuScenes 现成 info 里的 `data_path` 是相对路径 `./data/nuscenes/samples/...`（在别处生成时留下的），
需要改成绝对路径 `/data2/nuscenes/samples/...`，并抽一个小子集，方便云服务器只读少量样本。

```bash
# 在本机执行一次（生成小子集 info 到 /data2/wyc 下，云服务器直接读）
PY=/data2/30033/nuplan-devkit/miniconda3/envs/nuplan/bin/python
"$PY" /data2/wyc/nuplan_maptrv2/tools/fix_nuscenes_infos_path.py \
  --infos /data2/nuscenes/nuscenes_map_infos_temporal_val.pkl \
  --nuscenes-root /data2/nuscenes \
  --max-samples 8 \
  --out /data2/wyc/nuplan_maptrv2/data/infos/nuscenes_map_infos_eval_sub.pkl
```

> 说明：脚本会把 `./data/nuscenes/` 前缀替换成 `/data2/nuscenes/`，只保留前 8 个样本。
> 若云服务器**只挂载了 /data2/wyc 没挂 /data2/nuscenes**，请改用 `--copy-images-to` 参数，
> 把这 8 个样本的图像拷贝到 `/data2/wyc/.../eval_nuscenes_sub/` 下并改写成指向该目录的绝对路径。

### 2.3 NVIDIA：抽帧 + 生成 info（只能可视化，无地图 GT）

> ✅ **本机已生成**：`/data2/wyc/nuplan_maptrv2/data/infos/nvidia_map_infos_eval_sub.pkl`（3 帧）及图像 `data/infos/nvidia_imgs/`。云服务器直接用它即可。

NVIDIA 数据（`pai_subset`）只有相机 mp4 + 标定 + egomotion，**没有矢量地图标注**，
所以**无法算 chamfer 指标**，只能做预测可视化。脚本会从 1 个 clip 抽几帧、构造 nuScenes 风格 info。

```bash
# 在本机执行一次（抽帧 + 生成 info；输出图像与 info 都在 /data2/wyc 下）
PY=/data2/30033/nuplan-devkit/miniconda3/envs/nuplan/bin/python
"$PY" /data2/wyc/nuplan_maptrv2/tools/build_nvidia_infos.py \
  --data-root /data2/wyc/nuplan_maptrv2/MapTRV2/data/nvidia/raw1/pai_subset \
  --clip-id 25cd4769-5dcf-4b53-a351-bf2c5deb6124 \
  --n-frames 3 \
  --out-imgs /data2/wyc/nuplan_maptrv2/data/infos/nvidia_imgs \
  --out-info /data2/wyc/nuplan_maptrv2/data/infos/nvidia_map_infos_eval_sub.pkl
```

> 会生成 3 帧 × 6 路 = 18 张 jpg 到 `nvidia_imgs/`，info 的 `data_path` 指向它们（绝对路径）。

---

## 3. 评测：chamfer 指标（nuPlan 与 nuScenes）

用 `tools/test_evaluate.py`（支持 `--max-samples` 抽几个、`--ann-file` 换数据、`--map-ann` 指定 GT）。
该脚本会用配置里的 dataset（`CustomNuScenesOfflineLocalMapDataset`）加载 info，跑推理 + 官方 chamfer 评测，输出 mAP/AP。

### 3.1 nuPlan（抽 8 个样本）

```bash
cd /data2/wyc/nuplan_maptrv2/MapTRV2
export PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2
export CUDA_VISIBLE_DEVICES=0

python /data2/wyc/nuplan_maptrv2/tools/test_evaluate.py \
  projects/configs/maptrv2/maptrv2_nuplan_full.py \
  /data2/wyc/nuplan_maptrv2/work_dirs/full/latest.pth \
  --ann-file /data2/wyc/nuplan_maptrv2/data/infos/nuplan_map_infos_full_val.pkl \
  --map-ann /data2/wyc/nuplan_maptrv2/data/infos/nuplan_map_anns_full_val.json \
  --max-samples 8

python /data2/wyc/nuplan_maptrv2/tools/test_evaluate.py \
  projects/configs/maptrv2/maptrv2_nuplan_full.py \
  /data2/wyc/nuplan_maptrv2/work_dirs/full/latest.pth \
  --ann-file /data2/wyc/nuplan_maptrv2/data/infos/nuplan_val_eval_sub.pkl \
  --map-ann /data2/wyc/nuplan_maptrv2/data/infos/nuplan_map_anns_full_val_sub.json \
  --max-samples 8
```

> 第一次跑会生成 `nuplan_map_anns_full_val.json`（GT，脚本自动从 info 生成），稍等即可。
> 输出示例：`NuscMap_chamfer/mAP: 0.xxxx, NuscMap_chamfer/divider_AP: ...` 等。

### 3.2 nuScenes（抽 8 个样本）

```bash
cd /data2/wyc/nuplan_maptrv2/MapTRV2
export PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2
export CUDA_VISIBLE_DEVICES=0

python /data2/wyc/nuplan_maptrv2/tools/test_evaluate.py \
  projects/configs/maptrv2/maptrv2_nuplan_full.py \
  /data2/wyc/nuplan_maptrv2/work_dirs/full/latest.pth \
  --ann-file /data2/wyc/nuplan_maptrv2/data/infos/nuscenes_map_infos_eval_sub.pkl \
  --map-ann /data2/nuscenes/nuscenes_map_anns_val.json \
  --max-samples 8
```

> 若 `nuscenes_map_infos_eval_sub.pkl` 未生成，先跑 2.2。

---

## 4. 可视化（三个数据集）

### 4.1 nuPlan：BEV 对比图（GT vs 预测）+ 相机叠加图（用 val 子集）

> ⚠️ **用 `nuplan_val_eval_sub.pkl`（8 个有图像覆盖的样本）而不是全量 val info**，否则可能抽到无图像的 log 报 `FileNotFoundError`（如 `.../2021.06.07.11.59.52_veh-35_00008_00083/CAM_F0/...jpg`）。
> 前置：先解压子集 log 图像（见 2.1b 第 2 步，`--logs data/infos/nuplan_val_eval_logs.txt`）。

```bash
cd /data2/wyc/nuplan_maptrv2/MapTRV2
export PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2
export CUDA_VISIBLE_DEVICES=0

# BEV 对比图（GT 红/预测绿），输出到 reports/pred_vis_full
python /data2/wyc/nuplan_maptrv2/tools/visualize_nuplan_pred.py \
  projects/configs/maptrv2/maptrv2_nuplan_full.py \
  /data2/wyc/nuplan_maptrv2/work_dirs/full/latest.pth \
  --show-dir /data2/wyc/nuplan_maptrv2/reports/pred_vis_full \
  --num-samples 6 --score-thresh 0.3 \
  --ann-file /data2/wyc/nuplan_maptrv2/data/infos/nuplan_val_eval_sub.pkl

# 相机叠加图（预测折线画在相机图上）
python /data2/wyc/nuplan_maptrv2/tools/visualize_nuplan_pred_cam.py \
  projects/configs/maptrv2/maptrv2_nuplan_full.py \
  /data2/wyc/nuplan_maptrv2/work_dirs/full/latest.pth \
  --show-dir /data2/wyc/nuplan_maptrv2/reports/pred_cam_full \
  --num-samples 4 --cam CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT,CAM_BACK \
  --ann-file /data2/wyc/nuplan_maptrv2/data/infos/nuplan_val_eval_sub.pkl
```

> 注意：`--cam` 用 nuScenes 命名（CAM_FRONT 等，nuPlan info 的相机名已映射为这些）；`--ann-file` 必须传子集（避免无图像 log）。
> 输出 PNG 在 `--show-dir` 下，云服务器上可直接打开或下载。

### 4.2 nuScenes：BEV 预测可视化

```bash
cd /data2/wyc/nuplan_maptrv2/MapTRV2
export PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2
export CUDA_VISIBLE_DEVICES=0

# 用 nuscenes 小子集 info 做预测可视化（GT vs 预测，BEV）
python /data2/wyc/nuplan_maptrv2/tools/visualize_nuplan_pred.py \
  projects/configs/maptrv2/maptrv2_nuplan_full.py \
  /data2/wyc/nuplan_maptrv2/work_dirs/full/latest.pth \
  --show-dir /data2/wyc/nuplan_maptrv2/reports/pred_vis_nusc \
  --num-samples 6 --score-thresh 0.3 \
  --ann-file /data2/wyc/nuplan_maptrv2/data/infos/nuscenes_map_infos_eval_sub.pkl
```

> `visualize_nuplan_pred.py` 若支持 `--ann-file` 就直接传；否则复制一份配置把 `data.test.ann_file` 指过去。

### 4.3 NVIDIA：预测可视化（无 GT，只画预测折线）

```bash
cd /data2/wyc/nuplan_maptrv2/MapTRV2
export PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2
export CUDA_VISIBLE_DEVICES=0

python /data2/wyc/nuplan_maptrv2/tools/visualize_nvidia_pred.py \
  /data2/wyc/nuplan_maptrv2/data/infos/nvidia_map_infos_eval_sub.pkl \
  /data2/wyc/nuplan_maptrv2/work_dirs/full/latest.pth \
  --show-dir /data2/wyc/nuplan_maptrv2/reports/pred_vis_nvidia
```

> 输出 BEV 预测折线图（无 GT），可观察模型在 NVIDIA 场景下的泛化效果。

---

## 5. 常见问题

| 现象 | 处理 |
|---|---|
| `FileNotFoundError: img file does not exist` | nuPlan val 图像未解压 → 先跑 2.1 的 val 解压；若是 nuScenes → info 里 data_path 是相对路径，重跑 2.2 |
| `FileNotFoundError: no archive for <log>/CAM_*`（val 解压/eval） | val 传感器分卷不完整（只覆盖 225/1381 log）；用 2.1b 子集方案，只解压/eval 有图像覆盖的 log |
| `KeyError: 'metadata'` | info 顶层缺 metadata；nuScenes/NVIDIA 生成脚本已带，旧文件用 `python -c "..."` 补 `d['metadata']={'version':'x'}` |
| `No module named 'projects'` | 没设 `PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2` |
| 评测 mAP 为 0 / 很低 | 训练还没收敛（现在才 epoch 5，loss 24）；或 GT 生成方式与训练配置不一致，先看可视化图判断预测是否合理 |
| NVIDIA 可视化无输出/全空 | 鱼眼内参是近似 K，预测可能不准；加大 `--score-thresh` 观察 |
| 云服务器内存/显存不足 | `--max-samples` 调小到 3~4，或单卡跑 |

---

## 6. 小结（一条龙命令）

```bash
# 0) 环境变量
export PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2
export CUDA_VISIBLE_DEVICES=0

# 1) nuScenes 小子集（本机生成一次）
PY=/data2/30033/nuplan-devkit/miniconda3/envs/nuplan/bin/python
"$PY" tools/fix_nuscenes_infos_path.py --infos /data2/nuscenes/nuscenes_map_infos_temporal_val.pkl \
  --nuscenes-root /data2/nuscenes --max-samples 8 \
  --out data/infos/nuscenes_map_infos_eval_sub.pkl

# 2) nuPlan 指标
python tools/test_evaluate.py projects/configs/maptrv2/maptrv2_nuplan_full.py \
  work_dirs/full/latest.pth --ann-file data/infos/nuplan_map_infos_full_val.pkl \
  --map-ann data/infos/nuplan_map_anns_full_val.json --max-samples 8

# 3) nuScenes 指标
python tools/test_evaluate.py projects/configs/maptrv2/maptrv2_nuplan_full.py \
  work_dirs/full/latest.pth --ann-file data/infos/nuscenes_map_infos_eval_sub.pkl \
  --map-ann /data2/nuscenes/nuscenes_map_anns_val.json --max-samples 8

# 4) nuPlan 可视化
python tools/visualize_nuplan_pred.py projects/configs/maptrv2/maptrv2_nuplan_full.py \
  work_dirs/full/latest.pth --show-dir reports/pred_vis_full --num-samples 6
```
