# nuPlan HD Map 真值训练 MapTRV2 —— 项目说明与续接文档

最后更新：2026-08-17
目的：本文件是项目的“接手说明书”。换电脑后，连上这台远程、直接读本文件，即可知道项目做到哪一步、如何继续。所有路径均为远程机上路径。

---

## 0. 文档索引（先读这里）

| 文档 | 内容 | 何时读 |
|---|---|---|
| **本文件**（`05_readme_baic.md`） | 项目总览、续接状态、环境、命令速查 | 每次接手必读 |
| `06_validation_report.md` | 验证报告：数据适配方法、训练/评测结果、GT 修复、外部 checkpoint 迁移测试 | 想知道"做了什么/结果如何" |
| `07_todo_full_training.md` | 保姆级待办：从零开始跑**完整** nuPlan 训练的分步指引 | 要扩展全量训练时 |
| `01_install.md` ~ `04_visualization.md` | MapTRV2 官方英文文档：环境安装、数据准备、训练评测、可视化 | 涉及 MapTRV2 原生用法时 |

**本次（2026-08-17）新增/变更**：
- **真值 bug 修复**：`boundary` 闭合环问题已修复（详见验证报告 2.4 节），val 标注已重建。
- **外部 checkpoint 迁移测试**：nuScenes/AV2 预训练权重能否直接用 → 结论"必须从头适配训练"（详见验证报告 7.2 节），含适配脚本 `MapTRV2/tools/adapt_external_ckpt.py`。
- 图片目录已清理：仅保留 epoch_6（`pred_vis_fix/`、`pred_cam/`）与 nuScenes 适配版（`pred_vis_nusc_adapted/`、`pred_cam_nusc_adapted/`）。

---

## 1. 项目目标

- 用 nuPlan 120 小时数据集的 HD Map（高精地图）作为真值，训练 MapTRV2 在线矢量地图重建模型。
- 当前先用 mini 数据集（64 个 log）打通全链路：数据 → 真值生成 → 训练 → 推理可视化 → 评测。
- 工作目录：`/data2/wyc/nuplan_maptrv2/`

## 2. 当前状态总览（重要）

- 数据链路已完全打通（nuPlan 原始数据 → MapTRV2 兼容 info → 训练 → 推理 → 评测）。
- **完整 mini 训练已于 2026-08-15 ~ 08-16 完成**（8×A100 分布式，6 epoch 全部跑完）：
  - 配置：`MapTRV2/projects/configs/maptrv2/maptrv2_nuplan_mini.py`
  - 训练数据：train 83857 样本（51 log）、val 19944 样本（13 log）
  - 8 卡、每卡 batch 8（总 batch 64）、6 epoch；每 epoch 1311 iter
  - 产出 checkpoint：`work_dirs/mini_full/epoch_6.pth`、`latest.pth`、`best_NuscMap_chamfer/mAP_epoch_6.pth`（最佳）
  - **val 评测 mAP = 0.1603**（divider 0.1785 / ped_crossing 0.1388 / boundary 0.1635），详见 6.5
  - 环境已重建（旧环境所在 /data 磁盘损坏），详见第 5 节
  - 说明：这个完整训练耗时长。此前已用 1 个 log 的 144 帧做过“过拟合训练”（`work_dirs/overfit3/epoch_24.pth`），已验证链路和预测效果（预测矢量地图能贴合真实道路）。
- **2026-08-17 发现并修复真值 bug**：`boundary` 提取时，当可行驶区域完整落在视野内会得到**闭合环**（横跨道路、视觉上像 divider，且污染训练数据）。已用 `_open_closed_ring` 修复并重建 val 标注（boundary 闭合环 21%→≈0）。⚠️ **旧 checkpoint（epoch_6）的 boundary 是在污染 GT 上学的，如需准确 boundary 需重建训练数据并重训**（详见验证报告 2.4 / 7 节）。
- **2026-08-17 外部 checkpoint 迁移测试**：nuScenes/AV2 预训练权重直接加载 → 分类头被跳过 → 满额乱预测；截断适配后 strict 加载零 mismatch、恢复自适应。结论：外部权重只能迁移 backbone/encoder，**分类头/实例嵌入/相机几何必须针对 nuPlan 重新学习**（详见验证报告 7.2 节）。

## 3. 目录结构与各文件作用

```
/data2/wyc/nuplan_maptrv2/
├── MapTRV2/docs/                      所有文档（本文件为 05，见第 0 节索引）
│   ├── 01_install.md ~ 04_visualization.md    MapTRV2 官方英文文档
│   ├── 05_readme_baic.md              本文件（项目说明与续接文档）
│   ├── 06_validation_report.md        验证报告
│   └── 07_todo_full_training.md       完整训练保姆级待办
├── raw/nuplan/
│   ├── dbs/                           64 个 nuPlan SQLite 日志数据库（已解压，14G）
│   ├── maps/                          nuPlan 官方地图（4 个 location 的 map.gpkg，1.4G）
│   └── sensor_blobs/                  从 sensor 分卷 zip 解压出的相机 JPG（170G，按需解压）
├── tools/
│   ├── nuplan_maptrv2/                核心转换库（Python 包）
│   │   ├── __init__.py                包导出
│   │   ├── coords.py                  坐标工具：四元数↔旋转矩阵、SE3、相机投影、矩阵互逆验证
│   │   ├── nuplan_db.py               nuPlan 数据库只读读取：相机标定、关键帧、8 相机时间同步
│   │   ├── nuplan_map.py              nuPlan GPKG 地图读取 + 局部矢量真值生成（divider/boundary/ped_crossing）
│   │   └── sensor_archive.py          按需从 sensor 分卷 zip 读取图像
│   ├── scan_sensor_blobs.py           扫描 18 个 sensor 分卷，建立 log→zip 索引，校验 DB 引用完整性
│   ├── extract_sensor_images.py       按 log 从 zip 解压相机图像到 raw/nuplan/sensor_blobs
│   ├── build_nuplan_infos.py          生成 MapTRV2 兼容的 info pkl（核心转换脚本）
│   ├── validate_nuplan_infos.py       校验 info：相机对齐、路径存在、类别统计、token 唯一
│   ├── validate_map_gt.py             批量真值统计（空 GT、类别、长度、坐标范围）
│   ├── visualize_nuplan_map_gt.py     BEV 真值图 + 相机投影叠加（验证坐标链用）
│   ├── visualize_nuplan_pred.py       BEV 预测 vs GT 对比图
│   ├── visualize_nuplan_pred_cam.py   预测矢量地图投影到真实相机图像（判断“有没有用”）
│   ├── eval_nuplan_pred.py            预测量化评估：类别分布、预测数、Chamfer 距离
│   ├── test_evaluate.py               评测链路验证：推理 → dataset.evaluate → Chamfer mAP
│   └── show_progress.sh               一键查看训练进度
├── data/infos/
│   ├── nuplan_map_infos_train.pkl     train info（83857 样本）
│   ├── nuplan_map_infos_val.pkl       val info（19944 样本）
│   ├── nuplan_map_infos_overfit.pkl   过拟合 info（144 样本）
│   └── nuplan_map_anns_test.json      评测用的 GT ann（由 _format_gt 生成）
├── configs/
│   ├── splits/mini_train_logs.txt      train 的 log 名单（51 个）
│   ├── splits/mini_val_logs.txt        val 的 log 名单（13 个）
│   └── maptrv2_nuplan_overfit.py      过拟合配置（早期版本，正式的在 MapTRV2/ 内）
├── MapTRV2/                           MapTRV2 基线（从 /data2/han/MapTR-maptrv2 复制，可写副本）
│   ├── tools/adapt_external_ckpt.py   外部 checkpoint 适配脚本（nuScenes/AV2 → nuPlan 结构，2026-08-17 新增）
│   ├── ckpts/maptrv2_nusc_adapted_nuplan.pth   适配后的 nuScenes checkpoint（迁移测试用）
│   └── projects/configs/maptrv2/
│       ├── maptrv2_nuplan_overfit.py  过拟合训练配置（144 样本）
│       └── maptrv2_nuplan_mini.py     完整 mini 训练配置（当前训练用）
├── work_dirs/
│   ├── mini_full/                     完整 mini 训练输出（日志 + 未来 checkpoint）
│   ├── overfit2/  overfit3/           过拟合训练输出（overfit3/epoch_24.pth 是最早验证用 checkpoint）
│   └── *.log                          各次训练日志
├── reports/                           可视化与评测结果
│   ├── pred_vis_fix/                  epoch_6 BEV 预测 vs GT（保留）
│   ├── pred_cam/                      epoch_6 相机投影（保留）
│   ├── pred_vis_nusc_adapted/         nuScenes 适配版 BEV（修复后 GT 重新生成）
│   ├── pred_cam_nusc_adapted/         nuScenes 适配版相机投影（修复后 GT 重新生成）
│   └── （AV2/检查目录已删除）
├── .toolchain/gxx-wrap.sh             mmdet3d 编译链接 wrapper（链接走系统 g++，解决 ld 找不到 libm）
└── .wheels/                           本地下载的 torch/torchvision/mmcv wheel
```

## 4. 核心：nuPlan 数据如何适配 MapTRV2

### 4.1 nuPlan 原始数据结构

- 每个 log 是一个 SQLite 数据库（`raw/nuplan/dbs/<log>.db`），关键表：
  - `log`：location、map_version（地图版本，与地图目录一致）
  - `camera`：每路相机标定，translation/rotation/intrinsic/distortion 存为 pickle BLOB
  - `image`：每帧图像，`filename_jpg`、timestamp、关联 camera 与 ego_pose
  - `lidar_pc`：关键帧锚点（token、timestamp、scene）
  - `ego_pose`：车辆全局位姿 x/y/z + 四元数(qw,qx,qy,qz) + epsg
- 相机 8 路：CAM_F0、CAM_B0、CAM_L0/L1/L2、CAM_R0/R1/R2
- 传感器实体在 18 个分卷 zip（9 相机 + 9 lidar），位于 `/data2/han/nuplan/archives/nuplan-v1.1/sensor_blobs/mini_set/`（只读，不整体解压，按需读）
- 地图在 `raw/nuplan/maps/maps/<location>/<version>/map.gpkg`，EPSG:4326 经纬度

### 4.2 转换流程（info 是怎么生成的）

1. 建立 sensor 索引：`scan_sensor_blobs.py` 扫描 18 个 zip，记录每个 zip 含哪些 log 的哪些相机，得到 `reports/sensor_blobs_index.json`。
2. 解压图像：`extract_sensor_images.py` 按 log 从 zip 把 8 路相机 JPG 解压到 `raw/nuplan/sensor_blobs/<log>/<CAM>/xxx.jpg`。
3. 生成 info：`build_nuplan_infos.py`（用训练环境 maptr 的 Python 跑，原因见第 5 节注意事项）：
   - 打开 DB，以 `lidar_pc` 为关键帧锚点；
   - 对每帧做 8 相机时间同步（按 timestamp 最近邻，时差 <0.1ms）；
   - 读取 ego_pose（全局位姿）和相机标定；
   - 用 `NuPlanMap` 从 gpkg 生成局部矢量真值（见 4.3）；
   - 输出每样本：
     ```
     {
       timestamp, lidar_path, e2g_translation, e2g_rotation, log_id, token,
       cams: {CAM_F0: {img_fpath, intrinsics(3x3), extrinsics(ego2cam 4x4), e2g_*}, ...},
       annotation: {divider: [np.array(N,2)], ped_crossing: [...], boundary: [...]}
     }
     ```
   - 这是 MapTRV2 官方 AV2 离线 dataset 的 info 结构（顶层 `samples` 列表）。
4. 校验：`validate_nuplan_infos.py`。

### 4.3 地图真值（HD Map → 三类矢量）

- 从 `map.gpkg` 读取图层（WKB 从字节偏移 40 起），坐标是 EPSG:4326 经纬度。
- 用 pyproj 把地图投影到 ego 位姿对应的 EPSG（UTM，如新加坡 32648）。
- 以 ego 为中心裁 60x30m 的 patch，再变换到 ego 局部坐标。
- 类别映射：
  - divider（车道分隔线）= 被 >=2 条 lane 引用的共享边界（`boundaries` 图层 + `lanes_polygons` 的 left/right_boundary_fid）
  - boundary（道路边界）= `road_segments` 多边形并集的外边界
  - ped_crossing（斑马线）= `crosswalks` 多边形的两条长边
- 实现见 `tools/nuplan_maptrv2/nuplan_map.py`。

### 4.4 MapTRV2 接入（训练侧）

- 新增 `NuPlanMapDataset`（`MapTRV2/projects/mmdet3d_plugin/datasets/nuplan_map_dataset.py`）：
  - 继承 MapTRV2 的 nuScenes 离线 dataset（复用 2D VectorizedLocalMap、vectormap_pipeline、evaluate）；
  - `load_annotations` 读我们 info 的 `samples`；
  - `get_data_info` 把 AV2 风格 info 转成 MapTRV2 的 input_dict（构造 lidar2img、can_bus、lidar2global 等）；
  - 覆写 `_format_gt`：用离线 annotation 生成评测 GT（Chamfer mAP）。
- 训练配置要点：8 相机（`num_cams=8`）、camera-only pipeline（无点云深度监督，参考官方 AV2 配置）、3 类、`point_cloud_range=[-15,-30,-10,15,30,10]`。

## 5. 环境说明（2026-08-15 在另一台远程重建）

> 旧训练环境在 `/data/Miniforge3/envs/maptr`，其所在本地盘 `/data`（nvme）已损坏（ls 报 Input/output error），已不可用，故在 `/home/bicv01` 重建。

- 转换环境：`/data2/wyc/nuplan_maptrv2/.venv`（Python 3.13 + numpy2.5 + shapely2.1 + pyproj + nuplan-devkit + matplotlib + pillow）
- **训练环境：`/home/bicv01/miniforge3/envs/maptr`**（Python 3.8.20 + torch1.9.1+cu111 + torchvision0.10.1+cu111 + mmcv-full1.4.0 + mmdet2.14.0 + mmseg0.14.1 + mmdet3d0.17.2 + GKT + numpy1.19.5 + numba0.48 + nuscenes-devkit1.0.4）
- conda 管理器：`/home/bicv01/miniforge3`（conda 26.3.2，镜像用中科大：`mirrors.ustc.edu.cn/anaconda/...`，清华 conda-forge repodata 会下载失败）
- 安装要点（重建环境时照做）：
  1. torch/torchvision/mmcv-full 用 `.wheels/` 里本地 wheel；**yapf 必须装 0.31.0**（新版 yapf 移除了 `FormatCode(verify=...)` 参数，否则 mmcv `cfg.dump` 崩溃）。
  2. mmdet3d 不重新编译（nvcc 12.5 与 torch1.9+cu111 不兼容），直接复用已编译 `.so`：在 site-packages 放 `maptrv2.pth`，内容为 MapTRV2 根目录 + `mmdetection3d` 目录两行。
  3. GKT（geometric_kernel_attn）已编译 egg 在 `projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn/dist/*.egg`，直接 `unzip` 解压到 site-packages（pip 24 不接受 egg 路径）。
  4. **运行训练必须设 `LD_LIBRARY_PATH=<env>/lib/python3.8/site-packages/torch/lib`**，否则 GKT .so 找不到 libc10.so。
  5. **启动命令必须用绝对路径 python + 显式干净 PATH**：旧 `.bashrc` 曾把坏盘 `/data/Miniforge3/bin` 加进 PATH，导致 `nohup`/`timeout` 的 execvp 报 Input/output error（已修复 `.bashrc`，但已开终端需重开或用干净 PATH）。
- 注意：info pkl 必须用 maptr 环境（numpy1.19）生成；用 .venv（numpy2.5）生成的 pkl 训练环境读不了（numpy._core 错误）。

## 6. 当前训练到哪一步 & 如何继续

### 6.1 看进度

```bash
bash /data2/wyc/nuplan_maptrv2/tools/show_progress.sh
# 或实时看日志
tail -f /data2/wyc/nuplan_maptrv2/work_dirs/mini_full/train_mini_full.log
```

### 6.2 重启/继续训练（8 卡分布式）

若训练进程停了（`ps aux | grep train.py` 无结果），重启：

```bash
cd /data2/wyc/nuplan_maptrv2/MapTRV2
export PATH=/home/bicv01/miniforge3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/home/bicv01/miniforge3/envs/maptr/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2
nohup /home/bicv01/miniforge3/envs/maptr/bin/python -m torch.distributed.launch \
  --nproc_per_node=8 --master_port=28509 \
  tools/train.py projects/configs/maptrv2/maptrv2_nuplan_mini.py \
  --launcher pytorch \
  --work-dir /data2/wyc/nuplan_maptrv2/work_dirs/mini_full \
  > /data2/wyc/nuplan_maptrv2/work_dirs/mini_full/train_mini_full.log 2>&1 &
```

- 8 卡每卡 batch 8（总 batch 64），6 epoch 约 14.8 小时；epoch 1 约 2.5 小时，可先跑出 epoch 1 checkpoint 就推理看效果。
- GPU 上跑着别人的 filler 进程（`calculate_filler.py`，adaptive 模式会自动避让），**不要 kill**。
- 若想加快：改 `maptrv2_nuplan_mini.py` 里 `samples_per_gpu` 或减 `total_epochs`；或重生成 info 时用更大 `--stride`（当前 5，即 2Hz；改成 10 即 1Hz，样本减半）。

### 6.3 训练出 checkpoint 后：推理可视化（判断有没有用）

```bash
cd /data2/wyc/nuplan_maptrv2/MapTRV2
# 预测投影到真实相机图像（最直观）
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /home/bicv01/miniforge3/envs/maptr/bin/python \
  /data2/wyc/nuplan_maptrv2/tools/visualize_nuplan_pred_cam.py \
  projects/configs/maptrv2/maptrv2_nuplan_mini.py \
  /data2/wyc/nuplan_maptrv2/work_dirs/mini_full/epoch_1.pth \
  --show-dir /data2/wyc/nuplan_maptrv2/reports/pred_cam_mini --score-thresh 0.15

# BEV 预测 vs GT
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /home/bicv01/miniforge3/envs/maptr/bin/python \
  /data2/wyc/nuplan_maptrv2/tools/visualize_nuplan_pred.py \
  projects/configs/maptrv2/maptrv2_nuplan_mini.py \
  /data2/wyc/nuplan_maptrv2/work_dirs/mini_full/epoch_1.pth \
  --show-dir /data2/wyc/nuplan_maptrv2/reports/pred_vis_mini --score-thresh 0.15
```

### 6.4 评测（Chamfer mAP）

```bash
cd /data2/wyc/nuplan_maptrv2/MapTRV2
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /home/bicv01/miniforge3/envs/maptr/bin/python \
  /data2/wyc/nuplan_maptrv2/tools/test_evaluate.py \
  projects/configs/maptrv2/maptrv2_nuplan_mini.py \
  /data2/wyc/nuplan_maptrv2/work_dirs/mini_full/epoch_1.pth \
  --ann-file /data2/wyc/nuplan_maptrv2/data/infos/nuplan_map_infos_val.pkl \
  --map-ann /data2/wyc/nuplan_maptrv2/data/infos/nuplan_map_anns_val.json
```

### 6.5 已经验证过的事实（可复现参考）

- 过拟合训练（144 样本，24 epoch）：`work_dirs/overfit3/epoch_24.pth`。
  - 预测投影到 CAM_F0 图像，预测虚线贴合真实车道线/路缘（`reports/pred_cam/`）。
  - Chamfer 总体 2.79m（6 epoch 时 3.25m）；ped_crossing 从没学到 → 能预测。
  - 评测 mAP ≈ 0.038（144 样本、过拟合早期，数值低属正常）。
- **完整 mini 训练（2026-08-15~16，8 卡，6 epoch，train 83857 / val 19944）**：
  - checkpoint：`work_dirs/mini_full/epoch_6.pth`（`latest.pth` 指向它），最佳 `best_NuscMap_chamfer/mAP_epoch_6.pth`
  - val 评测（自动 evaluate，Chamfer mAP）：**mAP 0.1603**（divider 0.1785 / ped_crossing 0.1388 / boundary 0.1635）
  - 训练末段 loss 约 33（iter 1300/1311），从初始 265 稳步下降，梯度稳定。
- 全 mini train/val info 已生成并校验（8 相机全对齐、无缺失文件）。
- **外部 checkpoint 迁移测试（2026-08-17）**：nuScenes `maptrv2_nusc_r50_24ep_w_centerline.pth` 直接加载 → 分类头跳过 → pred_inst 恒=50；截断适配（`MapTRV2/tools/adapt_external_ckpt.py`）→ strict 加载零 mismatch → pred_inst 恢复自适应（28/12/10/20/3/36）、centerline 消失。结论见验证报告 7.2 节。

## 7. 关键注意事项 / 已知问题

1. `/data2/han` 是只读（MapTRV2 基线、nuplan 归档、sensor blobs 源），不要写；所有产物在 `/data2/wyc/nuplan_maptrv2`。
2. info 生成必须用 maptr 环境（见第 5 节），否则 pickle 不兼容。
3. sensor 分卷 zip 只按需读取/解压，不要整体解压（约 1.1TB）。
4. 地图 gpkg 几何是 EPSG:4326，必须用 pyproj 投影到 ego 的 EPSG；DB 的 `location` 字段是简写（如 las_vegas），要用 `map_version` 定位地图。
5. 8 卡训练约 14.8 小时，可按需调整 batch/epoch/采样。
6. 训练进程用 nohup 启动可脱离终端；普通 `&` 后台会在终端会话清理时被杀（第一次中断即此原因）。
7. **运行训练/推理必须设 `LD_LIBRARY_PATH` 指向 torch/lib**（GKT .so 依赖），且启动命令用绝对路径 python + 干净 PATH（坏盘 `/data` 的历史遗留问题，见第 5 节）。
8. 本地盘 `/data` 已损坏，不要在 `/data` 下读写；conda/环境放 `/home/bicv01/miniforge3`。
9. **`build_nuplan_infos.py` 的 `--stride` 必须与解压图片的 `--frame-stride` 一致**（mini 用 5 = 2Hz）。若用 stride=1 全量采样，会采样到没有解压图片的帧，可视化时报 `FileNotFoundError`。
10. **可视化脚本要从项目根目录 `/data2/wyc/nuplan_maptrv2` 运行**（`img_fpath` 是相对路径，从 `MapTRV2/` 子目录运行会找不到图）。命令需 `PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2` + maptr 环境 + `LD_LIBRARY_PATH`。
11. **重建 info 要用 nuplan-devkit 环境**：`/data2/30033/nuplan-devkit/miniconda3/envs/nuplan/bin/python` + `PYTHONPATH=/data2/30033/nuplan-devkit`（maptr 环境缺 pyproj / nuplan 模块，DB 的 pickle BLOB 反序列化需要 nuplan 包）。
12. **真值 bug**：旧 `boundary` 真值有闭合环问题（详见验证报告 2.4 节），已修复并重建 val；重建后的 val 标注为 `data/infos/nuplan_map_infos_val.pkl`（19944 样本，stride=5）。若要用正确 GT 训练，需用修复后的 `nuplan_map.py` 重新生成 train info。

## 8. 下一步（路线）

- 等完整 mini 训练跑出 epoch 1 checkpoint → 推理可视化 + Chamfer 评测。
- **⚠️ 重要提醒**：旧训练数据（train info）的 boundary 含闭合环污染。若要训练出准确的 boundary，需先用修复后的 `tools/nuplan_maptrv2/nuplan_map.py` **重新生成 train info**（`--logs configs/splits/mini_train_logs.txt --stride 5`），再重训。val 标注已重建。
- 确认预测质量后，可扩展：全量图像已具备，直接生成更大规模 info（调整 stride / 换 120h 数据）继续训练。
- 完整扩展训练的分步指引见 `07_todo_full_training.md`。
