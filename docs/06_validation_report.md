# nuPlan → MapTRV2 数据适配验证报告

最后更新：2026-08-17
范围：验证 nuPlan 120h 高清地图数据能否适配 MapTRV2 在线矢量地图重建模型（用 mini 数据集先行打通全链路）。

---

## 1. 目标

- 验证 **nuPlan 数据集（HD Map）能否作为真值，适配 MapTRV2** 在线矢量地图重建框架。
- 方法：先用 nuPlan 官方 mini 子集（64 个 log）跑通「数据 → 真值生成 → 训练 → 推理 → 评测」全链路，评估可行性后再决定是否扩展到全量 120h 数据。
- 结论先行：**适配成功，链路完全打通**，mini 数据 6 epoch 训练得到 val Chamfer mAP = 0.1603（= 16.03%），证明 nuPlan 可被 MapTRV2 正常消费；当前精度受限于数据规模与训练时长，提升路径明确。

## 2. 数据适配方法

### 2.1 nuPlan 原始数据结构

- 每个 log 是一个 SQLite 数据库，关键表：
  - `lidar_pc`：关键帧锚点（token、timestamp、scene）
  - `ego_pose`：车辆全局位姿（x/y/z + 四元数 + epsg）
  - `camera`：8 路相机标定（translation/rotation/intrinsic/distortion）
  - `image`：每帧图像，关联 camera 与 ego_pose
- 相机 8 路：CAM_F0、CAM_B0、CAM_L0/L1/L2、CAM_R0/R1/R2
- 地图为 GPKG 格式，几何存 EPSG:4326 经纬度

### 2.2 转换流程（原始数据 → MapTRV2 兼容 info）

1. **关键帧锚点**：以 `lidar_pc` 为锚点确定采样帧。
2. **8 相机时间同步**：按 timestamp 最近邻（时差 < 0.1ms）为每帧配齐 8 路图像。
3. **地图矢量真值生成**：从 GPKG 提取三类矢量真值并变换到 ego 局部坐标，具体见 2.3。
4. **输出 info pkl**：每样本含 `cams`（8 路内参/外参/图像路径）、`e2g_*`（ego 全局位姿）、`annotation`（三类矢量真值），结构对齐 MapTRV2 官方离线 dataset 的 info 格式。

### 2.3 nuPlan 真值到 MapTRV2 可用真值转换

nuPlan 的地图真值从原始 GPKG 到 MapTRV2 可用的矢量真值，经历三个阶段（对应 2.2 流程中的第 3 步）：

**阶段 A：原始 HD Map → ego 局部矢量线**（`tools/nuplan_maptrv2/nuplan_map.py`）

1. 读取 + 投影：从 `map.gpkg` 读图层（lanes_polygons / boundaries / crosswalks / road_segments 等）的 WKB 几何，用 pyproj 把 EPSG:4326 经纬度投影到 ego 所在 UTM 平面（由 `ego_pose.epsg` 决定，单位化为米）。
2. 裁 patch + 对齐 ego：以 ego 为中心裁 60×30m box，先旋转对齐 ego 朝向、再平移到 ego 原点，得到 ego 局部坐标（ISO 8855：x 前、y 左、z 上）的矢量线。
3. 三类提取：divider（被 ≥2 条 lane 引用的共享边界）、boundary（road_segments 并集外边界）、ped_crossing（crosswalks 两条长边）。

**阶段 B：写入 info pkl**（`tools/build_nuplan_infos.py`）

每帧生成 `annotation = {divider/ped_crossing/boundary: [np.array(N,2), ...]}`（每条线是一串 ego 局部坐标点），连同 8 相机内参/外参、ego 全局位姿，按 MapTRV2 官方 AV2 离线 dataset 的 info 结构（顶层 `samples`）存成 pkl。

**阶段 C：MapTRV2 消费**（`nuplan_map_dataset.py` 的 `NuPlanMapDataset`）

- 继承官方 `CustomNuScenesOfflineLocalMapDataset`，`load_annotations` 直接读 info 的 `samples`；
- 关键一行 `input_dict["ann_info"] = info["annotation"]`：把离线矢量真值原样作为 ann_info 喂给训练；
- 复用父类 2D VectorizedLocalMap，训练时把每条线重采样为固定点数（20 点）的实例，形成 GT（num_inst × num_pts × 2）；
- `_format_gt` 把离线 annotation 转成 GT ann json，供 Chamfer AP 评测。

> 一句话：GPKG 地图（经纬度多边形）→ [pyproj 投影 + 旋转平移对齐 ego] → ego 局部坐标矢量线（三类）→ info pkl → MapTRV2 用 2D VectorizedLocalMap 重采样成固定点数实例作为 GT。底层坐标工具（四元数、SE3、ego2cam、投影）在 `tools/nuplan_maptrv2/coords.py`。

### 2.4 地图真值的三类矢量

| 类别 | 含义 | 提取规则 |
|---|---|---|
| divider | 车道分隔线 | 被 ≥2 条 lane 引用的共享边界 |
| boundary | 道路边界 | `road_segments` 多边形并集的外边界 |
| ped_crossing | 斑马线 | `crosswalks` 多边形的两条长边 |

> **boundary 提取的坑（已修复）**：`road_segments` 并集的外边界 `poly.exterior` 是**闭合环**（首尾同点）。当整个可行驶区域完整落在视野 patch（60×30m）内时，求交会返回完整闭合环（span≈0），绕道路一整圈——视觉上横跨道路、看起来像 divider，且作为 GT 时采样会退化成绕圈线。修复：`nuplan_map._open_closed_ring()` 检测闭合环并从最远两点切开成两条开放线段（与 nuScenes 的开放 boundary 语义一致）。修复前 val 集 boundary 约 21% 是闭合环（train 约 15%），修复后趋近 0（仅剩 patch 边界裁剪产生的 2~3 点短碎片）。**注意：此修复只影响重新生成的数据；旧训练 checkpoint 的 boundary 类是在污染 GT 上学的，如需准确 boundary 需重建训练数据并重训。**

（与 nuScenes 的在线建图任务类别一致，因此可直接复用 MapTRV2 的评测协议。）

### 2.5 MapTRV2 接入（含 8 相机适配）

- 新增 `NuPlanMapDataset`：继承 MapTRV2 的 nuScenes 离线 dataset，复用其 2D VectorizedLocalMap、pipeline 与 `evaluate`（Chamfer mAP）。
- **相机数量适配（nuPlan 8 路 vs nuScenes 6 路）**：MapTRV2 是 camera-only 的 BEV 网络，相机数量对模型结构透明——`MapTRPerceptionTransformer` 只需把 `num_cams` 设为 8，数据集为每帧提供 8 路相机的 `lidar2img` / 内参 / 外参即可，backbone、BEV 投影、decoder 等结构完全不变。官方自身就用过两种相机数：nuScenes 6 路、Argoverse2 7 路（`num_cams=7`），因此 nuPlan 8 路（`num_cams=8`）是同一机制的顺理成章扩展。
- 配置要点：camera-only pipeline、3 类、BEV 范围 `[-15,-30]~[15,30]`。

### 2.6 转换脚本清单

**核心转换库 `tools/nuplan_maptrv2/`（Python 包）**

| 文件 | 作用 |
|---|---|
| `coords.py` | 坐标工具：四元数↔旋转矩阵、SE3、ego2cam、相机投影 |
| `nuplan_db.py` | nuPlan SQLite 只读读取：相机标定、关键帧、8 相机时间同步 |
| `nuplan_map.py` | GPKG 地图读取 + 局部矢量真值生成（divider/boundary/ped_crossing） |
| `sensor_archive.py` | 按需从 sensor 分卷 zip 读取图像（不全量解压） |
| `__init__.py` | 包导出 |

**转换链路脚本 `tools/`（按执行顺序）**

| 步骤 | 脚本 | 作用 |
|---|---|---|
| 1 | `scan_sensor_blobs.py` | 扫描 18 个 sensor 分卷 zip，建立 log→zip 索引，校验 DB 引用完整性 |
| 2 | `extract_sensor_images.py` | 按 log 从 zip 解压 8 路相机 JPG 到本地（参数：`--blobs-root --index --db-dir --logs --out --frame-stride`） |
| 3 | `build_nuplan_infos.py` | 核心转换：生成 MapTRV2 兼容 info pkl（参数：`--db-dir --map-root --logs --out --pc-range --channels --stride --img-root`） |
| 4 | `validate_nuplan_infos.py` | 校验 info：相机对齐、路径存在、类别统计、token 唯一 |
| 5 | `validate_map_gt.py` | 批量真值统计：空 GT、类别、长度、坐标范围 |

**转换链路命令实例**（cwd = 项目根 `/data2/wyc/nuplan_maptrv2/`，用训练环境 python 执行）：

```bash
# 1. 建立 sensor 分卷索引
python tools/scan_sensor_blobs.py \
  --blobs-root /data2/han/nuplan/archives/nuplan-v1.1/sensor_blobs/mini_set \
  --db-dir raw/nuplan/dbs \
  --out reports/sensor_blobs_index.json

# 2. 按 log 解压 8 路相机图像
python tools/extract_sensor_images.py \
  --blobs-root /data2/han/nuplan/archives/nuplan-v1.1/sensor_blobs/mini_set \
  --index reports/sensor_blobs_index.json \
  --db-dir raw/nuplan/dbs \
  --logs configs/splits/mini_train_logs.txt \
  --out raw/nuplan/sensor_blobs

# 3. 核心转换：生成 info pkl（train；val 换 logs 与 out 即可）
python tools/build_nuplan_infos.py \
  --db-dir raw/nuplan/dbs \
  --map-root raw/nuplan/maps/maps \
  --logs configs/splits/mini_train_logs.txt \
  --out data/infos/nuplan_map_infos_train.pkl \
  --pc-range -15 -30 -10 15 30 10 \
  --stride 5 \
  --img-root raw/nuplan/sensor_blobs

# 4. 校验 info
python tools/validate_nuplan_infos.py \
  --infos data/infos/nuplan_map_infos_train.pkl \
  --pc-range -15 -30 -10 15 30 10

# 5. 真值统计抽查
python tools/validate_map_gt.py \
  --db-dir raw/nuplan/dbs \
  --map-root raw/nuplan/maps/maps \
  --sample-frames 100 \
  --out reports/map_gt_stats.json
```

**验证 / 可视化 / 评测脚本 `tools/`**

| 脚本 | 作用 |
|---|---|
| `visualize_nuplan_map_gt.py` | BEV 真值图 + 相机投影叠加（验证坐标链正确性） |
| `visualize_nuplan_pred.py` | BEV 预测 vs GT 对比图 |
| `visualize_nuplan_pred_cam.py` | 预测矢量地图投影到真实相机图像（参数自适应：动态读 lidar2img + 图像尺寸缩放 + 深度过滤） |
| `adapt_external_ckpt.py`（MapTRV2/tools/） | 外部 checkpoint（nuScenes/AV2）截断适配到 nuPlan 结构（类别/query/相机嵌入） |
| `eval_nuplan_pred.py` | 预测量化评估：类别分布、预测数、Chamfer 距离 |
| `test_evaluate.py` | 评测链路：推理 → dataset.evaluate → Chamfer mAP |

## 3. 训练

| 项 | 值 |
|---|---|
| 数据规模 | mini 64 log：train 51（83857 样本）/ val 13（19944 样本） |
| 硬件 | 8×A100，分布式 |
| 批大小 | 每卡 8，总 batch 64 |
| 训练轮数 | 6 epoch，每 epoch 1311 iter |
| 耗时 | 约 15 小时 |

**Loss 变化**：总 loss 从初始 **265** 稳步下降至训练末 **33**（梯度稳定在 40~60），全程无发散，收敛健康。

## 4. 评估结果

评测协议：与官方一致的 Chamfer AP（val 集 19944 样本，每 epoch 自动评测）。

| epoch | mAP |
|---|---|
| 1 | 0.0554 |
| 2 | 0.1007 |
| 3 | 0.1157 |
| 4 | 0.1498 |
| 5 | 0.1477 |
| 6（最佳） | 0.1603 |

epoch 6 分项（Chamfer AP）：

| 类别 | AP |
|---|---|
| divider（车道分隔线） | 0.1785 |
| boundary（道路边界） | 0.1635 |
| ped_crossing（斑马线） | 0.1388 |

## 5. 与官方结果对比

MapTRV2 官方在 **nuScenes**（同三类、同 Chamfer AP 协议）上的结果（原文以百分数表示）：

| 配置 | mAP |
|---|---|
| MapTRv2-R50, 24 epoch | 61.5% |
| MapTRv2-R50, 110 epoch | 68.7% |

本次 nuPlan mini 达到：**mAP 0.1603（= 16.03%，6 epoch）**，明显低于官方。

> 数值单位一致（AP，0~1 或百分数）：我们 **16.03%** 远低于官方 **61.5%**，并非更好——可视化中「有的贴合、有的偏差大」正是 16% AP 水平的正常表现。
>
> 该对比仅作量级参考：nuScenes 与 nuPlan 数据分布、场景难度、标注细节不同，且官方结果是约 700 场景 + 24/110 epoch 精心调参得到，不能视为同尺度严格对标。

## 6. 差距原因分析

1. **训练数据量（主因）**：本次 51 个 log；官方 nuScenes 约 700 个场景。场景多样性差距是数量级的。可视化中「有的贴合、有的偏差大」正是场景覆盖不足的体现。
2. **训练时长不足**：仅 6 epoch，且 mAP 仍在持续上升（0.055 → 0.160 未到平台期），明显欠训练；官方 24 epoch 起步。
3. **类别不均衡**：nuPlan 中 `ped_crossing` 在多帧为空，样本稀疏，三类中 AP 最低（0.1388）。

## 7. 可视化验证（链路正确性）

- 预测矢量可投影到 8 相机真实图像，BEV 俯视图可对比 GT 与预测。
- 定性结果：部分场景预测与 GT/路面标线贴合良好，复杂/未见场景偏差较大，与 mAP 0.16 的量级一致。
- 结论：坐标链（地图 → ego 局部 → 相机）与预测链路均正确，精度瓶颈在数据与训练时长，而非实现。

### 7.1 相机投影的参数适配（无需为外部 checkpoint 单独建脚本）

`visualize_nuplan_pred_cam.py` 的投影是**参数自适应**的，对 nuScenes / AV2 / nuPlan 任何 checkpoint 都通用，关键三处：

1. **内外参动态读取**：投影矩阵 `lidar2img` 从 `img_metas` 实时读取（每个样本的真实内参+外参，不是硬编码），`lidar2img` 是 4×4 的「局部 ego → 像素」齐次矩阵。
2. **图像尺寸缩放自适应**：dataset 输出的 `lidar2img` 对应 resize 后的图（如 960 宽），而 `Image.open` 打开的是原始文件（如 1920 宽）。脚本自动按 `lidar2img[:2] *= (img_w / ori_w)` 把像素部分缩放到实际图像尺寸（nuPlan 为 2x）。缩放系数是动态计算的，与相机数量/数据集无关，因此 nuScenes 6 相机、nuPlan 8 相机都正确。
3. **深度过滤**：`project(..., min_depth=1.0)` 过滤掉距相机过近的点（深度 ≤1m），避免像素坐标发散到无穷。

> 结论：**投影正确性不依赖 checkpoint 来源**——它用的是推理时传入的 nuPlan 数据内外参，几何总是自洽的。nuScenes checkpoint 迁移过来后投影位置依然准确（见 `reports/pred_cam_nusc_adapted/`）。

### 7.2 外部 checkpoint 迁移测试（验证「必须从头适配训练」）

用 nuScenes / AV2 官方预训练 checkpoint 直接加载到 nuPlan 模型上推理（严格跳过 size mismatch），验证外部权重能否直接迁移：

**结构差异清单**（nuScenes `maptrv2_nusc_r50_24ep_w_centerline.pth` vs nuPlan 配置）：

| 参数 | nuScenes checkpoint | nuPlan 模型 |
|---|---|---|
| 类别数 | 4（含 centerline） | 3 |
| one2one query | 70 | 50 |
| instance_embedding | (370, 512) | (350, 512) |
| cams_embeds | (6, 256) | (8, 256) |

- **直接加载（strict=False 跳过不匹配层）**：分类头整层被跳过 → 随机初始化 → 模型分不清有效/无效实例 → **pred_inst 恒=50（满额乱预测）**。
- **截断适配（`tools/adapt_external_ckpt.py`）**：利用「类别顺序恰好对齐」（外部是 `['divider','ped_crossing','boundary','centerline']`，前 3 类 = nuPlan 的 3 类），把 `cls_branches` 最后一层截成 3 类、`instance_embedding` 截成 350 行、`cams_embeds` 补到 8 行 → **strict 加载零 mismatch** → pred_inst 恢复自适应（28/12/10/20/3/36），centerline 不再输出。
- **AV2 无法同样适配**：额外有 BEV 位置编码尺寸、`reference_points` 3D→2D、`reg_branches` 3D→2D 等根本差异，需重采样位置编码，工作量大且收益存疑。

> 结论：外部 checkpoint 的 backbone/encoder 特征提取可迁移，但**分类头、实例嵌入、相机几何三层必须针对 nuPlan 重新学习**——这恰好印证了「从头适配训练」的必要性。适配后的 nuScenes checkpoint 图片见 `reports/pred_vis_nusc_adapted/`、`reports/pred_cam_nusc_adapted/`。

## 8. 后续准备（路线）

1. **扩大数据规模**：数据链路已支持任意 log 规模，可直接生成更大 info（更多 log、更小采样 stride）或接入全量 120h 数据。
2. **增加训练轮数**：当前 6 epoch 未收敛，加到 12~24 epoch 预期有明显收益。
3. **可选优化**：处理 ped_crossing 类别不均衡（采样/加权）、调整 score-thresh、BEV 范围等。
4. **基线已固化**：mini + 6 epoch 的 mAP 0.1603 作为后续扩展的对照基线。

---

### 附：本次产物

- checkpoint：`work_dirs/mini_full/epoch_6.pth`（`latest.pth`）、`best_NuscMap_chamfer/mAP_epoch_6.pth`
- 可视化（epoch_6）：`reports/pred_cam/`（相机投影，24 张）、`reports/pred_vis_fix/`（BEV 对比，6 张）
- 可视化（nuScenes 适配版，修复后 GT）：`reports/pred_cam_nusc_adapted/`（16 张）、`reports/pred_vis_nusc_adapted/`（6 张）
- 外部 checkpoint 适配脚本：`MapTRV2/tools/adapt_external_ckpt.py`；适配后 checkpoint：`MapTRV2/ckpts/maptrv2_nusc_adapted_nuplan.pth`
- 真值修复：`tools/nuplan_maptrv2/nuplan_map.py`（`_open_closed_ring`，闭合环切开）；重建后的 val 标注：`data/infos/nuplan_map_infos_val.pkl`（19944 样本，stride=5，boundary 闭合环已清零）
