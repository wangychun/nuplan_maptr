# 数据目录（不入库）

本目录存放训练/评测所需的数据（nuScenes、nuPlan 等），因体积大（TB 级）不入库。

- **nuScenes**：从 [nuScenes 官网](https://www.nuscenes.org/download) 下载（v1.0-trainval + CAN bus expansion）。
- **nuPlan**：从 [nuPlan 官网](https://www.nuscenes.org/nuplan) 下载（DB 与 sensor blobs 分卷）；本仓库数据处理流程见根目录 `docs/07_todo_full_training.md`。

生成物（info pkl 等）落在仓库 `data/infos/`，同样不入库。
