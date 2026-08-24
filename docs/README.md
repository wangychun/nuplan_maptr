# MapTRV2 × nuPlan 项目文档索引

本目录集中存放本项目（nuPlan HD Map 训练 MapTRV2 在线建图）的全部文档。
编号 `NN_` 前缀表示文档的先后顺序，从环境搭建到完整训练的推进路径：

| 编号 | 文件 | 内容 | 阶段 |
|---|---|---|---|
| 01 | `01_install.md` | MapTRV2 官方安装步骤（英文） | 环境搭建 |
| 02 | `02_prepare_dataset.md` | MapTRV2 官方数据准备（英文，nuScenes） | 数据准备 |
| 03 | `03_train_eval.md` | MapTRV2 官方训练与评测（英文） | 训练评测 |
| 04 | `04_visualization.md` | MapTRV2 官方可视化脚本说明（英文） | 可视化 |
| 05 | `05_readme_baic.md` | **项目说明与续接文档**（接手必读：状态、环境、命令速查） | 项目总览 |
| 06 | `06_validation_report.md` | **验证报告**（数据适配方法、训练/评测结果、GT 修复、外部 checkpoint 迁移测试） | 验证总结 |
| 07 | `07_todo_full_training.md` | **完整 nuPlan 训练保姆级待办**（从零到全量训练的分步指引） | 扩展路线 |

## 推荐阅读顺序

1. **接手项目**：先读 `05_readme_baic.md`（总览 + 当前状态 + 命令速查）。
2. **了解成果**：再看 `06_validation_report.md`（做了什么、结果如何、已知问题）。
3. **深入技术**：涉及 MapTRV2 原生用法时参考 `01~04` 官方文档。
4. **扩展训练**：要跑完整 nuPlan 训练时，严格按 `07_todo_full_training.md` 执行。

## 关键路径速查

- 项目根目录：`/data2/wyc/nuplan_maptrv2/`
- 训练环境：`/home/bicv01/miniforge3/envs/maptr`（Python 3.8 + torch1.9）
- 转换/重建 info 需用 nuplan-devkit 环境：`/data2/30033/nuplan-devkit/miniconda3/envs/nuplan/bin/python`
- 训练配置：`MapTRV2/projects/configs/maptrv2/maptrv2_nuplan_mini.py`
- 核心转换库：`tools/nuplan_maptrv2/`（`nuplan_map.py` 含 boundary 闭合环修复）
