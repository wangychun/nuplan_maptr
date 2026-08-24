# 获取 mmdetection3d（本目录大部分内容不入库）

本目录是 MapTRV2 依赖的 OpenMMLab **mmdetection3d** 框架，因体积较大（约 393MB），
除本说明与官方 README 外**不入库**（Git LFS / 大文件均被 `.gitignore` 排除）。

请从官方仓库获取（MapTRV2 使用的版本为 **v1.0.0rc6**）：

```bash
git clone -b v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git
```

放置方式：将克隆得到的 `mmdetection3d/` 目录放回本处（`MapTRV2/mmdetection3d/`）。

MapTRV2 通过 `PYTHONPATH=.../MapTRV2` 引用该目录下的 `mmdetection3d/` 包。

> 注：本项目在其基础上按需做了一些适配（训练/数据流程改动在 `MapTRV2/projects/` 与仓库 `tools/` 中，均已入库）。
