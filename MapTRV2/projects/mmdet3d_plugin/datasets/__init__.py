from .nuscenes_dataset import CustomNuScenesDataset
from .builder import custom_build_dataset

from .nuscenes_map_dataset import CustomNuScenesLocalMapDataset
from .nuscenes_offlinemap_dataset import CustomNuScenesOfflineLocalMapDataset
from .nuplan_map_dataset import NuPlanMapDataset
# 说明：AV2 相关 dataset（CustomAV2LocalMapDataset/CustomAV2OfflineLocalMapDataset）
# 依赖 av2 包（需 numpy>=1.20 的 numpy.typing），与 mmdet3d 要求的 numpy<1.20 冲突。
# nuPlan 训练不使用 AV2 配置，故此处不导入；如需 AV2 配置请单独处理依赖。
__all__ = [
    'CustomNuScenesDataset','CustomNuScenesLocalMapDataset'
]
