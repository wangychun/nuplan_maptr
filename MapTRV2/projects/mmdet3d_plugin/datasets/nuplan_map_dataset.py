"""nuPlan MapTRV2 离线地图数据集。

基于 CustomNuScenesOfflineLocalMapDataset，复用其 2D VectorizedLocalMap、
vectormap_pipeline、prepare_train_data 与 evaluate；仅覆写 get_data_info
以适配 nuPlan info（AV2 风格字段：e2g_translation/e2g_rotation/cams/annotation）。
"""
from __future__ import annotations

import numpy as np
import torch
import mmcv
from mmdet.datasets import DATASETS
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from .nuscenes_offlinemap_dataset import CustomNuScenesOfflineLocalMapDataset

# 默认 8 相机；可通过 dataset 的 cam_names 参数覆盖（如只喂 6 路以匹配 nuScenes 模型）
DEFAULT_CAM_NAMES = ["CAM_F0", "CAM_B0", "CAM_L0", "CAM_L1", "CAM_L2",
                     "CAM_R0", "CAM_R1", "CAM_R2"]


@DATASETS.register_module()
class NuPlanMapDataset(CustomNuScenesOfflineLocalMapDataset):
    """nuPlan 8 相机离线局部地图数据集。

    参数：
        cam_names (list[str], optional)：要保留的相机通道名，默认全部 8 路。
    """

    def __init__(self, *args, cam_names=None, **kwargs):
        self.cam_names = cam_names if cam_names is not None else DEFAULT_CAM_NAMES
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """读取 nuPlan info（顶层键为 samples，AV2 风格）。"""
        data = mmcv.load(ann_file, file_format="pkl")
        data_infos = list(sorted(data["samples"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = None
        self.version = None
        return data_infos

    def _format_gt(self):
        """用离线 annotation 生成 GT ann json（{GTs: [...]}），供 Chamfer 评测。"""
        gt_annos = []
        for sample_id in range(len(self)):
            info = self.data_infos[sample_id]
            ann = info["annotation"]  # {divider/ped_crossing/boundary: [np.array(N,2)]}
            gt_vec_list = []
            for cls_idx, cls_name in enumerate(self.MAPCLASSES):
                for pts in ann.get(cls_name, []):
                    pts = np.asarray(pts, dtype=np.float64)
                    if pts.ndim != 2 or len(pts) < 2:
                        continue
                    gt_vec_list.append(
                        dict(pts=pts, pts_num=len(pts), cls_name=cls_name, type=cls_idx)
                    )
            gt_annos.append({"sample_token": info["token"], "vectors": gt_vec_list})
        nusc_submissions = {"GTs": gt_annos}
        print("\n GT anns writes to", self.map_ann_file)
        mmcv.dump(nusc_submissions, self.map_ann_file)

    def get_data_info(self, index):
        """把 nuPlan info（AV2 风格）转换为 MapTRV2 input_dict。"""
        info = self.data_infos[index]
        translation = np.asarray(info["e2g_translation"], dtype=np.float64)
        rotation = np.asarray(info["e2g_rotation"], dtype=np.float64)

        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            lidar_path=info["lidar_path"],
            sweeps=[],
            ego2global_translation=translation,
            ego2global_rotation=rotation,
            lidar2ego_translation=np.zeros(3, dtype=np.float64),
            lidar2ego_rotation=np.eye(3, dtype=np.float64),
            prev_idx=-1,
            next_idx=-1,
            scene_token=info["log_id"],
            frame_idx=0,
            timestamp=info["timestamp"],
            map_location=info.get("_meta", {}).get("map_version", ""),
        )
        lidar2ego = np.eye(4, dtype=np.float64)
        input_dict["lidar2ego"] = lidar2ego

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            camera2ego = []
            camego2global = []
            cam_types = []
            for cam_type, cam_info in info["cams"].items():
                if cam_type not in self.cam_names:
                    continue
                image_paths.append(cam_info["img_fpath"])
                ego2cam = np.asarray(cam_info["extrinsics"], dtype=np.float64)
                intrinsic = np.asarray(cam_info["intrinsics"], dtype=np.float64)
                viewpad = np.eye(4, dtype=np.float64)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

                # lidar 视为与 ego 重合
                lidar2cam_rt = ego2cam
                lidar2img_rt = viewpad @ lidar2cam_rt
                lidar2img_rts.append(lidar2img_rt)
                lidar2cam_rts.append(lidar2cam_rt)
                cam_intrinsics.append(viewpad)
                camera2ego.append(np.linalg.inv(ego2cam))
                cam_types.append(cam_type)

                cam_e2g = np.eye(4, dtype=np.float64)
                cam_e2g[:3, :3] = np.asarray(cam_info["e2g_rotation"], dtype=np.float64)
                cam_e2g[:3, 3] = np.asarray(cam_info["e2g_translation"], dtype=np.float64)
                camego2global.append(torch.from_numpy(cam_e2g))

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    camera2ego=camera2ego,
                    camera_intrinsics=cam_intrinsics,
                    camego2global=camego2global,
                    cam_type=cam_types,
                )
            )

        input_dict["ann_info"] = info["annotation"]

        # can_bus（与 AV2 模板一致，18 维）
        can_bus = np.ones(18, dtype=np.float64)
        can_bus[:3] = translation
        quat = Quaternion._from_matrix(rotation)
        can_bus[3:7] = quat
        patch_angle = quaternion_yaw(quat) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        input_dict["can_bus"] = can_bus

        # lidar2global（union2one / pv_seg 使用）
        ego2global = np.eye(4, dtype=np.float64)
        ego2global[:3, :3] = Quaternion._from_matrix(rotation).rotation_matrix
        ego2global[:3, 3] = translation
        input_dict["lidar2global"] = ego2global @ lidar2ego
        return input_dict
