#!/usr/bin/env python3
"""从 NVIDIA pai_subset 抽少量帧，生成 MapTRV2 可用的 nuScenes 风格 info（仅可视化用，无地图 GT）。

NVIDIA 数据（pai_subset）只有 7 路相机 mp4 + 标定 + egomotion，没有矢量地图标注，
所以生成的 info['annotation'] 为空，只能用于"预测可视化"，不能算 chamfer 指标。

输入（--data-root 指向 pai_subset）：
  camera/<cam>/<clip_id>.<cam>.mp4                     视频
  camera/<cam>/<clip_id>.<cam>.timestamps.parquet      每帧时间戳
  calibration/camera_intrinsics/*.parquet              (clip_id, camera) -> cx/cy/width/height/fw_poly/bw_poly
  calibration/sensor_extrinsics/*.parquet              (clip_id, sensor) -> qx/qy/qz/qw/x/y/z（相机在车体系）
  labels/egomotion/egomotion.chunk_*.zip               内含 <clip_id>.egomotion.parquet（全局位姿）

输出：从 1 个 clip 抽 n_frames 帧 × 6 路相机 = n*6 张 jpg + 一个 nuScenes 风格 info pkl。
注意：鱼眼相机用针孔 K 近似（fx=fy=focal_ratio*width），仅用于可视化演示。

用法:
    python build_nvidia_infos.py \
        --data-root /data2/wyc/nuplan_maptrv2/MapTRV2/data/nvidia/raw1/pai_subset \
        --clip-id 25cd4769-5dcf-4b53-a351-bf2c5deb6124 \
        --n-frames 3 \
        --out-imgs /data2/wyc/nuplan_maptrv2/data/infos/nvidia_imgs \
        --out-info /data2/wyc/nuplan_maptrv2/data/infos/nvidia_map_infos_eval_sub.pkl
"""
from __future__ import annotations

import argparse
import io
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# NVIDIA 相机 -> nuScenes 六视角命名（近似对应，模型按 info 顺序吃 6 路）
CAM_MAP = {
    "camera_front_wide_120fov": "CAM_FRONT",
    "camera_cross_left_120fov": "CAM_FRONT_LEFT",
    "camera_cross_right_120fov": "CAM_FRONT_RIGHT",
    "camera_rear_left_70fov": "CAM_BACK_LEFT",
    "camera_rear_right_70fov": "CAM_BACK_RIGHT",
    "camera_rear_tele_30fov": "CAM_BACK",
}
NUSC_CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                  "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def quat_wxyz(qx, qy, qz, qw):
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def quat_to_rotmat(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--clip-id", default="25cd4769-5dcf-4b53-a351-bf2c5deb6124")
    ap.add_argument("--n-frames", type=int, default=3)
    ap.add_argument("--focal-ratio", type=float, default=0.5,
                    help="针孔 K 的焦距近似 = focal_ratio * width")
    ap.add_argument("--out-imgs", required=True)
    ap.add_argument("--out-info", required=True)
    args = ap.parse_args()

    root = Path(args.data_root)
    clip = args.clip_id
    out_imgs = Path(args.out_imgs)
    out_imgs.mkdir(parents=True, exist_ok=True)

    # ---------- 读标定 ----------
    intr = pd.read_parquet(root / "calibration/camera_intrinsics/camera_intrinsics.chunk_0000.parquet")
    extr = pd.read_parquet(root / "calibration/sensor_extrinsics/sensor_extrinsics.chunk_0000.parquet")

    # ---------- 读 egomotion（zip 内 <clip>.egomotion.parquet）----------
    egomotion_df = None
    for zpath in sorted((root / "labels/egomotion").glob("*.zip")):
        with zipfile.ZipFile(zpath) as z:
            name = f"{clip}.egomotion.parquet"
            if name in z.namelist():
                egomotion_df = pd.read_parquet(io.BytesIO(z.read(name)))
                break
    if egomotion_df is None:
        raise FileNotFoundError(f"egomotion for {clip} not found in {root}/labels/egomotion")
    ego_ts = egomotion_df["timestamp"].values.astype(np.float64)

    # ---------- 抽帧 ----------
    infos = []
    n_imgs = 0
    import cv2  # 延迟导入，便于提示安装 opencv-python

    # 用第一路相机确定抽哪些时间戳（各相机帧数/时间轴基本对齐）
    first_cam = list(CAM_MAP.keys())[0]
    ts_path = root / "camera" / first_cam / f"{clip}.{first_cam}.timestamps.parquet"
    ts_df = pd.read_parquet(ts_path)
    ts_col = "timestamp" if "timestamp" in ts_df.columns else ts_df.columns[0]
    cam_ts = ts_df[ts_col].values.astype(np.float64)
    n_total = len(cam_ts)
    target_ts = [cam_ts[int(i * (n_total - 1) / max(args.n_frames - 1, 1))] for i in range(args.n_frames)]

    for ti, t_sel in enumerate(target_ts):
        # egomotion 最近行 -> 该帧 ego 全局位姿
        ei = int(np.argmin(np.abs(ego_ts - t_sel)))
        ego_row = egomotion_df.iloc[ei]
        ego2global_t = np.array([ego_row["x"], ego_row["y"], ego_row["z"]], dtype=np.float64)
        ego2global_q = quat_wxyz(ego_row["qx"], ego_row["qy"], ego_row["qz"], ego_row["qw"])
        ego_se3 = np.eye(4)
        ego_se3[:3, :3] = quat_to_rotmat(*ego2global_q)
        ego_se3[:3, 3] = ego2global_t

        cams = {}
        for nv_cam, nusc_cam in CAM_MAP.items():
            mp4 = root / "camera" / nv_cam / f"{clip}.{nv_cam}.mp4"
            if not mp4.exists():
                continue
            # 内参（针孔近似）
            try:
                irow = intr.loc[(clip, nv_cam)]
            except KeyError:
                continue
            cx, cy, W, H = float(irow["cx"]), float(irow["cy"]), int(irow["width"]), int(irow["height"])
            f = args.focal_ratio * W
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            # 外参：相机在车体系（sensor2ego）
            erow = extr.loc[(clip, nv_cam)]
            s2e_q = quat_wxyz(erow["qx"], erow["qy"], erow["qz"], erow["qw"])
            s2e_t = np.array([erow["x"], erow["y"], erow["z"]], dtype=np.float64)
            s2e_se3 = np.eye(4)
            s2e_se3[:3, :3] = quat_to_rotmat(*s2e_q)
            s2e_se3[:3, 3] = s2e_t
            # 相机在全局
            cam_e2g = ego_se3 @ s2e_se3

            # 从 mp4 抽对应帧
            cap = cv2.VideoCapture(str(mp4))
            frame = None
            idx = 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                if abs(float(ts_df[ts_col].values[idx]) - t_sel) < 1e4:  # 时间戳近似匹配
                    frame = fr
                    break
                idx += 1
            cap.release()
            if frame is None:
                continue
            img_rel = f"{clip}/{ti:03d}_{nv_cam}.jpg"
            img_dst = out_imgs / img_rel
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_dst), frame)
            n_imgs += 1

            cams[nusc_cam] = {
                "data_path": str(img_dst),
                "type": "camera",
                "sample_data_token": f"{clip}_{nv_cam}_{ti}",
                "sensor2ego_translation": s2e_t,
                "sensor2ego_rotation": s2e_q,
                "ego2global_translation": cam_e2g[:3, 3],
                "ego2global_rotation": _rot2quat(cam_e2g[:3, :3]),
                "sensor2lidar_rotation": quat_to_rotmat(*s2e_q),
                "sensor2lidar_translation": s2e_t,
                "timestamp": int(t_sel),
                "cam_intrinsic": K,
            }
        if len(cams) < 3:
            continue
        # 按 nuScenes 官方顺序重排
        cams = {k: cams[k] for k in NUSC_CAM_ORDER if k in cams}
        infos.append({
            "lidar_path": f"{clip}_{int(t_sel)}",
            "token": f"{clip}_{int(t_sel)}",
            "prev": "", "next": "",
            "can_bus": np.zeros(18, dtype=np.float64),
            "frame_idx": ti,
            "sweeps": [],
            "cams": cams,
            "map_location": "nvidia",
            "scene_token": clip,
            "lidar2ego_translation": np.zeros(3, dtype=np.float64),
            "lidar2ego_rotation": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "ego2global_translation": ego2global_t,
            "ego2global_rotation": ego2global_q,
            "timestamp": int(t_sel),
            "annotation": {"divider": [], "ped_crossing": [], "boundary": []},
            "_meta": {"clip": clip, "egomotion_idx": int(ei)},
        })

    out_info = Path(args.out_info)
    out_info.parent.mkdir(parents=True, exist_ok=True)
    with open(out_info, "wb") as f:
        pickle.dump({"infos": infos, "metadata": {"version": "nvidia"}}, f, protocol=4)
    print(f"imgs: {n_imgs}, samples: {len(infos)} -> {out_info}")


def _rot2quat(R):
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])


if __name__ == "__main__":
    main()
