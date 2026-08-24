#!/usr/bin/env python3
"""生成 MapTRV2 兼容的 nuPlan info pkl（以 AV2 离线 dataset 字段为模板）。

输出结构:
    {"samples": [
        {
          "timestamp": int,
          "lidar_path": str,             # 相对 sensor 路径
          "e2g_translation": np.ndarray, # ego 在全局
          "e2g_rotation": np.ndarray,    # 3x3
          "log_id": str,
          "token": str,
          "cams": {cam: {
              "img_fpath": str,
              "intrinsics": np.ndarray,     # 3x3
              "extrinsics": np.ndarray,     # ego2cam 4x4
              "e2g_translation": np.ndarray,
              "e2g_rotation": np.ndarray,
          }},
          "annotation": {
              "divider": [np.ndarray(N,2)],
              "ped_crossing": [...],
              "boundary": [...],
          },
        }, ...]
    }

用法:
    python build_nuplan_infos.py \
        --db-dir raw/nuplan/dbs \
        --map-root raw/nuplan/maps/maps \
        --logs logs.txt \
        --split train --stride 10 \
        --out data/infos/nuplan_map_infos_train.pkl \
        --pc-range -15 -30 -10 15 30 10 \
        --channels CAM_F0 CAM_F0_L ... (默认全部 8 路)
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from nuplan_maptrv2 import NuPlanDBReader, NuPlanMap, patch_angle_from_ego
from nuplan_maptrv2.coords import (build_se3, quat_to_rotmat, quat_trans_to_se3,
                                   rotmat_to_quat)

MAPS = {
    "sg-one-north": "sg-one-north/9.17.1964/map.gpkg",
    "us-ma-boston": "us-ma-boston/9.12.1817/map.gpkg",
    "us-nv-las-vegas-strip": "us-nv-las-vegas-strip/9.15.1915/map.gpkg",
    "us-pa-pittsburgh-hazelwood": "us-pa-pittsburgh-hazelwood/9.17.1937/map.gpkg",
}
DEFAULT_CHANNELS = ["CAM_F0", "CAM_F0_L", "CAM_F0_R",
                    "CAM_B0", "CAM_L0", "CAM_L1", "CAM_L2",
                    "CAM_R0", "CAM_R1", "CAM_R2"]

# nuPlan 6 路 -> nuScenes 六视角相机命名（对齐 nuScenes 数据集，仅 --format nuscenes 生效）
NUSC_CAM_MAP = {
    "CAM_F0": "CAM_FRONT",
    "CAM_B0": "CAM_BACK",
    "CAM_L0": "CAM_FRONT_LEFT",
    "CAM_R0": "CAM_FRONT_RIGHT",
    "CAM_L2": "CAM_BACK_LEFT",
    "CAM_R2": "CAM_BACK_RIGHT",
}
# nuScenes 官方六视角顺序（对齐 custom_nusc_map_converter 的 camera_types）
NUSC_CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                  "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def build_sample_nuscenes(db_path: Path, r: NuPlanDBReader, frame: dict,
                          channels: list, patch_xy, pc_range, img_root: Path,
                          map_root: Path, map_cache: dict) -> dict:
    """生成与官方 nuScenes info（nuscenes_map_infos_temporal_*.pkl）对齐的条目。

    字段契约对应 MapTRV2 官方 `CustomNuScenesOfflineLocalMapDataset`
    （mmdet3d NuScenesDataset.load_annotations 读顶层 data['infos']，
    get_data_info 读 lidar2ego_*/ego2global_*/cams/can_bus/sweeps/annotation 等）。
    参考 docs/09_ref.md 的 nuPlan->nuScenes 对齐思路。
    """
    pose = r.ego_pose(frame["ego_pose_token"])
    if pose is None:
        return None
    map_version = r.log["map_version"]
    rel = MAPS.get(map_version)
    if rel is None:
        return None
    if map_version not in map_cache:
        map_cache[map_version] = NuPlanMap(str(map_root / rel), pose["epsg"])
    mp = map_cache[map_version]

    angle = patch_angle_from_ego([pose["qw"], pose["qx"], pose["qy"], pose["qz"]])
    local = mp.get_local_map((pose["x"], pose["y"]), patch_xy, angle)

    sync = r.sync_images_for_frame(frame, channels=channels)
    if len(sync) < 3:
        return None

    # ego 在全局（nuScenes quaternion 顺序为 [w,x,y,z]，与 nuPlan qw,qx,qy,qz 一致）
    e2g_t = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64)
    e2g_q = np.array([pose["qw"], pose["qx"], pose["qy"], pose["qz"]], dtype=np.float64)
    ego_se3 = quat_trans_to_se3(e2g_q, e2g_t)

    cams = {}
    for ch in channels:
        img = sync.get(ch)
        if img is None:
            continue
        cam = r.cameras[ch]
        # 相机在 ego 坐标系中的位姿（sensor2ego = inv(ego2cam)，DB 里直接存了相机位姿）
        cam_quat = np.asarray(cam["cam_quat"], dtype=np.float64)      # wxyz
        cam_trans = np.asarray(cam["cam_translation"], dtype=np.float64)
        sensor2ego_se3 = build_se3(quat_to_rotmat(cam_quat), cam_trans)
        # 相机在全局：用图像帧对应的 ego pose（若该帧无独立 pose 则退化为关键帧 pose）
        img_pose = r.ego_pose(img["ego_pose_token"]) or pose
        img_ego_se3 = quat_trans_to_se3(
            [img_pose["qw"], img_pose["qx"], img_pose["qy"], img_pose["qz"]],
            [img_pose["x"], img_pose["y"], img_pose["z"]])
        cam_e2g = img_ego_se3 @ sensor2ego_se3
        # 假设 lidar 与 ego 重合（MapTRV2 不用 lidar），sensor2lidar == sensor2ego
        cam_key = NUSC_CAM_MAP.get(ch, ch)   # 键名对齐 nuScenes（如 CAM_F0 -> CAM_FRONT）
        cams[cam_key] = {
            "data_path": str(img_root / img["filename"]),
            "type": "camera",
            "sample_data_token": img["token"],
            "sensor2ego_translation": cam_trans,
            "sensor2ego_rotation": cam_quat,
            "ego2global_translation": cam_e2g[:3, 3],
            "ego2global_rotation": rotmat_to_quat(cam_e2g[:3, :3]),
            "sensor2lidar_rotation": quat_to_rotmat(cam_quat),
            "sensor2lidar_translation": cam_trans,
            "timestamp": img["timestamp"],
            "cam_intrinsic": cam["intrinsic"],
        }

    # 按 nuScenes 官方六视角顺序重排，使 info['cams'] 与 nuScenes 数据集完全一致
    if cams and all(k in NUSC_CAM_ORDER for k in cams):
        cams = {k: cams[k] for k in NUSC_CAM_ORDER if k in cams}

    info = {
        "lidar_path": frame["filename"],
        "token": f"{r.log['logfile']}_{frame['timestamp']}",
        "prev": "",
        "next": "",
        # 官方 server scene 的 can_bus 就是 zeros(18)；get_data_info 只覆写 [:7] 与 [-2:]
        "can_bus": np.zeros(18, dtype=np.float64),
        "frame_idx": 0,
        "sweeps": [],
        "cams": cams,
        "map_location": map_version,
        "scene_token": r.log["logfile"],
        # lidar 视为与 ego 重合
        "lidar2ego_translation": np.zeros(3, dtype=np.float64),
        "lidar2ego_rotation": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ego2global_translation": e2g_t,
        "ego2global_rotation": e2g_q,
        "timestamp": frame["timestamp"],
        "annotation": {
            "divider": local["divider"],
            "ped_crossing": local["ped_crossing"],
            "boundary": local["boundary"],
        },
        "_meta": {
            "db": db_path.name,
            "map_version": map_version,
            "epsg": pose["epsg"],
            "frame_idx": frame["token"].hex(),
        },
    }
    return info


def build_sample(db_path: Path, r: NuPlanDBReader, frame: dict,
                 channels: list, patch_xy, pc_range, img_root: Path,
                 map_root: Path, map_cache: dict) -> dict:
    pose = r.ego_pose(frame["ego_pose_token"])
    if pose is None:
        return None
    map_version = r.log["map_version"]
    rel = MAPS.get(map_version)
    if rel is None:
        return None
    if map_version not in map_cache:
        map_cache[map_version] = NuPlanMap(str(map_root / rel), pose["epsg"])
    mp = map_cache[map_version]

    angle = patch_angle_from_ego([pose["qw"], pose["qx"], pose["qy"], pose["qz"]])
    local = mp.get_local_map((pose["x"], pose["y"]), patch_xy, angle)

    sync = r.sync_images_for_frame(frame, channels=channels)
    if len(sync) < 3:
        return None

    cams = {}
    e2g_t = np.array([pose["x"], pose["y"], pose["z"]])
    e2g_R = quat_to_rotmat([pose["qw"], pose["qx"], pose["qy"], pose["qz"]])
    for ch in channels:
        img = sync.get(ch)
        if img is None:
            continue
        cam = r.cameras[ch]
        # 图像对应的 ego pose（若不同）
        img_pose = r.ego_pose(img["ego_pose_token"]) or pose
        cams[ch] = {
            "img_fpath": str(img_root / img["filename"]),
            "intrinsics": cam["intrinsic"],
            "extrinsics": cam["ego2cam"],
            "e2g_translation": np.array([img_pose["x"], img_pose["y"], img_pose["z"]]),
            "e2g_rotation": quat_to_rotmat([img_pose["qw"], img_pose["qx"], img_pose["qy"], img_pose["qz"]]),
        }

    sample = {
        "timestamp": frame["timestamp"],
        "lidar_path": frame["filename"],
        "e2g_translation": e2g_t,
        "e2g_rotation": e2g_R,
        "log_id": r.log["logfile"],
        "token": f"{r.log['logfile']}_{frame['timestamp']}",
        "cams": cams,
        "annotation": {
            "divider": local["divider"],
            "ped_crossing": local["ped_crossing"],
            "boundary": local["boundary"],
        },
        "_meta": {
            "db": db_path.name,
            "map_version": map_version,
            "epsg": pose["epsg"],
            "frame_idx": frame["token"].hex(),
        },
    }
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-dir", required=True)
    ap.add_argument("--map-root", required=True)
    ap.add_argument("--logs", default=None,
                    help="要处理的 log 文件名列表（每行一个，不含 .db）；默认全部")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pc-range", type=float, nargs="+",
                    default=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0])
    ap.add_argument("--channels", nargs="*", default=DEFAULT_CHANNELS,
                    help="相机通道（默认为 nuPlan 9 通道候选，实际按 DB 存在过滤）")
    ap.add_argument("--stride", type=int, default=1, help="关键帧采样间隔")
    ap.add_argument("--img-root", default="raw/nuplan/sensor_blobs",
                    help="解压后图像根目录，img_fpath 将指向该目录")
    ap.add_argument("--limit", type=int, default=0, help="每个 log 最多样本数（调试用）")
    ap.add_argument("--format", choices=["nuscenes", "av2"], default="nuscenes",
                    help="info 输出格式：nuscenes=对齐官方 nuScenes info（默认，可直连 "
                         "CustomNuScenesOfflineLocalMapDataset）；av2=旧 AV2 风格（NuPlanMapDataset）")
    args = ap.parse_args()

    db_dir = Path(args.db_dir)
    map_root = Path(args.map_root)
    img_root = Path(args.img_root)
    pc_range = args.pc_range
    patch_xy = (pc_range[4] - pc_range[1], pc_range[3] - pc_range[0])

    logs = None
    if args.logs:
        logs = set()
        for l in Path(args.logs).read_text().splitlines():
            l = l.strip()
            if l:
                logs.add(l[:-3] if l.endswith(".db") else l)

    dbs = sorted(db_dir.glob("*.db"))
    if logs:
        dbs = [d for d in dbs if d.stem in logs]

    build_fn = build_sample_nuscenes if args.format == "nuscenes" else build_sample
    map_cache = {}
    samples = []
    discarded = 0
    for db_path in dbs:
        r = NuPlanDBReader(str(db_path))
        channels = [c for c in args.channels if c in r.cameras]
        frames = r.frames[:: args.stride]
        if args.limit:
            frames = frames[: args.limit]
        db_samples = []
        for frame in frames:
            try:
                s = build_fn(db_path, r, frame, channels, patch_xy, pc_range,
                             img_root, map_root, map_cache)
            except Exception as e:  # noqa: BLE001
                print(f"[err] {db_path.name} {frame['token'].hex()}: {type(e).__name__}: {e}")
                discarded += 1
                continue
            if s is None:
                discarded += 1
                continue
            db_samples.append(s)
        # nuScenes 格式：同 log 内连接 prev/next，并赋 frame_idx
        for i, s in enumerate(db_samples):
            s["frame_idx"] = i
            if args.format == "nuscenes":
                s["prev"] = db_samples[i - 1]["token"] if i > 0 else ""
                s["next"] = db_samples[i + 1]["token"] if i < len(db_samples) - 1 else ""
        samples.extend(db_samples)
        r.close()
        print(f"[{db_path.name}] samples so far={len(samples)} discarded={discarded}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    key = "infos" if args.format == "nuscenes" else "samples"
    with open(out, "wb") as f:
        pickle.dump({key: samples}, f, protocol=4)
    print(f"saved {len(samples)} samples (discarded {discarded}) -> {out}")


if __name__ == "__main__":
    main()
