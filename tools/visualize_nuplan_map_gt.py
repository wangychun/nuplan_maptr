#!/usr/bin/env python3
"""可视化 nuPlan 局部地图真值：BEV 鸟瞰图 + 相机图像投影叠加。

用法:
    python visualize_nuplan_map_gt.py \
        --db /path/dbs/<log>.db \
        --map /path/maps/<location>/<ver>/map.gpkg \
        --blobs-root /path/mini_set \
        --index /path/reports/sensor_blobs_index.json \
        --frame-idx 100 \
        --out reports/vis/frame100.png \
        [--cam CAM_F0] [--pc-range -15 -30 -10 15 30 10]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from nuplan_maptrv2 import NuPlanDBReader, NuPlanMap, patch_angle_from_ego, project_lidar_to_img
from nuplan_maptrv2.sensor_archive import SensorBlobStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--blobs-root", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--frame-idx", type=int, default=100)
    ap.add_argument("--out", default="reports/vis/gt_bev.png")
    ap.add_argument("--pc-range", type=float, nargs="+",
                    default=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0])
    ap.add_argument("--cam", default=None, help="叠加到指定相机图像；默认仅 BEV")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pc_range = args.pc_range
    patch_xy = (pc_range[4] - pc_range[1], pc_range[3] - pc_range[0])  # (h, w)

    r = NuPlanDBReader(args.db)
    pose = r.ego_pose(r.frames[args.frame_idx]["ego_pose_token"])
    map_db = NuPlanMap(args.map, pose["epsg"])
    angle = patch_angle_from_ego([pose["qw"], pose["qx"], pose["qy"], pose["qz"]])
    local = map_db.get_local_map((pose["x"], pose["y"]), patch_xy, angle)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---------- BEV ----------
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(-patch_xy[1] / 2, patch_xy[1] / 2)
    ax.set_ylim(-patch_xy[0] / 2, patch_xy[0] / 2)
    ax.set_aspect("equal")
    ax.set_title(f"BEV GT frame {args.frame_idx} log={Path(args.db).stem} angle={angle:.1f}")
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    colors = {"divider": "red", "ped_crossing": "green", "boundary": "blue"}
    for cls, col in colors.items():
        for pts in local[cls]:
            ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2)
    ax.legend(
        [plt.Line2D([0], [0], color=c, lw=2) for c in colors.values()],
        list(colors.keys()),
    )
    fig.savefig(out, dpi=110)
    print("saved BEV ->", out)

    # ---------- 相机投影 ----------
    if args.cam:
        store = SensorBlobStore(args.blobs_root, args.index)
        sync = r.sync_images_for_frame(r.frames[args.frame_idx], channels=[args.cam])
        if args.cam not in sync:
            print(f"[warn] no image for {args.cam}")
        else:
            img_bytes = store.read(sync[args.cam]["filename"])
            import io

            from PIL import Image

            img = np.asarray(Image.open(io.BytesIO(img_bytes)))
            cam = r.cameras[args.cam]
            K = cam["intrinsic"]
            ego2cam = cam["ego2cam"]

            fig2, ax2 = plt.subplots(1, 1, figsize=(12, 7))
            ax2.imshow(img)
            for cls, col in colors.items():
                for pts in local[cls]:
                    pts3 = np.hstack([pts, np.zeros((pts.shape[0], 1))])
                    pix, valid = project_lidar_to_img(pts3, ego2cam, K)
                    if valid.sum() < 2:
                        continue
                    ax2.plot(pix[valid, 0], pix[valid, 1], color=col, lw=1.5)
            ax2.set_title(f"{args.cam} projection")
            cam_out = out.with_name(out.stem + f"_{args.cam}.png")
            fig2.savefig(cam_out, dpi=110, bbox_inches="tight")
            print("saved cam ->", cam_out)

    r.close()


if __name__ == "__main__":
    main()
