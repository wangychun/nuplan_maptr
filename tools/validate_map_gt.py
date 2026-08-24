#!/usr/bin/env python3
"""对随机样本批量生成局部地图真值并输出统计，用于数据质量验收。

用法:
    python validate_map_gt.py --db-dir raw/nuplan/dbs \
        --map-root raw/nuplan/maps/maps \
        --sample-frames 100 --seed 0 \
        --out reports/map_gt_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from nuplan_maptrv2 import NuPlanDBReader, NuPlanMap, patch_angle_from_ego

PC_RANGE = [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0]
# DB log.map_version 与地图归档目录一致
MAPS = {
    "sg-one-north": "sg-one-north/9.17.1964/map.gpkg",
    "us-ma-boston": "us-ma-boston/9.12.1817/map.gpkg",
    "us-nv-las-vegas-strip": "us-nv-las-vegas-strip/9.15.1915/map.gpkg",
    "us-pa-pittsburgh-hazelwood": "us-pa-pittsburgh-hazelwood/9.17.1937/map.gpkg",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-dir", required=True)
    ap.add_argument("--map-root", required=True)
    ap.add_argument("--sample-frames", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/map_gt_stats.json")
    ap.add_argument("--dbs", nargs="*", default=None, help="限定某些 db 文件名")
    args = ap.parse_args()

    db_dir = Path(args.db_dir)
    map_root = Path(args.map_root)
    patch_xy = (PC_RANGE[4] - PC_RANGE[1], PC_RANGE[3] - PC_RANGE[0])

    dbs = sorted(db_dir.glob("*.db"))
    if args.dbs:
        dbs = [d for d in dbs if d.name in args.dbs]
    rng = random.Random(args.seed)

    map_cache = {}
    per_class_counts = {c: 0 for c in ["divider", "ped_crossing", "boundary"]}
    empty_samples = 0
    total = 0
    length_list = {c: [] for c in ["divider", "ped_crossing", "boundary"]}
    coord_minmax = {}
    errors = []

    frames_to_check = []
    for db in dbs:
        r = NuPlanDBReader(str(db))
        n = len(r.frames)
        if n == 0:
            r.close()
            continue
        idxs = rng.sample(range(n), min(args.sample_frames, n))
        for i in idxs:
            frames_to_check.append((db, r, i))
        r.close()

    for db, r, fi in frames_to_check:
        try:
            frame = r.frames[fi]
            pose = r.ego_pose(frame["ego_pose_token"])
            loc = r.log["map_version"]
            rel = MAPS.get(loc)
            if rel is None:
                errors.append(f"{db.name}#{fi}: unknown location {loc}")
                continue
            if loc not in map_cache:
                map_cache[loc] = NuPlanMap(str(map_root / rel), pose["epsg"])
            mp = map_cache[loc]
            angle = patch_angle_from_ego([pose["qw"], pose["qx"], pose["qy"], pose["qz"]])
            local = mp.get_local_map((pose["x"], pose["y"]), patch_xy, angle)
            total += 1
            sample_has = False
            for cls in per_class_counts:
                arr = local[cls]
                per_class_counts[cls] += len(arr)
                if arr:
                    sample_has = True
                    for a in arr:
                        length_list[cls].append(
                            float(np.linalg.norm(np.diff(a[:, :2], axis=0), axis=1).sum())
                        )
            if not sample_has:
                empty_samples += 1
            # 全局坐标范围检查（局部坐标应在 patch 内）
            for cls in ["divider", "ped_crossing", "boundary"]:
                for a in local[cls]:
                    c = a[:, :2]
                    mm = coord_minmax.setdefault(cls, {})
                    for k, v in [("xmin", c[:, 0].min()), ("xmax", c[:, 0].max()),
                                 ("ymin", c[:, 1].min()), ("ymax", c[:, 1].max())]:
                        mm[k] = min(mm[k], v) if k in mm and k in ("xmin", "ymin") else (
                            max(mm[k], v) if k in mm else v)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{db.name}#{fi}: {type(e).__name__}: {e}")
        finally:
            r.close()

    stats = {
        "pc_range": PC_RANGE,
        "samples": total,
        "empty_gt": empty_samples,
        "empty_ratio": round(empty_samples / total, 4) if total else None,
        "per_class_instances": per_class_counts,
        "per_class_mean_length": {
            c: round(float(np.mean(length_list[c])), 2) if length_list[c] else None
            for c in length_list
        },
        "per_class_max_length": {
            c: round(float(np.max(length_list[c])), 2) if length_list[c] else None
            for c in length_list
        },
        "coord_minmax": coord_minmax,
        "errors": errors[:50],
        "error_count": len(errors),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
