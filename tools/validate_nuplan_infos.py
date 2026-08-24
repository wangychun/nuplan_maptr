#!/usr/bin/env python3
"""校验 nuPlan info pkl 完整性：路径存在性、坐标范围、标注统计、token 唯一性。

用法:
    python validate_nuplan_infos.py --infos data/infos/nuplan_map_infos_train.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--pc-range", type=float, nargs="+",
                    default=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0])
    args = ap.parse_args()

    with open(args.infos, "rb") as f:
        data = pickle.load(f)
    # 兼容旧 AV2 风格（顶层 samples）与 nuScenes 对齐风格（顶层 infos）
    if "infos" in data:
        samples = data["infos"]
        fmt = "nuscenes"
    else:
        samples = data["samples"]
        fmt = "av2"
    print(f"format={fmt} samples: {len(samples)}")

    tokens = set()
    missing_files = 0
    missing_cams = 0
    out_of_range = 0
    per_class = {"divider": 0, "ped_crossing": 0, "boundary": 0}
    empty = 0
    n_cams_min = 99
    n_cams_max = 0
    cam_counts = {}
    bad_tokens = 0

    for s in samples:
        if s["token"] in tokens:
            bad_tokens += 1
        tokens.add(s["token"])

        n_cams = len(s["cams"])
        n_cams_min = min(n_cams_min, n_cams)
        n_cams_max = max(n_cams_max, n_cams)
        cam_counts[n_cams] = cam_counts.get(n_cams, 0) + 1

        for ch, cam in s["cams"].items():
            # nuScenes 对齐格式用 data_path，旧 AV2 风格用 img_fpath
            cam_path = cam.get("data_path") or cam.get("img_fpath")
            if cam_path is None:
                continue
            p = Path(cam_path)
            if not p.exists():
                missing_files += 1

        ann = s["annotation"]
        has_any = False
        for cls in per_class:
            arr = ann.get(cls, [])
            per_class[cls] += len(arr)
            if arr:
                has_any = True
            for a in arr:
                c = np.asarray(a)
                if c.shape[1] >= 2:
                    x, y = c[:, 0], c[:, 1]
                    if x.min() < args.pc_range[0] - 1 or x.max() > args.pc_range[3] + 1 \
                       or y.min() < args.pc_range[1] - 1 or y.max() > args.pc_range[4] + 1:
                        out_of_range += 1
        if not has_any:
            empty += 1

    print(f"token duplicate: {bad_tokens}")
    print(f"cams per sample: min={n_cams_min} max={n_cams_max} dist={cam_counts}")
    print(f"missing img files: {missing_files}")
    print(f"empty gt samples: {empty}")
    print(f"instances: {per_class}")
    print(f"out-of-range line coords: {out_of_range}")


if __name__ == "__main__":
    main()
