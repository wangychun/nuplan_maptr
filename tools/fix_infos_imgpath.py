#!/usr/bin/env python3
"""把 nuScenes 风格 info 里的 data_path 修正为绝对路径。

背景：build 时 --img-root 用相对路径（raw/nuplan_full/sensor_blobs），生成的 info 里
data_path 是相对路径（相对 /data2/wyc/nuplan_maptrv2 解析）。训练在 MapTRV2/ 目录运行时，
相对路径会解析成 .../MapTRV2/raw/... 而找不到图像。此脚本把 data_path 统一改为绝对路径。

用法:
    python fix_infos_imgpath.py --infos data/infos/nuplan_map_infos_full_train.pkl \
        --project-root /data2/wyc/nuplan_maptrv2
    # 默认覆盖原文件；可用 --out 指定新文件
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--project-root", default="/data2/wyc/nuplan_maptrv2",
                    help="build 时相对路径的基准（相对 data_path 相对它解析）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(args.infos)
    dst = Path(args.out) if args.out else src
    root = str(Path(args.project_root)).rstrip("/")

    with open(src, "rb") as f:
        data = pickle.load(f)
    infos = data.get("infos", data.get("samples", []))

    n = 0
    for s in infos:
        for cam in s["cams"].values():
            key = "data_path" if "data_path" in cam else "img_fpath"
            p = cam.get(key)
            if p is None:
                continue
            p = str(p)
            if not p.startswith("/"):
                cam[key] = root + "/" + p
                n += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=4)
    os.replace(tmp, dst)
    print(f"fixed {n} paths -> {dst}")


if __name__ == "__main__":
    main()
