#!/usr/bin/env python3
"""修正 nuScenes info 的 data_path 为绝对路径，并可抽一个小子集供云服务器评测。

nuScenes 现成 info（/data2/nuscenes/nuscenes_map_infos_temporal_val.pkl）的 data_path
形如 './data/nuscenes/samples/CAM_FRONT/xxx.jpg'（相对路径，在别处生成时留下的）。
本脚本把 './data/nuscenes/' 前缀替换为 <nuscenes-root>/，并可选只保留前 N 个样本。

若云服务器没挂载 /data2/nuscenes，可用 --copy-images-to 把样本图像拷贝到该目录，
并把 data_path 改写成指向该目录的绝对路径（用于只挂载 /data2/wyc 的场景）。

用法:
    python fix_nuscenes_infos_path.py \
        --infos /data2/nuscenes/nuscenes_map_infos_temporal_val.pkl \
        --nuscenes-root /data2/nuscenes \
        --max-samples 8 \
        --out data/infos/nuscenes_map_infos_eval_sub.pkl
    # 可选拷图像：
    #   --copy-images-to /data2/wyc/nuplan_maptrv2/data/infos/nuscenes_img_sub
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--nuscenes-root", default="/data2/nuscenes")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="只保留前 N 个样本（0=全部）")
    ap.add_argument("--copy-images-to", default=None,
                    help="若提供，把样本图像拷贝到该目录并改写成指向它的绝对路径")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = Path(args.infos)
    nusc_root = str(Path(args.nuscenes_root)).rstrip("/")

    with open(src, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"]
    if args.max_samples > 0:
        data["infos"] = infos[: args.max_samples]
        infos = data["infos"]

    copy_root = str(Path(args.copy_images_to).resolve()) if args.copy_images_to else None
    n = 0
    for s in infos:
        for cam in s["cams"].values():
            p = cam.get("data_path")
            if not p:
                continue
            # 去掉 './data/nuscenes/' 前缀
            rel = p
            for prefix in ("./data/nuscenes/", "data/nuscenes/"):
                if rel.startswith(prefix):
                    rel = rel[len(prefix):]
                    break
            if copy_root:
                dst = os.path.join(copy_root, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(os.path.join(nusc_root, rel), dst)
                cam["data_path"] = dst
            else:
                cam["data_path"] = os.path.join(nusc_root, rel)
            n += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(data, f, protocol=4)
    print(f"fixed {n} paths, kept {len(infos)} samples -> {out}")


if __name__ == "__main__":
    main()
