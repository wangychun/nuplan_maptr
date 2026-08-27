#!/usr/bin/env python3
"""把已生成的 8 路 nuScenes 风格 info 修正为 6 路（nuScenes 命名），无需重跑 build。

之前 build 时未加 --channels，info['cams'] 是 8 路（键为 nuPlan 名 CAM_F0/CAM_B0/...）。
本脚本做纯后处理：
  1) 只保留 6 路（CAM_F0 CAM_B0 CAM_L0 CAM_R0 CAM_L2 CAM_R2）；
  2) 键名映射为 nuScenes 命名（CAM_F0 -> CAM_FRONT 等，需与 tools/build_nuplan_infos.py 的 NUSC_CAM_MAP 一致）；
  3) 按 nuScenes 官方六视角顺序重排（CAM_FRONT, CAM_FRONT_RIGHT, ...）。

结果与用 `--channels ...` 重新 build 完全等价（build 对每路相机独立生成，与是否生成其他路无关）。
内存占用约等于原 pkl 大小（本机 1TB 内存，直接加载没问题）。

用法:
    python reduce_infos_cams.py --infos data/infos/nuplan_map_infos_full_train.pkl
    # 默认覆盖原文件；可用 --out 指定新文件
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path


# 与 tools/build_nuplan_infos.py 的 NUSC_CAM_MAP / NUSC_CAM_ORDER 保持一致
NUSC_CAM_MAP = {
    "CAM_F0": "CAM_FRONT",
    "CAM_B0": "CAM_BACK",
    "CAM_L0": "CAM_FRONT_LEFT",
    "CAM_R0": "CAM_FRONT_RIGHT",
    "CAM_L2": "CAM_BACK_LEFT",
    "CAM_R2": "CAM_BACK_RIGHT",
}
NUSC_CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                  "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
# 6 路方案对应的 nuPlan 原通道
TARGET_NUPLAN_CAMS = ["CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0", "CAM_L2", "CAM_R2"]


def reduce_infos(infos: list) -> int:
    """就地修正每个 sample 的 cams，返回 6 路样本数。"""
    n_6 = 0
    for s in infos:
        cams = {}
        for ch in TARGET_NUPLAN_CAMS:
            if ch in s["cams"]:
                cams[NUSC_CAM_MAP[ch]] = s["cams"][ch]
        s["cams"] = {k: cams[k] for k in NUSC_CAM_ORDER if k in cams}
        if len(s["cams"]) == 6:
            n_6 += 1
    return n_6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--out", default=None, help="输出 pkl 路径（默认覆盖 --infos）")
    args = ap.parse_args()

    src = Path(args.infos)
    dst = Path(args.out) if args.out else src

    print(f"load {src} ...", flush=True)
    with open(src, "rb") as f:
        data = pickle.load(f)
    if "infos" not in data:
        sys.exit(f"[err] {src} 顶层无 'infos'（不是 nuScenes 风格）：{list(data.keys())}")
    infos = data["infos"]
    print(f"samples: {len(infos)}", flush=True)

    n_6 = reduce_infos(infos)
    print(f"6-cam samples: {n_6} / {len(infos)}", flush=True)

    # 先写临时文件再原子替换，避免写失败破坏原文件
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=4)
    os.replace(tmp, dst)
    print(f"saved -> {dst}", flush=True)


if __name__ == "__main__":
    main()
