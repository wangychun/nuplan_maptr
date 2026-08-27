#!/usr/bin/env python3
"""从完整 val info 抽取"图像已覆盖"的少量样本，生成 eval 子集 info + log 名单。

背景：val_set 传感器分卷不完整（24 个分卷只覆盖 225/1381 个 val log），全量 val info
里大部分 log 没有图像，extract/eval 会报 `no archive for <log>/CAM_*`。
本脚本：
  1. 从 val 索引（reports/sensor_blobs_index_full_val.json）取"有传感器覆盖"的 log 集合；
  2. 从完整 val info 中筛出这些 log 的样本，取前 N 个；
  3. 输出 eval 子集 info + 涉及的 log 名单（供 extract 只解压这几个 log）。

用法:
    python make_val_eval_subset.py \
        --infos data/infos/nuplan_map_infos_full_val.pkl \
        --index reports/sensor_blobs_index_full_val.json \
        --max-samples 8 \
        --out-info data/infos/nuplan_val_eval_sub.pkl \
        --out-logs data/infos/nuplan_val_eval_logs.txt
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--out-info", required=True)
    ap.add_argument("--out-logs", required=True)
    args = ap.parse_args()

    idx = json.load(open(args.index))
    covered = set()
    for meta in idx["archives"].values():
        covered.update(meta.get("coverage", {}).keys())
    print(f"索引覆盖 log 数: {len(covered)}")

    with open(args.infos, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"]
    sel = []
    logs = []
    seen = set()
    for s in infos:
        log = s["scene_token"]
        if log not in covered:
            continue
        sel.append(s)
        if log not in seen:
            seen.add(log)
            logs.append(log)
        if len(sel) >= args.max_samples:
            break
    print(f"抽到样本: {len(sel)}, 涉及 log: {len(logs)}")

    data["infos"] = sel
    out_info = Path(args.out_info)
    out_info.parent.mkdir(parents=True, exist_ok=True)
    with open(out_info, "wb") as f:
        pickle.dump(data, f, protocol=4)
    Path(args.out_logs).write_text("\n".join(logs) + "\n")
    print(f"-> {out_info}\n-> {args.out_logs}")


if __name__ == "__main__":
    main()
