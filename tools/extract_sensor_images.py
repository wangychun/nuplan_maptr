#!/usr/bin/env python3
"""从 nuPlan sensor 分卷 zip 按需提取相机图像到本地目录。

用法:
    python extract_sensor_images.py \
        --blobs-root /path/mini_set \
        --index reports/sensor_blobs_index.json \
        --db-dir raw/nuplan/dbs \
        --logs logs.txt \
        --channels CAM_F0 CAM_L0 ... \
        --out raw/nuplan/sensor_blobs \
        --frame-stride 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from nuplan_maptrv2 import NuPlanDBReader
from nuplan_maptrv2.sensor_archive import SensorBlobStore

DEFAULT_CHANNELS = ["CAM_F0", "CAM_B0", "CAM_L0", "CAM_L1", "CAM_L2",
                    "CAM_R0", "CAM_R1", "CAM_R2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blobs-root", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--db-dir", required=True)
    ap.add_argument("--logs", required=True, help="log 名列表（每行一个）")
    ap.add_argument("--channels", nargs="*", default=DEFAULT_CHANNELS)
    ap.add_argument("--out", default="raw/nuplan/sensor_blobs")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    store = SensorBlobStore(args.blobs_root, args.index)
    out_root = Path(args.out)
    db_dir = Path(args.db_dir)
    logs = [l.strip() for l in Path(args.logs).read_text().splitlines() if l.strip()]

    total = 0
    skipped = 0
    for log in logs:
        db = db_dir / f"{log}.db"
        if not db.exists():
            print(f"[skip] no db {db.name}")
            continue
        r = NuPlanDBReader(str(db))
        channels = [c for c in args.channels if c in r.cameras]
        frames = r.frames[:: args.frame_stride]
        for fi, frame in enumerate(frames):
            sync = r.sync_images_for_frame(frame, channels=channels)
            for ch, img in sync.items():
                rel = img["filename"]  # <log>/<CAM>/<file>
                dst = out_root / rel
                if args.skip_existing and dst.exists():
                    skipped += 1
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(store.read(rel))
                total += 1
        r.close()
        print(f"[{log}] extracted={total} skipped={skipped}", flush=True)
    print(f"done: extracted {total}, skipped {skipped} -> {out_root}")


if __name__ == "__main__":
    main()
