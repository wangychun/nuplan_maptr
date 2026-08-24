#!/usr/bin/env python3
"""扫描 nuPlan sensor blob 归档分卷，建立 log->camera/lidar 覆盖索引，
并校验 mini DB 中的相机/点云文件引用能否在归档中找到。

用法:
    python scan_sensor_blobs.py --blobs-root /path/to/mini_set \
        --db-dir /path/to/dbs \
        --out /path/to/reports/sensor_blobs_index.json \
        [--sample-per-db 8]

只读操作：只读取 zip central directory 与 DB 索引，不写入 /data2/han。
"""
import argparse
import json
import sqlite3
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZipFile


class _TarArchive:
    """把未压缩 tar 包装成与 ZipFile 兼容的只读接口（sensor blob 是 .zip 命名的 tar）。"""

    def __init__(self, path):
        self._tf = tarfile.open(path, "r:")
        self._names = self._tf.getnames()

    def namelist(self):
        return self._names

    def close(self):
        self._tf.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _open_archive(path: Path):
    """按文件头自动识别 zip 或 tar。"""
    path = str(path)
    if tarfile.is_tarfile(path):
        return _TarArchive(path)
    return ZipFile(path)


def scan_archive(path: Path, prefix_hint: str):
    """返回该归档内 {log: {cam: count}} 覆盖表。"""
    coverage: dict = {}
    with _open_archive(path) as z:
        for name in z.namelist():
            parts = name.split("/")
            # 形如 <zip_dir>/<log>/<CAM|MergedPointCloud>/<file>
            if len(parts) < 4:
                continue
            log = parts[1]
            sensor = parts[2]
            coverage.setdefault(log, {}).setdefault(sensor, 0)
            coverage[log][sensor] += 1
    return coverage


def _scan_one(zf: Path):
    kind = "camera" if "camera" in zf.name else ("lidar" if "lidar" in zf.name else "other")
    print(f"scanning {zf.name} ...", flush=True)
    return zf.name, {
        "kind": kind,
        "size_gb": round(zf.stat().st_size / 1e9, 2),
        "coverage": scan_archive(zf, kind),
    }


def build_archive_index(blobs_root: Path, jobs: int = 1):
    zfiles = sorted(blobs_root.glob("*.zip"))
    if jobs <= 1 or len(zfiles) <= 1:
        return dict(_scan_one(zf) for zf in zfiles)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return dict(ex.map(_scan_one, zfiles))


def resolve_archive(archive_index, log: str, sensor: str):
    """返回包含 (log, sensor) 的 zip 名，否则 None。"""
    for zname, meta in archive_index.items():
        if log in meta["coverage"] and sensor in meta["coverage"][log]:
            return zname
    return None


def check_db(db_path: Path, archive_index, sample_per_db: int):
    db_name = db_path.name
    report = {"db": db_name, "camera_missing": [], "lidar_missing": [], "ok": True}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        cur = conn.execute(
            "SELECT filename_jpg FROM image ORDER BY timestamp LIMIT ?",
            (sample_per_db,),
        )
        for row in cur:
            rel = row["filename_jpg"]
            parts = rel.split("/")
            log, sensor = parts[0], parts[1]
            if resolve_archive(archive_index, log, sensor) is None:
                report["camera_missing"].append(rel)
                report["ok"] = False

        cur = conn.execute(
            "SELECT filename FROM lidar_pc ORDER BY timestamp LIMIT ?",
            (sample_per_db,),
        )
        for row in cur:
            rel = row["filename"]
            parts = rel.split("/")
            log, sensor = parts[0], parts[1]
            if resolve_archive(archive_index, log, sensor) is None:
                report["lidar_missing"].append(rel)
                report["ok"] = False
        conn.close()
    except sqlite3.DatabaseError as e:
        # 单个 DB 损坏（malformed）不应中断整个扫描，记录后继续
        report["ok"] = False
        report["error"] = f"sqlite: {e}"
        try:
            conn.close()
        except Exception:
            pass
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blobs-root", required=True, help="sensor blob 归档目录")
    ap.add_argument("--db-dir", required=True, help="mini 数据库目录")
    ap.add_argument("--out", required=True, help="输出 JSON 路径")
    ap.add_argument("--sample-per-db", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=8, help="并行扫描归档数（tar 遍历较慢，默认 8）")
    ap.add_argument("--skip-db-check", action="store_true",
                    help="跳过 DB 引用校验（full 数据集建议加：dbs 含大量未发布传感器数据的 log，MISSING 是预期的）")
    args = ap.parse_args()

    blobs_root = Path(args.blobs_root)
    db_dir = Path(args.db_dir)
    archive_index = build_archive_index(blobs_root, jobs=args.jobs)

    # 先把归档索引落盘，避免后续 DB 校验出任何问题丢结果
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = {"archives": archive_index, "dbs": [], "summary": {"db_total": 0, "db_ok": 0, "db_missing": 0}}
    out.write_text(json.dumps(partial, indent=2, ensure_ascii=False))

    db_reports = []
    if args.skip_db_check:
        print("跳过 DB 校验（--skip-db-check）")
    else:
        for db_path in sorted(db_dir.glob("*.db")):
            rep = check_db(db_path, archive_index, args.sample_per_db)
            db_reports.append(rep)
            if "error" in rep:
                status = "ERROR"
            elif rep["ok"]:
                status = "OK"
            else:
                status = "MISSING"
            print(f"[{status}] {db_path.name}" + (f"  {rep['error']}" if "error" in rep else ""))

    result = {
        "archives": {
            k: {
                "kind": v["kind"],
                "size_gb": v["size_gb"],
                "coverage": {
                    log: sorted(sensors.keys())
                    for log, sensors in v["coverage"].items()
                },
            }
            for k, v in archive_index.items()
        },
        "dbs": db_reports,
        "summary": {
            "db_total": len(db_reports),
            "db_ok": sum(1 for r in db_reports if r["ok"]),
            "db_missing": sum(1 for r in db_reports if not r["ok"]),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    sys.exit(main())
