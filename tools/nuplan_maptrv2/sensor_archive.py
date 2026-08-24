"""按需从 nuPlan sensor blob 分卷归档读取相机图像与点云。

不将 ~1.1TB 传感器全部解压，而是通过归档成员名建立
log/sensor -> 归档 的映射，按需读取单个文件字节。

注意：nuPlan 官方 sensor blob 是 tar 格式，但文件名以 .zip 结尾
（tar 顶层目录名 = 去 .zip 后缀的文件名），open_archive 按文件头自动识别。
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from zipfile import ZipFile


class _TarArchive:
    """把未压缩 tar 包装成与 ZipFile 兼容的只读接口。"""

    def __init__(self, path):
        self._tf = tarfile.open(path, "r:")
        # tar 无中央目录，getnames 需顺序遍历一次，提前缓存
        self._names = self._tf.getnames()

    def namelist(self):
        return self._names

    def read(self, name):
        f = self._tf.extractfile(name)
        if f is None:
            raise KeyError(name)
        return f.read()

    def close(self):
        self._tf.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_archive(path):
    """按文件头自动识别 zip 或 tar（sensor blob 是 .zip 命名的 tar）。"""
    path = str(path)
    if tarfile.is_tarfile(path):
        return _TarArchive(path)
    return ZipFile(path)


class SensorBlobStore:
    """基于分卷 zip 的只读传感器存储。"""

    def __init__(self, blobs_root: str, index_path: str):
        self.blobs_root = Path(blobs_root)
        # {zip_name: {"kind": str, "coverage": {log: {sensor: count}}}}
        raw = json.loads(Path(index_path).read_text())
        self.archives = raw["archives"]
        # 为快速定位 log/sensor -> zip
        self._lookup: dict = {}
        for zname, meta in self.archives.items():
            for log, sensors in meta.get("coverage", {}).items():
                for sensor in sensors:
                    self._lookup.setdefault((log, sensor), zname)
        self._zf_cache: dict = {}

    @classmethod
    def from_scan(cls, blobs_root: str, index_path: str) -> "SensorBlobStore":
        return cls(blobs_root, index_path)

    def _zip_for(self, log: str, sensor: str):
        zname = self._lookup.get((log, sensor))
        if zname is None:
            raise FileNotFoundError(f"no archive for {log}/{sensor}")
        zf = self._zf_cache.get(zname)
        if zf is None:
            zf = open_archive(self.blobs_root / zname)
            self._zf_cache[zname] = zf
        return zf, zname

    def read(self, rel_path: str) -> bytes:
        """按 DB 中的相对路径 <log>/<sensor>/<file> 读取文件字节。"""
        parts = rel_path.split("/")
        if len(parts) < 3:
            raise ValueError(f"bad rel path: {rel_path}")
        log, sensor = parts[0], parts[1]
        zf, zname = self._zip_for(log, sensor)
        inner = f"{zname[:-4]}/{rel_path}"
        return zf.read(inner)

    def read_image(self, rel_path: str) -> "io.BytesIO":
        return io.BytesIO(self.read(rel_path))

    def has(self, rel_path: str) -> bool:
        try:
            parts = rel_path.split("/")
            if len(parts) < 3:
                return False
            zname = self._lookup.get((parts[0], parts[1]))
            if zname is None:
                return False
            zf = self._zf_cache.get(zname) or open_archive(self.blobs_root / zname)
            self._zf_cache[zname] = zf
            inner = f"{zname[:-4]}/{rel_path}"
            return inner in zf.namelist()
        except Exception:
            return False
