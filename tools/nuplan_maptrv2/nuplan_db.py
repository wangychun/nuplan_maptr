"""nuPlan SQLite 数据库读取器（只读）。

提供：
- 日志元数据（location / map_version / vehicle / date）
- 相机标定（channel -> intrinsic / distortion / ego2cam / 图像尺寸）
- lidar 位姿
- 关键帧（lidar_pc 锚点）索引
- ego pose 查询
- 关键帧与相机图像的时间同步
"""
from __future__ import annotations

import pickle
import sqlite3
from typing import Dict, List, Optional

import numpy as np

from .coords import ego2cam_from_cam_pose, quat_trans_to_se3


def _unpickle(blob) -> object:
    return pickle.loads(blob)


class NuPlanDBReader:
    """只读打开一个 nuPlan 日志数据库。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        self._log = self._load_log()
        self._cameras = self._load_cameras()
        self._lidar = self._load_lidar()
        self._ego_pose_tokens = self._load_ego_pose_tokens()
        self._lidar_pc_frames = self._load_lidar_pc_frames()
        self._image_index = self._load_image_index()

    # ---------- 基础加载 ----------
    def _load_log(self) -> Dict:
        row = self.conn.execute("SELECT * FROM log LIMIT 1").fetchone()
        return {
            "token": row["token"],
            "vehicle_name": row["vehicle_name"],
            "date": row["date"],
            "logfile": row["logfile"],
            "location": row["location"],
            "map_version": row["map_version"],
            "db_path": self.db_path,
        }

    def _load_cameras(self) -> Dict[str, Dict]:
        cams = {}
        for row in self.conn.execute("SELECT * FROM camera"):
            intrinsic = np.asarray(_unpickle(row["intrinsic"]), dtype=np.float64)
            distortion = np.asarray(_unpickle(row["distortion"]), dtype=np.float64)
            cam_quat = np.asarray(_unpickle(row["rotation"]), dtype=np.float64)
            cam_trans = np.asarray(_unpickle(row["translation"]), dtype=np.float64)
            cams[row["channel"]] = {
                "token": row["token"],
                "channel": row["channel"],
                "model": row["model"],
                "intrinsic": intrinsic,
                "distortion": distortion,
                "cam_quat": cam_quat,      # 相机在 ego 坐标中的 wxyz
                "cam_translation": cam_trans,
                "ego2cam": ego2cam_from_cam_pose(cam_quat, cam_trans),
                "width": row["width"],
                "height": row["height"],
            }
        return cams

    def _load_lidar(self) -> Optional[Dict]:
        row = self.conn.execute("SELECT * FROM lidar LIMIT 1").fetchone()
        if row is None:
            return None
        quat = np.asarray(_unpickle(row["rotation"]), dtype=np.float64)
        trans = np.asarray(_unpickle(row["translation"]), dtype=np.float64)
        return {
            "token": row["token"],
            "channel": row["channel"],
            "model": row["model"],
            "quat": quat,
            "translation": trans,
            "ego2lidar": quat_trans_to_se3(quat, trans),
        }

    def _load_ego_pose_tokens(self) -> Dict[bytes, Dict]:
        out = {}
        for row in self.conn.execute(
            "SELECT token, timestamp, x, y, z, qw, qx, qy, qz, epsg, log_token FROM ego_pose"
        ):
            out[row["token"]] = {
                "timestamp": row["timestamp"],
                "x": row["x"], "y": row["y"], "z": row["z"],
                "qw": row["qw"], "qx": row["qx"], "qy": row["qy"], "qz": row["qz"],
                "epsg": row["epsg"],
                "log_token": row["log_token"],
            }
        return out

    def _load_lidar_pc_frames(self) -> List[Dict]:
        """以 lidar_pc 为关键帧锚点，按时间升序。"""
        rows = self.conn.execute(
            "SELECT token, timestamp, filename, scene_token, ego_pose_token "
            "FROM lidar_pc ORDER BY timestamp"
        ).fetchall()
        return [
            {
                "token": r["token"],
                "timestamp": r["timestamp"],
                "filename": r["filename"],
                "scene_token": r["scene_token"],
                "ego_pose_token": r["ego_pose_token"],
            }
            for r in rows
        ]

    def _load_image_index(self) -> Dict[str, List[Dict]]:
        """按相机通道建立按时间升序的图像索引。"""
        idx: Dict[str, List[Dict]] = {}
        cam_token_to_channel = {
            row["token"]: row["channel"]
            for row in self.conn.execute("SELECT token, channel FROM camera")
        }
        for row in self.conn.execute(
            "SELECT token, timestamp, filename_jpg, ego_pose_token, camera_token "
            "FROM image"
        ):
            ch = cam_token_to_channel.get(row["camera_token"])
            if ch is None:
                continue
            idx.setdefault(ch, []).append(
                {
                    "token": row["token"],
                    "timestamp": row["timestamp"],
                    "filename": row["filename_jpg"],
                    "camera_token": row["camera_token"],
                    "ego_pose_token": row["ego_pose_token"],
                    "channel": ch,
                }
            )
        for lst in idx.values():
            lst.sort(key=lambda x: x["timestamp"])
        return idx

    # ---------- 公开接口 ----------
    @property
    def log(self) -> Dict:
        return self._log

    @property
    def cameras(self) -> Dict[str, Dict]:
        return self._cameras

    @property
    def camera_channels(self) -> List[str]:
        return list(self._cameras.keys())

    @property
    def lidar(self) -> Optional[Dict]:
        return self._lidar

    @property
    def frames(self) -> List[Dict]:
        return self._lidar_pc_frames

    def ego_pose(self, token) -> Optional[Dict]:
        return self._ego_pose_tokens.get(token)

    def ego_pose_se3(self, pose: Dict) -> np.ndarray:
        q = [pose["qw"], pose["qx"], pose["qy"], pose["qz"]]
        t = [pose["x"], pose["y"], pose["z"]]
        return quat_trans_to_se3(q, t)

    def images_at_pose(self, ego_pose_token) -> List[Dict]:
        return self._image_index.get(ego_pose_token, [])

    def sync_images_for_frame(self, frame: Dict, channels: Optional[List[str]] = None,
                              max_dt_ns: int = 200_000_000) -> Dict[str, Dict]:
        """对关键帧做相机时间同步（按时间戳最近邻）。

        返回 {channel: {filename, timestamp, dt_ns, ...}}；超过 max_dt_ns 的通道丢弃。
        """
        from bisect import bisect_left

        channels = channels or list(self._cameras.keys())
        frame_ts = frame["timestamp"]
        out = {}
        for ch in channels:
            cands = self._image_index.get(ch, [])
            if not cands:
                continue
            ts_list = [c["timestamp"] for c in cands]
            pos = bisect_left(ts_list, frame_ts)
            best = None
            if pos < len(cands):
                best = cands[pos]
            if pos > 0:
                cand = cands[pos - 1]
                if best is None or abs(cand["timestamp"] - frame_ts) < abs(
                    best["timestamp"] - frame_ts
                ):
                    best = cand
            if best is None:
                continue
            dt = abs(best["timestamp"] - frame_ts)
            if dt > max_dt_ns:
                continue
            out[ch] = {**best, "dt_ns": dt}
        return out

    def close(self):
        self.conn.close()
