"""nuPlan 坐标工具。

约定（与 nuplan-devkit 一致）：
- vehicle/ego 坐标系：ISO 8855，x 前、y 左、z 上。
- 全局坐标系：EPSG 投影平面坐标（米），由 ego_pose.epsg 标注。
- Rotation 四元数顺序为 [w, x, y, z]。
- camera.translation/rotation 描述相机在 vehicle 坐标系中的位姿 T_ego_cam。
- ego_pose 描述 vehicle 在全局坐标系中的位姿 T_global_ego。
"""
from __future__ import annotations

import numpy as np


def quat_to_rotmat(q) -> np.ndarray:
    """wxyz 四元数 -> 3x3 旋转矩阵 (numpy float64)。"""
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 -> wxyz 四元数。"""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def build_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """旋转矩阵 + 平移 -> 4x4 齐次矩阵。"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def quat_trans_to_se3(q, t) -> np.ndarray:
    """(wxyz 四元数, 平移) -> 4x4 齐次矩阵。"""
    return build_se3(quat_to_rotmat(q), np.asarray(t, dtype=np.float64))


def ego2cam_from_cam_pose(cam_quat, cam_translation) -> np.ndarray:
    """由相机在 ego 坐标中的位姿 (T_ego_cam) 构建 ego->cam 4x4。

    T_ego_cam = [R_cam | t_cam]，把相机坐标点变换到 ego 坐标。
    ego2cam = T_ego_cam^{-1} = [R_cam^T | -R_cam^T t_cam]
    """
    R_cam = quat_to_rotmat(cam_quat)
    t_cam = np.asarray(cam_translation, dtype=np.float64)
    R = R_cam.T
    t = -R @ t_cam
    return build_se3(R, t)


def project_lidar_to_img(points_ego: np.ndarray, ego2cam: np.ndarray, K: np.ndarray):
    """把 ego 坐标 3D 点投影到图像像素坐标。返回 (N,2) 像素与有效 mask。"""
    pts = np.asarray(points_ego, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homo = np.hstack([pts, ones])  # N x 4
    cam = (ego2cam @ homo.T).T  # N x 4
    valid = cam[:, 2] > 0
    pix = np.zeros((pts.shape[0], 2), dtype=np.float64)
    if np.any(valid):
        xyz = cam[valid, :3]
        uv = (K @ xyz.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        pix[valid] = uv
    return pix, valid


def verify_inverse(T: np.ndarray, tol: float = 1e-9) -> float:
    """验证 T 与 inv(T) 乘积误差（Frobenius 范数）。"""
    err = np.linalg.norm(T @ np.linalg.inv(T) - np.eye(4))
    return float(err)


def yaw_from_rotmat(R: np.ndarray) -> float:
    """从旋转矩阵提取偏航角（弧度），约等于 nuScenes quaternion_yaw。"""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def patch_angle_from_ego(ego_quat) -> float:
    """MapTRV2 使用的 patch 角度（度，0~360）。"""
    R = quat_to_rotmat(ego_quat)
    yaw = yaw_from_rotmat(R)
    ang = np.degrees(yaw)
    if ang < 0:
        ang += 360.0
    return ang
