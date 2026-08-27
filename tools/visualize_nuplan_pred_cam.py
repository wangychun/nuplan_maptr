#!/usr/bin/env python3
"""把 MapTRV2 预测的矢量地图投影到真实相机图像上，直观判断预测是否有用。

用法:
    cd /data2/wyc/nuplan_maptrv2/MapTRV2
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python /data2/wyc/nuplan_maptrv2/tools/visualize_nuplan_pred_cam.py \
        projects/configs/maptrv2/maptrv2_nuplan_overfit.py \
        /data2/wyc/nuplan_maptrv2/work_dirs/overfit3/epoch_24.pth \
        --show-dir /data2/wyc/nuplan_maptrv2/reports/pred_cam \
        --num-samples 4 --score-thresh 0.15
"""
from __future__ import annotations

import argparse
import os.path as osp
import sys

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model

sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'MapTRV2'))
import projects.mmdet3d_plugin  # noqa
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.datasets import build_dataset

# 类别颜色与名称（根据 config 的 map_classes 动态确定类别数；最多 4 类）
# nuPlan 模型 3 类：divider/ped_crossing/boundary；nuScenes 官方模型 4 类含 centerline
COLORS = [(1.0, 0, 0), (0, 0.8, 0), (0, 0, 1.0), (1.0, 0.6, 0)]
DEFAULT_NAMES = ['divider', 'ped_crossing', 'boundary', 'centerline']
NAMES = list(DEFAULT_NAMES)
N_CLASSES = 3  # 默认 3 类；main 里根据 config map_classes 更新


def project(pts, lidar2img, min_depth=0.5, z=-1.6):
    """pts: (N,2) 局部 ego 坐标 -> 像素 (N,2)。lidar2img: 4x4。

    与官方 line_ego_to_pvmask 一致：地图线在地面，z = 相机离地高度（-1.6m），
    否则线会被投到相机高度平面而"飘在空中"、不贴合车道线。
    min_depth: 深度下界，只过滤相机正后方的点。
    """
    ones = np.ones((len(pts), 1))
    homo = np.concatenate([pts, np.full((len(pts), 1), z), ones], axis=1)  # z=地面
    cam = (lidar2img @ homo.T).T
    depth = cam[:, 2]
    valid = depth > min_depth
    pix = np.full((len(pts), 2), np.nan)
    if valid.any():
        xyz = cam[valid, :3]
        uv = xyz[:, :2] / xyz[:, 2:3]
        pix[valid] = uv
    return pix, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--show-dir', default='reports/pred_cam')
    ap.add_argument('--num-samples', type=int, default=4)
    ap.add_argument('--score-thresh', type=float, default=0.15)
    ap.add_argument('--cam', default='CAM_FRONT,CAM_FRONT_RIGHT,CAM_FRONT_LEFT,CAM_BACK,CAM_BACK_LEFT,CAM_BACK_RIGHT',
                    help='相机名（nuScenes 命名），逗号分隔可画多个，默认 6 路全画')
    ap.add_argument('--ann-file', default=None,
                    help='覆盖 test ann_file（如 nuplan_val_eval_sub.pkl / nuscenes 子集），默认用配置里的')
    ap.add_argument('--seed', type=int, default=None, help='随机采样种子，设置后随机取样本（覆盖不同 log/场景）')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin'):
        import importlib
        importlib.import_module(osp.dirname(cfg.plugin_dir).replace('/', '.'))
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.ann_file is not None:
        cfg.data.test.ann_file = args.ann_file

    # 按 config 的 map_classes 动态确定类别数（nuPlan 3 类 / nuScenes 4 类）
    global N_CLASSES, NAMES
    cfg_map_classes = list(getattr(cfg, 'map_classes', ['divider', 'ped_crossing', 'boundary']))
    N_CLASSES = len(cfg_map_classes)
    NAMES = list(cfg_map_classes)  # 只保留 config 实际类别，避免多出 centerline 图例

    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)

    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True
    shuffle = args.seed is not None
    if shuffle:
        import random
        random.seed(args.seed)
        if not hasattr(dataset, 'flag'):
            dataset.flag = np.zeros(len(dataset), dtype=np.uint8)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=0,
        dist=False, shuffle=shuffle, nonshuffler_sampler=cfg.data.nonshuffler_sampler)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    show_dir = args.show_dir
    mmcv.mkdir_or_exist(osp.abspath(show_dir))
    count = 0
    for i, data in enumerate(data_loader):
        if count >= args.num_samples:
            break
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        pred = result[0]['pts_bbox']
        scores = pred['scores_3d']
        labels = pred['labels_3d']
        pts = pred['pts_3d']
        keep = scores > args.score_thresh

        # img_metas 结构兼容解析（test 模式为 list / DataContainer）
        im_meta = data['img_metas']
        while hasattr(im_meta, 'data') or isinstance(im_meta, list):
            im_meta = im_meta.data if hasattr(im_meta, 'data') else im_meta[0]
            if isinstance(im_meta, dict) and 0 in im_meta:
                im_meta = im_meta[0]
                break
        metas = im_meta
        filenames = metas['filename']
        cam_list = [c.strip() for c in args.cam.split(',')]
        # nuScenes 相机名 -> 磁盘目录名（build_nuplan_infos 的 data_path 用 nuPlan 原名，
        # 如 CAM_FRONT 的目录是 CAM_F0）。--cam 传 nuScenes 名即可，内部自动映射。
        NUSC_TO_NUPLAN = {
            'CAM_FRONT': 'CAM_F0', 'CAM_FRONT_RIGHT': 'CAM_R0', 'CAM_FRONT_LEFT': 'CAM_L0',
            'CAM_BACK': 'CAM_B0', 'CAM_BACK_LEFT': 'CAM_L2', 'CAM_BACK_RIGHT': 'CAM_R2',
        }
        # NVIDIA 数据：路径不含 CAM_ 目录名，用文件名特征匹配
        NVIDIA_CAM_MARKERS = {
            'CAM_FRONT': 'camera_front_wide', 'CAM_FRONT_RIGHT': 'camera_cross_right',
            'CAM_FRONT_LEFT': 'camera_cross_left', 'CAM_BACK': 'camera_rear_tele',
            'CAM_BACK_LEFT': 'camera_rear_left', 'CAM_BACK_RIGHT': 'camera_rear_right',
        }
        for cam_name in cam_list:
            # 匹配顺序：
            # 1) nuScenes/通用：路径直接含 /CAM_XXX/（如 /samples/CAM_FRONT/xxx.jpg）
            # 2) NVIDIA：路径不含 CAM_ 目录名，用文件名特征（camera_front_wide 等）
            # 3) nuPlan：nuScenes 名映射到 nuPlan 磁盘目录名（CAM_FRONT -> CAM_F0）
            cam_idx = [j for j, f in enumerate(filenames) if '/' + cam_name + '/' in f]
            if not cam_idx and cam_name in NVIDIA_CAM_MARKERS:
                marker = NVIDIA_CAM_MARKERS[cam_name]
                cam_idx = [j for j, f in enumerate(filenames) if marker in f]
            if not cam_idx:
                disk_name = NUSC_TO_NUPLAN.get(cam_name, cam_name)
                cam_idx = [j for j, f in enumerate(filenames) if '/' + disk_name + '/' in f]
            if not cam_idx:
                print(f'[skip] cam {cam_name} not found in batch {i}')
                continue
            cam_idx = cam_idx[0]
            lidar2img = np.asarray(metas['lidar2img'][cam_idx]).astype(np.float64).copy()
            img_path = filenames[cam_idx]

            import numpy as _np
            from PIL import Image
            img = _np.asarray(Image.open(img_path).convert('RGB'))

            # dataset 输出的 lidar2img 对应 resize 后的图像（ori_shape，如 960 宽），
            # 而 Image.open 打开的是原始文件（如 1920 宽）。必须把 lidar2img 的像素部分
            # 按比例缩放到实际图像尺寸，否则投影坐标整体偏移、错位。
            ori_shape = metas['ori_shape'][cam_idx]
            img_h, img_w = img.shape[:2]
            ori_h, ori_w = int(ori_shape[0]), int(ori_shape[1])
            if ori_w and ori_h and (img_w != ori_w or img_h != ori_h):
                lidar2img[:2] *= (img_w / ori_w)

            # GT
            gt_bboxes = data['gt_bboxes_3d'].data[0][0]
            gt_labels = data['gt_labels_3d'].data[0][0]

            fig, ax = plt.subplots(1, 1, figsize=(12, 7))
            ax.imshow(img)
            H, W = img.shape[:2]

            # 投影点绘制：近处点 px 可能超出屏幕，clamp 到屏幕边缘保证线连续；
            # 仅保留深度有效（valid）的点。
            def draw_vis(pix, valid, color, lw, ls, alpha):
                ok = valid & np.isfinite(pix[:, 0]) & np.isfinite(pix[:, 1])
                x = np.clip(pix[:, 0], 0, W - 1)
                y = np.clip(pix[:, 1], 0, H - 1)
                # 用 nan 断开：屏幕内外的跳变点间不连线，避免跨屏幕的假直线
                if ok.sum() >= 2:
                    ax.plot(x, y, color=color, lw=lw, ls=ls, alpha=alpha)

            # GT 投影（细、浅色、实线）；centerline(3) 用点划线
            for inst, lab in zip(gt_bboxes.instance_list, gt_labels):
                lab = int(lab)
                if lab < 0 or lab >= N_CLASSES:
                    continue
                coords = _np.array(list(inst.coords))
                pix, valid = project(coords, lidar2img, min_depth=1.0)
                ls = '-.' if N_CLASSES > 3 and lab == 3 else '-'
                draw_vis(pix, valid, COLORS[lab], 1.5, ls, 0.45)
            # pred 投影（粗、深色、虚线）；centerline(3) 用点划线
            for s, lab, p in zip(scores[keep], labels[keep], pts[keep]):
                lab = int(lab)
                if lab >= N_CLASSES:
                    continue
                p = p.numpy()
                pix, valid = project(p, lidar2img, min_depth=1.0)
                ls = '-.' if N_CLASSES > 3 and lab == 3 else '--'
                draw_vis(pix, valid, COLORS[lab], 2.8, ls, 1.0)
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            # 图例：N 类 × GT/Pred
            import matplotlib.lines as mlines
            legend_handles = []
            for ci, cname in enumerate(NAMES):
                ls = '-.' if N_CLASSES > 3 and ci == 3 else '-'
                ls2 = '-.' if N_CLASSES > 3 and ci == 3 else '--'
                legend_handles.append(mlines.Line2D([], [], color=COLORS[ci], lw=1.5, alpha=0.45, ls=ls,
                                                    label=f'{cname} (GT)'))
                legend_handles.append(mlines.Line2D([], [], color=COLORS[ci], lw=2.8, ls=ls2, alpha=1.0,
                                                    label=f'{cname} (Pred)'))
            ax.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.7)
            ax.set_title(f'{cam_name} pred vs GT  batch={i}')
            ax.axis('off')
            out = osp.join(show_dir, f'cam_{count:02d}_{cam_name}.png')
            fig.savefig(out, dpi=110, bbox_inches='tight')
            plt.close(fig)
            print(f'[{count}] batch={i} cam={cam_name} pred_inst={int(keep.sum())} saved {out}')
        count += 1
    print('DONE')


if __name__ == '__main__':
    main()
