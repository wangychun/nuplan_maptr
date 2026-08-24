#!/usr/bin/env python3
"""诊断投影问题：打印 lidar2img、GT 坐标、pred 坐标、投影后像素范围。"""
from __future__ import annotations
import argparse
import os.path as osp
import sys

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'MapTRV2'))
import projects.mmdet3d_plugin  # noqa
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.datasets import build_dataset


def project(pts, lidar2img):
    ones = np.ones((len(pts), 1))
    homo = np.concatenate([pts, np.zeros((len(pts), 1)), ones], axis=1)  # z=0
    cam = (lidar2img @ homo.T).T
    valid = cam[:, 2] > 0
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
    ap.add_argument('--cam', default='CAM_F0')
    ap.add_argument('--score-thresh', type=float, default=0.15)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin'):
        import importlib
        importlib.import_module(osp.dirname(cfg.plugin_dir).replace('/', '.'))
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)

    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True
    data_loader = build_dataloader(
        dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=0,
        dist=False, shuffle=False, nonshuffler_sampler=cfg.data.nonshuffler_sampler)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    for i, data in enumerate(data_loader):
        im_meta = data['img_metas']
        while hasattr(im_meta, 'data') or isinstance(im_meta, list):
            im_meta = im_meta.data if hasattr(im_meta, 'data') else im_meta[0]
            if isinstance(im_meta, dict) and 0 in im_meta:
                im_meta = im_meta[0]
                break
        metas = im_meta
        filenames = metas['filename']
        cam_idx = [j for j, f in enumerate(filenames) if '/' + args.cam + '/' in f][0]
        lidar2img = np.asarray(metas['lidar2img'][cam_idx])
        img_shape = metas['img_shape'][cam_idx] if 'img_shape' in metas else None
        print(f'===== batch {i} cam={args.cam} img_shape={img_shape} =====')
        print('lidar2img:\n', np.round(lidar2img, 2))

        # GT 坐标范围
        gt_bboxes = data['gt_bboxes_3d'].data[0][0]
        gt_labels = data['gt_labels_3d'].data[0][0]
        all_gt = np.concatenate([np.array(list(inst.coords)) for inst in gt_bboxes.instance_list]) if len(gt_bboxes.instance_list) else np.zeros((0, 2))
        print(f'GT insts={len(gt_bboxes.instance_list)} pts range x[{all_gt[:,0].min():.1f},{all_gt[:,0].max():.1f}] y[{all_gt[:,1].min():.1f},{all_gt[:,1].max():.1f}]')

        # GT 投影像素范围
        gt_pix = [project(np.array(list(inst.coords)), lidar2img)[0] for inst in gt_bboxes.instance_list]
        gt_pix = np.concatenate([p[~np.isnan(p[:,0])] for p in gt_pix]) if gt_pix else np.zeros((0, 2))
        if len(gt_pix):
            print(f'GT pix range u[{gt_pix[:,0].min():.1f},{gt_pix[:,0].max():.1f}] v[{gt_pix[:,1].min():.1f},{gt_pix[:,1].max():.1f}]')
        else:
            print('GT pix: none valid')

        # 推理
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        pred = result[0]['pts_bbox']
        scores = pred['scores_3d'].numpy()
        labels = pred['labels_3d'].numpy()
        pts = pred['pts_3d'].numpy()
        keep = scores > args.score_thresh
        print(f'pred total={len(scores)} keep={int(keep.sum())} scores range [{scores.min():.3f},{scores.max():.3f}]')

        if keep.sum() > 0:
            pk = pts[keep]
            print(f'pred pts shape={pk.shape} range x[{pk[:,:,0].min():.1f},{pk[:,:,0].max():.1f}] y[{pk[:,:,1].min():.1f},{pk[:,:,1].max():.1f}]')
            pred_pix = [project(p, lidar2img)[0] for p in pk]
            pred_pix = np.concatenate([p[~np.isnan(p[:,0])] for p in pred_pix])
            if len(pred_pix):
                print(f'pred pix range u[{pred_pix[:,0].min():.1f},{pred_pix[:,0].max():.1f}] v[{pred_pix[:,1].min():.1f},{pred_pix[:,1].max():.1f}]')
            else:
                print('pred pix: none valid')
        break


if __name__ == '__main__':
    main()
