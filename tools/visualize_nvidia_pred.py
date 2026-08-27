#!/usr/bin/env python3
"""NVIDIA 预测可视化：加载 nuPlan 训练的 checkpoint，在 NVIDIA info 上推理，
把预测的矢量地图折线画在 BEV 图上（NVIDIA 无地图 GT，只画预测）。

用法:
    cd /data2/wyc/nuplan_maptrv2/MapTRV2
    PYTHONPATH=/data2/wyc/nuplan_maptrv2/MapTRV2 CUDA_VISIBLE_DEVICES=0 \
      python /data2/wyc/nuplan_maptrv2/tools/visualize_nvidia_pred.py \
        --info /data2/wyc/nuplan_maptrv2/data/infos/nvidia_map_infos_eval_sub.pkl \
        --checkpoint /data2/wyc/nuplan_maptrv2/work_dirs/full/latest.pth \
        --show-dir /data2/wyc/nuplan_maptrv2/reports/pred_vis_nvidia
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'MapTRV2'))
import projects.mmdet3d_plugin  # noqa: F401  触发注册
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.datasets import build_dataset

COLORS = ['red', 'green', 'blue', 'orange']
NAMES = ['divider', 'ped_crossing', 'boundary', 'centerline']
N_CLASSES = 4


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--info', required=True, help='NVIDIA info pkl')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', default=None,
                    help='训练配置（默认 nuplan_full）')
    ap.add_argument('--show-dir', default='reports/pred_vis_nvidia')
    ap.add_argument('--num-samples', type=int, default=6)
    ap.add_argument('--score-thresh', type=float, default=0.3)
    return ap.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg_path = args.config or osp.join(osp.dirname(__file__), '..', 'MapTRV2',
                                       'projects/configs/maptrv2/maptrv2_nuplan_full.py')
    cfg = Config.fromfile(cfg_path)
    if hasattr(cfg, 'plugin'):
        if hasattr(cfg, 'plugin_dir'):
            _module_path = os.path.dirname(cfg.plugin_dir).replace('/', '.')
            import importlib
            importlib.import_module(_module_path)
    cfg.model.pretrained = None

    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = args.info
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=0,
        dist=False, shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    print(f'dataset len: {len(dataset)}')

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cuda:0')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    pc_range = cfg.point_cloud_range
    show_dir = args.show_dir
    mmcv.mkdir_or_exist(osp.abspath(show_dir))

    count = 0
    for i, data in enumerate(data_loader):
        if count >= args.num_samples:
            break
        # 修复 img 维度（同 visualize_nuplan_pred.py）
        _imgs = []
        for _dc in data['img']:
            _t = _dc.data
            if isinstance(_t, list):
                _t = _t[0]
            if _t.dim() == 6:
                _t = _t[:, -1]
            _imgs.append(_t)
        data['img'] = _imgs

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        pred = result[0]['pts_bbox']
        scores = pred['scores_3d']
        labels = pred['labels_3d']
        pts = pred['pts_3d']
        keep = scores > args.score_thresh

        fig, ax = plt.subplots(1, 1, figsize=(7, 11))
        ax.set_xlim(pc_range[0], pc_range[3])
        ax.set_ylim(pc_range[1], pc_range[4])
        ax.set_aspect('equal')
        ax.set_title(f'NVIDIA MapTRv2 prediction  batch={i} (no GT)')
        ax.axhline(0, color='k', lw=0.5, alpha=0.3)
        ax.axvline(0, color='k', lw=0.5, alpha=0.3)

        for s, lab, p in zip(scores[keep], labels[keep], pts[keep]):
            lab = int(lab)
            if lab >= N_CLASSES:
                continue
            p = p.numpy()
            ls = '-.' if lab == 3 else '--'
            ax.plot(p[:, 0], p[:, 1], color=COLORS[lab], lw=2.8, ls=ls, alpha=1.0)

        out = osp.join(show_dir, f'nvidia_pred_{count:02d}_batch{i}.png')
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f'[{count}] batch={i} pred_inst={int(keep.sum())} saved {out}')
        count += 1

    print('DONE')


if __name__ == '__main__':
    main()
