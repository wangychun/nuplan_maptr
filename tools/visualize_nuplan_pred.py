#!/usr/bin/env python3
"""用训练好的 checkpoint 验证 nuPlan 矢量地图预测：BEV 对比图（GT vs 预测）。

用法:
    cd /data2/wyc/nuplan_maptrv2/MapTRV2
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python /data2/wyc/nuplan_maptrv2/tools/visualize_nuplan_pred.py \
        /data2/wyc/nuplan_maptrv2/MapTRV2/projects/configs/maptrv2/maptrv2_nuplan_overfit.py \
        /data2/wyc/nuplan_maptrv2/work_dirs/overfit2/epoch_6.pth \
        --show-dir /data2/wyc/nuplan_maptrv2/reports/pred_vis \
        --num-samples 6 --score-thresh 0.3
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor

sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'MapTRV2'))
import projects.mmdet3d_plugin  # noqa: F401  触发注册
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.datasets import build_dataset

# 类别颜色与名称（根据 config 的 map_classes 动态确定类别数；最多 4 类）
COLORS = ['red', 'green', 'blue', 'orange']
DEFAULT_NAMES = ['divider', 'ped_crossing', 'boundary', 'centerline']
NAMES = list(DEFAULT_NAMES)
N_CLASSES = 3  # 默认 3 类；main 里根据 config map_classes 更新


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='test config')
    ap.add_argument('checkpoint', help='checkpoint file')
    ap.add_argument('--show-dir', default='reports/pred_vis')
    ap.add_argument('--num-samples', type=int, default=6)
    ap.add_argument('--score-thresh', type=float, default=0.3)
    ap.add_argument('--ann-file', default=None,
                    help='覆盖 test ann_file（如 nuscenes 小子集 info），默认用配置里的')
    ap.add_argument('--seed', type=int, default=None, help='随机采样种子，设置后随机取样本（覆盖不同 log/场景）')
    return ap.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin'):
        if hasattr(cfg, 'plugin_dir'):
            _module_dir = os.path.dirname(cfg.plugin_dir)
            _module_path = _module_dir.replace('/', '.')
            import importlib
            importlib.import_module(_module_path)
    cfg.model.pretrained = None

    # 按 config 的 map_classes 动态确定类别数（nuPlan 3 类 / nuScenes 4 类）
    global N_CLASSES, NAMES
    cfg_map_classes = list(getattr(cfg, 'map_classes', ['divider', 'ped_crossing', 'boundary']))
    N_CLASSES = len(cfg_map_classes)
    NAMES = list(cfg_map_classes)  # 只保留 config 实际类别，避免多出 centerline 图例

    cfg.data.test.test_mode = True
    if args.ann_file is not None:
        cfg.data.test.ann_file = args.ann_file
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True  # 测试时也生成 GT 向量
    shuffle = args.seed is not None
    if shuffle:
        import random
        random.seed(args.seed)
        if not hasattr(dataset, 'flag'):
            dataset.flag = np.zeros(len(dataset), dtype=np.uint8)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=0,
        dist=False, shuffle=shuffle,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    print(f'dataset len: {len(dataset)}')

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 推理保持全精度：fp16 下部分算子（如 unfold）报 "not implemented for 'Half'"
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
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0][0]
        gt_labels_3d = data['gt_labels_3d'].data[0][0]
        if not (gt_labels_3d != -1).any():
            print(f'[skip] empty gt batch {i}')
            continue

        # 修复 img 维度：data['img'] 是 [DC]，DC.data 是 [B,N,C,H,W] 的 5 维 tensor，
        # 去掉 DC 包装直接喂 5 维（否则模型 forward 时会被包成 6 维导致 backbone 报错）
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
        ax.set_title(f'nuPlan MapTRv2 pred vs GT  batch={i}')
        ax.axhline(0, color='k', lw=0.5, alpha=0.3)
        ax.axvline(0, color='k', lw=0.5, alpha=0.3)

        # GT 实线（细、浅色）；centerline(3) 用点划线样式
        for inst, lab in zip(gt_bboxes_3d.instance_list, gt_labels_3d):
            lab = int(lab)
            if lab < 0 or lab >= N_CLASSES:
                continue
            coords = np.array(list(inst.coords))
            ls = '-.' if N_CLASSES > 3 and lab == 3 else '-'
            ax.plot(coords[:, 0], coords[:, 1], color=COLORS[lab], lw=1.5, alpha=0.45, ls=ls)
        # pred 虚线（粗、深色）；centerline(3) 用点划样式
        for s, lab, p in zip(scores[keep], labels[keep], pts[keep]):
            lab = int(lab)
            if lab >= N_CLASSES:
                continue
            p = p.numpy()
            ls = '-.' if N_CLASSES > 3 and lab == 3 else '--'
            ax.plot(p[:, 0], p[:, 1], color=COLORS[lab], lw=2.8, ls=ls, alpha=1.0)
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
        out = osp.join(show_dir, f'pred_{count:02d}_batch{i}.png')
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f'[{count}] batch={i} pred_inst={int(keep.sum())} saved {out}')
        count += 1

    print('DONE')


if __name__ == '__main__':
    main()
