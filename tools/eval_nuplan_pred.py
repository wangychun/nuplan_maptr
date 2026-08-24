#!/usr/bin/env python3
"""评估 nuPlan MapTRv2 预测质量：类别/分数分布 + 预测到 GT 的 Chamfer 距离。

用法:
    cd /data2/wyc/nuplan_maptrv2/MapTRV2
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python /data2/wyc/nuplan_maptrv2/tools/eval_nuplan_pred.py \
        projects/configs/maptrv2/maptrv2_nuplan_overfit.py \
        /data2/wyc/nuplan_maptrv2/work_dirs/overfit2/epoch_6.pth \
        --out /data2/wyc/nuplan_maptrv2/reports/pred_eval.json
"""
from __future__ import annotations

import argparse
import json
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

from shapely.geometry import LineString


def point_to_lines_dist(points, lines):
    """每个点到最近 GT 折线的距离均值。"""
    if not lines:
        return float('nan')
    d = np.inf
    for pt in points:
        p = LineString([(pt[0], pt[1]), (pt[0], pt[1])])
        m = min(p.distance(l) for l in lines)
        d = min(d, m)
    return float(d)


def chamfer_pred_to_gt(pred_pts, gt_lines):
    """pred 折线各点到最近 GT 折线的平均距离。"""
    if not gt_lines:
        return float('nan')
    dists = []
    for p in pred_pts:
        dists.append(min(LineString([(p[0], p[1]), (p[0], p[1])]).distance(l) for l in gt_lines))
    return float(np.mean(dists)) if dists else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--out', default='reports/pred_eval.json')
    ap.add_argument('--score-thresh', type=float, default=0.2)
    ap.add_argument('--num-samples', type=int, default=0)  # 0=全部
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
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    names = ['divider', 'ped_crossing', 'boundary']
    agg = {
        'n_samples': 0,
        'n_pred_total': 0,
        'n_gt_total': 0,
        'pred_cls': [0, 0, 0],
        'gt_cls': [0, 0, 0],
        'scores': [],
        'pred_lens': [],
        'chamfer_by_cls': [[] for _ in names],
    }

    for i, data in enumerate(data_loader):
        if args.num_samples and i >= args.num_samples:
            break
        gt_bboxes = data['gt_bboxes_3d'].data[0][0]
        gt_labels = data['gt_labels_3d'].data[0][0].tolist()
        if not (np.array(gt_labels) != -1).any():
            continue

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        pred = result[0]['pts_bbox']
        scores = pred['scores_3d']
        labels = pred['labels_3d']
        pts = pred['pts_3d']
        keep = scores > args.score_thresh

        # GT 折线按类
        gt_by_cls = {c: [] for c in range(3)}
        for inst, lab in zip(gt_bboxes.instance_list, gt_labels):
            if lab != -1:
                gt_by_cls[lab].append(inst)
        # pred 按类
        pred_by_cls = {c: [] for c in range(3)}
        for s, lab, p in zip(scores[keep], labels[keep], pts[keep]):
            lab = int(lab)
            if 0 <= lab < 3:
                pred_by_cls[lab].append(p.numpy())

        agg['n_samples'] += 1
        agg['n_pred_total'] += int(keep.sum())
        agg['n_gt_total'] += len(gt_labels)
        for c in range(3):
            agg['pred_cls'][c] += len(pred_by_cls[c])
            agg['gt_cls'][c] += len(gt_by_cls[c])
        agg['scores'].extend([float(s) for s in scores[keep]])
        for c in range(3):
            for p in pred_by_cls[c]:
                agg['pred_lens'].append(len(p))
                agg['chamfer_by_cls'][c].append(chamfer_pred_to_gt(p, gt_by_cls[c]))
        if i % 20 == 0:
            print(f'[{i}] samples={agg["n_samples"]} pred={agg["n_pred_total"]}')

    summary = {
        'n_samples': agg['n_samples'],
        'n_pred_total': agg['n_pred_total'],
        'n_gt_total': agg['n_gt_total'],
        'pred_per_sample': round(agg['n_pred_total'] / max(agg['n_samples'], 1), 2),
        'score_mean': round(float(np.mean(agg['scores'])), 3) if agg['scores'] else None,
        'score_min': round(float(np.min(agg['scores'])), 3) if agg['scores'] else None,
        'score_max': round(float(np.max(agg['scores'])), 3) if agg['scores'] else None,
        'pred_cls': dict(zip(names, agg['pred_cls'])),
        'gt_cls': dict(zip(names, agg['gt_cls'])),
        'pred_len_mean': round(float(np.mean(agg['pred_lens'])), 1) if agg['pred_lens'] else None,
        'chamfer_mean_by_cls': {
            names[c]: (round(float(np.nanmean(agg['chamfer_by_cls'][c])), 2)
                       if agg['chamfer_by_cls'][c] and not all(np.isnan(agg['chamfer_by_cls'][c])) else None)
            for c in range(3)
        },
        'chamfer_overall': round(float(np.nanmean([x for c in range(3) for x in agg['chamfer_by_cls'][c]])), 2)
        if any(agg['chamfer_by_cls'][c] for c in range(3)) else None,
        'score_thresh': args.score_thresh,
    }
    out = args.out
    mmcv.mkdir_or_exist(osp.dirname(osp.abspath(out)))
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
