#!/usr/bin/env python
"""分析 nuPlan GT / Pred 的 divider/boundary 语义是否合理。

核心思路：
- divider（车道分隔线）：应位于道路内部，两侧都有可行驶区域
- boundary（道路边界）：应位于可行驶区域（road_segments 并集）外沿

通过统计每条 GT/Pred 线段的几何特征来判断分类是否自洽。
"""
import argparse, os, sys, random
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin  # noqa: F401
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint', nargs='?', default=None)
    ap.add_argument('--ann', default='/data2/wyc/nuplan_maptrv2/data/nuplan/mini_infos_val.pkl')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num-samples', type=int, default=2)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    cfg.data.test.ann_file = args.ann

    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True
    random.seed(args.seed)
    if not hasattr(dataset, 'flag'):
        dataset.flag = np.zeros(len(dataset), dtype=np.uint8)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=0,
        dist=False, shuffle=True, nonshuffler_sampler=cfg.data.nonshuffler_sampler)

    if args.checkpoint:
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, args.checkpoint, map_location='cuda:0')
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
    else:
        model = None

    NAMES = ['divider', 'ped_crossing', 'boundary']
    count = 0
    for i, data in enumerate(data_loader):
        if count >= args.num_samples:
            break
        gt_bboxes = data['gt_bboxes_3d'].data[0][0]
        gt_labels = data['gt_labels_3d'].data[0][0]
        if not (gt_labels != -1).any():
            continue

        _imgs = []
        for _dc in data['img']:
            _t = _dc.data
            if isinstance(_t, list):
                _t = _t[0]
            if _t.dim() == 6:
                _t = _t[:, -1]
            _imgs.append(_t)
        data['img'] = _imgs

        # GT 分析
        print(f'\n========== batch {count} ==========')
        gt_np = gt_labels.numpy()
        for c in range(3):
            idx = np.where(gt_np == c)[0]
            n = len(idx)
            if n == 0:
                print(f'  GT {NAMES[c]:14s}: 0 条')
                continue
            stats = []
            for j in idx:
                inst = gt_bboxes.instance_list[j]
                pts = np.array(list(inst.coords))
                span = float(np.linalg.norm(pts[0] - pts[-1]))
                # 相对 ego(0,0) 的平均距离 -> 粗略判断是否在道路中心区域
                dist = float(np.mean(np.linalg.norm(pts, axis=1)))
                stats.append((span, dist))
            spans = [s for s, _ in stats]
            dists = [d for _, d in stats]
            print(f'  GT {NAMES[c]:14s}: {n} 条 | 跨度 {min(spans):.1f}~{max(spans):.1f} | '
                  f'距ego {min(dists):.1f}~{max(dists):.1f}')

        # Pred 分析
        if model is None:
            count += 1
            continue
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        pred = result[0]['pts_bbox']
        pred_pts = pred['pts_3d']
        pred_labels = pred['labels_3d'].numpy()
        pred_scores = pred['scores_3d'].numpy()
        print(f'  --- 预测（阈值 0.15）---')
        for c in range(3):
            mask = pred_labels == c
            n = int(mask.sum())
            if n == 0:
                print(f'  PR {NAMES[c]:14s}: 0 条')
                continue
            stats = []
            for pts, sc in zip(pred_pts[mask], pred_scores[mask]):
                pts = pts.numpy()
                span = float(np.linalg.norm(pts[0] - pts[-1]))
                dist = float(np.mean(np.linalg.norm(pts, axis=1)))
                stats.append((span, dist, float(sc)))
            spans = [s for s, _, _ in stats]
            dists = [d for _, d, _ in stats]
            scs = [s for _, _, s in stats]
            print(f'  PR {NAMES[c]:14s}: {n} 条 | 跨度 {min(spans):.1f}~{max(spans):.1f} | '
                  f'距ego {min(dists):.1f}~{max(dists):.1f} | 得分 {min(scs):.2f}~{max(scs):.2f}')
        count += 1


if __name__ == '__main__':
    import torch
    main()
