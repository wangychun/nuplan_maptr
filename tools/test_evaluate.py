#!/usr/bin/env python3
"""快速验证 NuPlanMapDataset 的 Chamfer 评测链路（推理 -> evaluate -> mAP）。

用法:
    cd /data2/wyc/nuplan_maptrv2/MapTRV2
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python /data2/wyc/nuplan_maptrv2/tools/test_evaluate.py \
        projects/configs/maptrv2/maptrv2_nuplan_mini.py \
        /data2/wyc/nuplan_maptrv2/work_dirs/overfit3/epoch_24.pth \
        --ann-file /data2/wyc/nuplan_maptrv2/data/infos/nuplan_map_infos_overfit.pkl \
        --map-ann /data2/wyc/nuplan_maptrv2/data/infos/nuplan_map_anns_test.json
"""
from __future__ import annotations

import argparse
import os.path as osp
import sys

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.models import build_model

sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'MapTRV2'))
import projects.mmdet3d_plugin  # noqa
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.datasets import build_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--ann-file', required=True)
    ap.add_argument('--map-ann', default='/data2/wyc/nuplan_maptrv2/data/infos/nuplan_map_anns_test.json')
    ap.add_argument('--max-samples', type=int, default=0)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin'):
        import importlib
        importlib.import_module(osp.dirname(cfg.plugin_dir).replace('/', '.'))
    cfg.model.pretrained = None

    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.map_ann_file = args.map_ann
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)

    dataset = build_dataset(cfg.data.test)
    if args.max_samples:
        dataset.data_infos = dataset.data_infos[: args.max_samples]
    dataset.is_vis_on_test = True
    data_loader = build_dataloader(
        dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=2,
        dist=False, shuffle=False, nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    print(f'test samples: {len(dataset)}')

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    print('running inference ...')
    results = single_gpu_test(model, data_loader, show=False)
    print('evaluating ...')
    eval_results = dataset.evaluate(results, metric='chamfer')
    for k, v in eval_results.items():
        print(f'  {k}: {v:.4f}')


if __name__ == '__main__':
    main()
