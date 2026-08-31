#!/usr/bin/env python3
"""合并多卡分片推理结果并统一评估（严格全局 Chamfer mAP）。

背景：test_evaluate.py 是单卡脚本（MMDataParallel + single_gpu_test）。
为用 8 卡加速，先分片推理（test_evaluate.py --shard-idx i --shard-total 8 --out shard_i.pkl），
本脚本把 N 份结果按全局 data_infos 顺序拼回，再用完整 dataset.evaluate 一次算全局指标。

用法:
    cd /data2/wyc/nuplan_maptrv2/MapTRV2
    PYTHONPATH=. python /data2/wyc/nuplan_maptrv2/tools/merge_eval_nuplan.py \
        --config projects/configs/maptrv2/maptrv2_nuplan_full.py \
        --ann-file /data2/wyc/nuplan_maptrv2/data/infos/nuplan_val_full_cam.pkl \
        --map-ann /data2/wyc/nuplan_maptrv2/data/infos/nuplan_map_anns_full_cam.json \
        --shards '/data2/wyc/nuplan_maptrv2/work_dirs/eval_shards/shard_*.pkl'
"""
from __future__ import annotations

import argparse
import glob
import os.path as osp
import pickle
import sys

from mmcv import Config
from mmdet3d.datasets import build_dataset

sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'MapTRV2'))
import projects.mmdet3d_plugin  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ann-file', required=True, help='完整 eval info pkl')
    ap.add_argument('--map-ann', required=True, help='GT map_ann json 路径（不存在会自动生成）')
    ap.add_argument('--shards', required=True, help='分片结果 pkl glob，如 work_dirs/eval_shards/shard_*.pkl')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin'):
        import importlib
        importlib.import_module(osp.dirname(cfg.plugin_dir).replace('/', '.'))
    cfg.model.pretrained = None

    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.map_ann_file = args.map_ann
    cfg.data.test.pop('samples_per_gpu', None)

    # 用完整 ann-file 构建 dataset（与单卡全量评估完全一致），data_infos 已按 timestamp 排序
    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True
    total = len(dataset)
    print(f'full dataset samples: {total}')

    shard_files = sorted(glob.glob(args.shards))
    if not shard_files:
        print(f'no shard files matched: {args.shards}')
        sys.exit(1)
    n = len(shard_files)
    print(f'loading {n} shards: {[osp.basename(s) for s in shard_files]}')

    # 交错切片放回：卡 i 对应 data_infos[i::n]，因此 merged[i::n] = shard_i
    merged = [None] * total
    for i, sf in enumerate(shard_files):
        with open(sf, 'rb') as f:
            res = pickle.load(f)
        if len(res) != len(range(i, total, n)):
            print(f'  WARN {sf}: {len(res)} results, expected {len(range(i, total, n))}')
        merged[i::n] = res
    missing = [j for j, r in enumerate(merged) if r is None]
    if missing:
        print(f'ERROR: {len(missing)} samples not covered, e.g. {missing[:5]}')
        sys.exit(1)
    print('merging done, all samples covered')

    eval_results = dataset.evaluate(merged, metric='chamfer')
    print('==== final Chamfer mAP ====')
    for k, v in eval_results.items():
        print(f'  {k}: {v:.4f}')


if __name__ == '__main__':
    sys.exit(main())
