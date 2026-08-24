#!/usr/bin/env python
"""
适配外部 checkpoint（nuScenes / AV2）到 nuPlan 模型结构。

背景
----
nuScenes / AV2 官方 checkpoint 是 4 类（divider/ped_crossing/boundary/centerline），
nuPlan 只有 3 类（divider/ped_crossing/boundary）。类别顺序恰好前 3 个对齐，
因此可以截断 checkpoint 的分类头最后一层，让模型只输出 3 类（centerline 消失），
而不是整个分类头因 size mismatch 被跳过（导致 pred_inst 满额 50 的乱预测）。

同时处理其余结构差异：
  - cls_branches.*.6.{weight,bias}   : (4, C) -> (3, C)   类别截断
  - instance_embedding.weight        : (370, 512) -> (350, 512)  one2one 70->50
  - transformer.cams_embeds          : (6 或 7, 256) -> (8, 256)  相机嵌入填充

用法
----
python tools/adapt_external_ckpt.py \
    --input  ckpts/maptrv2_nusc_r50_24ep_w_centerline.pth \
    --output ckpts/maptrv2_nusc_adapted_nuplan.pth
"""
import argparse
import torch


def adapt(ckpt_path, out_path,
          target_num_cls=3,
          target_one2one=50, target_one2many=300,
          target_cams=8,
          src_one2one=70):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    changed = []

    # ---- 1. 分类头：4 类 -> 3 类（取前 3 维，centerline 是第 4 维）----
    for k in list(sd.keys()):
        if '.cls_branches.' not in k:
            continue
        if k.endswith('.weight') and sd[k].ndim == 2 and sd[k].shape[0] == 4:
            sd[k] = sd[k][:target_num_cls]
            changed.append(f'{k} {4}->{target_num_cls}')
        elif k.endswith('.bias') and sd[k].ndim == 1 and sd[k].shape[0] == 4:
            sd[k] = sd[k][:target_num_cls]
            changed.append(f'{k} {4}->{target_num_cls}')

    # ---- 2. instance_embedding：one2one 70->50（前 70 行是 one2one，后 300 行是 one2many）----
    key = 'pts_bbox_head.instance_embedding.weight'
    if key in sd:
        w = sd[key]
        if w.shape[0] != target_one2one + target_one2many:
            src_one2one = w.shape[0] - target_one2many  # 推断源 one2one 数
            new_w = torch.cat([w[:target_one2one], w[src_one2one:]], dim=0)
            sd[key] = new_w
            changed.append(f'{key} {w.shape[0]}->{new_w.shape[0]}')

    # ---- 3. cams_embeds：6/7 -> 8（新相机行用小随机初始化）----
    key = 'pts_bbox_head.transformer.cams_embeds'
    if key in sd and sd[key].shape[0] != target_cams:
        w = sd[key]
        n_extra = target_cams - w.shape[0]
        extra = torch.randn(n_extra, w.shape[1]) * 0.02
        sd[key] = torch.cat([w, extra], dim=0)
        changed.append(f'{key} {w.shape[0]}->{target_cams}')

    torch.save(ckpt, out_path)
    print(f'[OK] 保存到 {out_path}')
    if changed:
        print('改动项:')
        for c in changed:
            print(f'  - {c}')
    else:
        print('无改动（checkpoint 结构已匹配）')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='源 checkpoint')
    ap.add_argument('--output', required=True, help='适配后 checkpoint 保存路径')
    args = ap.parse_args()
    adapt(args.input, args.output)
