#!/usr/bin/env python3
"""CUDA memory/gradient preflight for the single-match training graph."""

import argparse
import math
import time

import torch

from smoke_pqmax import build_model, parse_shape
from train import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='train_config_0825_single_match_flow.yaml')
    parser.add_argument(
        '--restore_ckpt',
        default='ckpt/0729_lc_v3s2/0729_lc_v3s2_800.pkl')
    parser.add_argument(
        '--shape', action='append', type=parse_shape,
        help='HEIGHTxWIDTHxBATCH; repeat for multiple phase shapes')
    parser.add_argument('--max_reserved_gib', type=float, default=29.5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA不可用，无法执行single-match显存preflight')

    shapes = args.shape or [(320, 576, 4), (336, 608, 4)]
    config = load_config(args.config)
    model = build_model(config)
    model.load_model(args.restore_ckpt, resume=False)
    model.set_local_enabled(True)
    amp_name = str(config['optim'].get('amp_dtype', 'bf16')).lower()
    amp_dtype = torch.bfloat16 if amp_name == 'bf16' else torch.float16
    use_amp = bool(config['optim'].get('amp', True))
    lr = float(config['optim']['lr_max'])

    total = sum(parameter.numel() for parameter in model.net.parameters())
    print(f'[single-match preflight] parameters={total:,} amp={amp_name}')
    for height, width, batch in shapes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        images = torch.rand(batch, 6, height, width, device='cuda')
        target = torch.rand(batch, 3, height, width, device='cuda')
        timestep = torch.full((batch, 1, 1, 1), 0.5, device='cuda')
        flow_gt = torch.randn(batch, 5, height, width, device='cuda')
        flow_gt[:, 4] = 1.0
        has_mv = torch.ones(batch, device='cuda')
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with torch.autocast(
                    'cuda', dtype=amp_dtype, enabled=use_amp):
                prediction, loss, _ = model.update(
                    images, target, timestep=timestep,
                    learning_rate=lr, training=True, loss_step=5000,
                    flow_gt=flow_gt, has_mv=has_mv)
            torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print('[single-match preflight] OOM '
                  f'shape={height}x{width} batch={batch}')
            raise
        elapsed = time.perf_counter() - started
        allocated = torch.cuda.max_memory_allocated() / 2 ** 30
        reserved = torch.cuda.max_memory_reserved() / 2 ** 30
        finite = bool(torch.isfinite(prediction).all()) and math.isfinite(loss)
        print(
            f'[single-match preflight] {height}x{width} batch={batch} '
            f'time={elapsed:.2f}s loss={loss:.6f} finite={finite} '
            f'peak_allocated={allocated:.2f}GiB '
            f'peak_reserved={reserved:.2f}GiB')
        if not finite:
            raise RuntimeError(
                'single-match preflight produced non-finite output/loss')
        if reserved > args.max_reserved_gib:
            raise RuntimeError(
                f'peak reserved {reserved:.2f}GiB exceeds safety limit '
                f'{args.max_reserved_gib:.2f}GiB')
    print('[single-match preflight] PASS')


if __name__ == '__main__':
    main()
