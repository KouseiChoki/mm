#!/usr/bin/env python3
"""CUDA memory/gradient preflight for the exact PQMax training graph."""

import argparse
import inspect
import math
import time

import torch

import config as cfg
from Trainer import Model
from train import MODEL_TRAIN_KEYS, load_config


def parse_shape(value):
    try:
        height, width, batch = (int(part) for part in value.lower().split('x'))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            'shape must be HEIGHTxWIDTHxBATCH') from exc
    if min(height, width, batch) <= 0 or height % 16 or width % 16:
        raise argparse.ArgumentTypeError(
            'height/width must be positive multiples of 16; batch must be >0')
    return height, width, batch


def build_model(config):
    model_config = config['model']
    structural = {
        key: value for key, value in model_config.items()
        if key not in ('F', 'depth', 'M', 'version') + MODEL_TRAIN_KEYS
    }
    cfg.MODEL_CONFIG['LOGNAME'] = config['exp_name'] + '_preflight'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=model_config['F'], depth=model_config['depth'],
        M=model_config.get('M', False), version=model_config['version'],
        **structural)
    accepted = inspect.signature(Model.__init__).parameters
    loss_options = {
        key: value for key, value in model_config.items()
        if key in accepted and key != 'self'
    }
    model = Model(**loss_options)
    model.configure_optimizer(config['optim'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='train_config_0825_pqmax_amt_mamba_max.yaml')
    parser.add_argument(
        '--restore_ckpt',
        default='ckpt/0729_lc_v3s2/0729_lc_v3s2_800.pkl')
    parser.add_argument(
        '--shape', action='append', type=parse_shape,
        help='HEIGHTxWIDTHxBATCH; repeat for multiple phase shapes')
    parser.add_argument('--max_reserved_gib', type=float, default=30.5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA不可用，无法执行PQMax显存preflight；请先恢复NVIDIA驱动')

    shapes = args.shape or [(384, 704, 1), (512, 896, 1)]
    config = load_config(args.config)
    model = build_model(config)
    model.load_model(args.restore_ckpt, resume=False)
    model.set_local_enabled(True)
    amp_name = str(config['optim'].get('amp_dtype', 'bf16')).lower()
    amp_dtype = torch.bfloat16 if amp_name == 'bf16' else torch.float16
    use_amp = bool(config['optim'].get('amp', True))
    lr = float(config['optim']['lr_max'])

    total_parameters = sum(parameter.numel() for parameter in model.net.parameters())
    print(f'[PQMax preflight] parameters={total_parameters:,} amp={amp_name}')
    for height, width, batch in shapes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        images = torch.rand(batch, 6, height, width, device='cuda')
        target = torch.rand(batch, 3, height, width, device='cuda')
        timestep = torch.full(
            (batch, 1, 1, 1), 0.5, device='cuda')
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with torch.autocast(
                    'cuda', dtype=amp_dtype, enabled=use_amp):
                prediction, loss, _ = model.update(
                    images, target, timestep=timestep,
                    learning_rate=lr, training=True, loss_step=5000)
            torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'[PQMax preflight] OOM shape={height}x{width} batch={batch}')
            raise
        elapsed = time.perf_counter() - started
        allocated = torch.cuda.max_memory_allocated() / 2 ** 30
        reserved = torch.cuda.max_memory_reserved() / 2 ** 30
        finite = bool(torch.isfinite(prediction).all()) and math.isfinite(loss)
        print(
            f'[PQMax preflight] {height}x{width} batch={batch} '
            f'time={elapsed:.2f}s loss={loss:.6f} finite={finite} '
            f'peak_allocated={allocated:.2f}GiB '
            f'peak_reserved={reserved:.2f}GiB')
        if not finite:
            raise RuntimeError('PQMax preflight produced non-finite output/loss')
        if reserved > args.max_reserved_gib:
            raise RuntimeError(
                f'peak reserved {reserved:.2f}GiB exceeds safety limit '
                f'{args.max_reserved_gib:.2f}GiB')

    print('[PQMax preflight] PASS')


if __name__ == '__main__':
    main()
