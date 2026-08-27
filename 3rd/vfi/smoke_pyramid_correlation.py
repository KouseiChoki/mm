#!/usr/bin/env python3
"""One real-batch CUDA preflight for the integrated correlation motion core."""

import argparse
import gc
import inspect
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import config as cfg
from Trainer import Model
from kousei_dataset import MixedTierDataset, resolve_train_lists


def build_model(config, checkpoint):
    model_config = config['model']
    named = ('F', 'depth', 'M', 'version')
    train_only = set(inspect.signature(Model).parameters)
    extra = {
        name: value for name, value in model_config.items()
        if name not in named and name not in train_only
    }
    cfg.MODEL_CONFIG['LOGNAME'] = config['exp_name']
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=model_config.get('F', 32),
        depth=model_config.get('depth', [2, 2, 2, 3, 3]),
        M=model_config.get('M', False),
        version=model_config.get('version', 2), **extra)
    kwargs = {
        name: value for name, value in model_config.items()
        if name in train_only
    }
    model = Model(**kwargs)
    model.load_model(str(checkpoint), resume=False)
    model.loaded_checkpoint = None
    phase = config['phases'][0]
    model.set_local_enabled(phase.get('local', True))
    model.set_trainable_scope(phase.get('trainable', 'all'))
    model.configure_optimizer(config['optim'])
    model.train()
    return model


def build_dataset(config):
    data = config['data']
    phase = config['phases'][0]
    lists = resolve_train_lists(
        data['lists_dir'], config['phases'], tiers=data.get('tiers'))
    dataset = MixedTierDataset(
        data['root'], lists, ratios=phase['ratios'],
        source_options=data.get('source_options'),
        crop_hw=tuple(phase['crop_sizes'][0][:2]),
        framesteps=tuple(data['framesteps']),
        t_half_prob=data['t_half_prob'], mv_prob=data['mv_prob'],
        mv_sign=tuple(data['mv_sign']),
        mv_symmetry_confidence=data.get('mv_symmetry_confidence', False),
        occ_alpha=data.get('occ_alpha', 0.05),
        occ_beta=data.get('occ_beta', 1.0),
        motion_aware_crop_prob=data.get('motion_aware_crop_prob', 0.0),
        motion_crop_threshold=data.get('motion_crop_threshold', 1.0),
        small_motion_min_pixels=data.get('small_motion_min_pixels', 8),
        small_motion_max_ratio=data.get('small_motion_max_ratio', 0.05),
        motion_crop_jitter=data.get('motion_crop_jitter', 0.2),
        interpolation_aware_crop_prob=data.get(
            'interpolation_aware_crop_prob', 0.0),
        interpolation_residual_threshold=data.get(
            'interpolation_residual_threshold', 0.04),
        augment_profile=data.get('augment_profile', 'legacy'))
    dataset.set_ratios(phase['ratios'], phase.get('batch_counts'))
    return dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--steps', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for correlation preflight')
    if args.steps < 1:
        raise ValueError('--steps must be >= 1')
    config = yaml.safe_load(Path(args.config).read_text())
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    dataset = build_dataset(config)
    batch_size = int(config['data']['batch_size'])
    samples = [dataset[index] for index in range(batch_size)]
    frames = torch.stack([sample[0] for sample in samples]).float().div_(255.0)
    timesteps = torch.stack([sample[1] for sample in samples])
    images = frames[:, :6]
    targets = frames[:, 6:9]

    model = build_model(config, Path(args.checkpoint))
    images = images.to(model._dev)
    targets = targets.to(model._dev)
    timesteps = timesteps.to(model._dev)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    step_seconds = []
    for _ in range(args.steps):
        step_started = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            prediction, loss, _ = model.update(
                images, targets, timestep=timesteps,
                learning_rate=float(config['optim']['lr_max']), training=True,
                scaler=None, accumulation_steps=1, accumulation_index=0)
        torch.cuda.synchronize()
        losses.append(float(loss))
        step_seconds.append(time.perf_counter() - step_started)

    stages = []
    for index, head in enumerate(model.net.block):
        correlation = getattr(head, 'correlation', None)
        if correlation is None:
            continue
        gradients = [
            parameter.grad.detach().abs().mean()
            for parameter in correlation.parameters()
            if parameter.grad is not None]
        stages.append({
            'stage': index,
            'radius': correlation.radius,
            'peak_probability': float(correlation.last_peak_probability),
            'normalized_entropy': float(
                correlation.last_normalized_entropy),
            'encoded_feature_abs': float(correlation.last_feature_abs),
            'mean_gradient_abs': (
                float(torch.stack(gradients).mean()) if gradients else 0.0),
        })

    report = {
        'status': 'ok',
        'device': torch.cuda.get_device_name(),
        'shape': list(images.shape),
        'steps': args.steps,
        'losses': losses,
        'step_seconds': step_seconds,
        'steady_step_seconds': (
            sum(step_seconds[1:]) / len(step_seconds[1:])
            if len(step_seconds) > 1 else step_seconds[0]),
        'prediction_finite': bool(torch.isfinite(prediction).all()),
        'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2 ** 30,
        'trainable_parameters': sum(
            parameter.numel() for parameter in model.net.parameters()
            if parameter.requires_grad),
        'correlation_stages': stages,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    del model, images, targets, timesteps, prediction
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
