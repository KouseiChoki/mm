#!/usr/bin/env python3
"""CUDA gate for the frozen correspondence-pyramid VFI integration."""

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
from Trainer import Model, load_checkpoint_file
from kousei_dataset import MixedTierDataset, resolve_train_lists


NEW_PREFIXES = ('correspondence_pyramid.',)


def is_new_parameter(name):
    return (name.startswith(NEW_PREFIXES)
            or '.external_correlation.' in name
            or name.endswith('.external_correlation_scale'))


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
        version=model_config.get('version', 3), **extra)
    kwargs = {
        name: value for name, value in model_config.items()
        if name in train_only
    }
    model = Model(**kwargs)
    model.configure_optimizer(config['optim'])
    model.load_model(str(checkpoint), resume=False)
    phase = config['phases'][0]
    model.set_local_enabled(phase.get('local', True))
    model.set_trainable_scope(phase.get('trainable', 'all'))
    model.configure_optimizer(config['optim'])
    return model


def base_checkpoint_mismatches(model, checkpoint):
    payload = load_checkpoint_file(str(checkpoint))
    checkpoint_state = payload.get('net', payload)
    model_state = model.net.state_dict()
    mismatches = []
    for name, value in model_state.items():
        if is_new_parameter(name):
            continue
        old = checkpoint_state.get(name)
        if old is None:
            mismatches.append(f'{name}: missing')
        elif old.shape != value.shape:
            mismatches.append(
                f'{name}: {tuple(old.shape)} != {tuple(value.shape)}')
        elif not torch.equal(old.cpu(), value.detach().cpu()):
            mismatches.append(f'{name}: value differs after load')
    return mismatches


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
        mv_cache_dirname=data.get('mv_cache_dirname'),
        mv_cache_required=data.get('mv_cache_required', False),
        mv_cache_preview_stride=data.get('mv_cache_preview_stride', 4),
        mv_cycle_confidence=data.get('mv_cycle_confidence', 'none'),
        mv_cycle_cache_required=data.get(
            'mv_cycle_cache_required', False),
        mv_cycle_cache_root=data.get('mv_cycle_cache_root'),
        mv_cycle_on_the_fly=data.get('mv_cycle_on_the_fly', False),
        augment_profile=data.get('augment_profile', 'legacy'))
    dataset.set_ratios(phase['ratios'], phase.get('batch_counts'))
    return dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='train_config_0821_lc_pretrained_corr_weekend.yaml')
    parser.add_argument(
        '--checkpoint',
        default='ckpt/0729_lc_v3s2/0729_lc_v3s2_800.pkl')
    parser.add_argument('--output', default=None)
    parser.add_argument('--max-peak-gib', type=float, default=29.0)
    parser.add_argument('--steps', type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this preflight')
    if args.steps < 1:
        raise ValueError('--steps must be >= 1')
    config = yaml.safe_load(Path(args.config).read_text())
    checkpoint = Path(args.checkpoint)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    dataset = build_dataset(config)
    batch_size = int(config['data']['batch_size'])
    samples = [dataset[index] for index in range(batch_size)]
    frames = torch.stack([sample[0] for sample in samples]).float().div_(255.0)
    timesteps = torch.stack([sample[1] for sample in samples])
    flow_gt = torch.stack([sample[2] for sample in samples])
    has_mv = torch.stack([sample[3] for sample in samples])
    images, targets = frames[:, :6], frames[:, 6:9]

    model = build_model(config, checkpoint)
    mismatches = base_checkpoint_mismatches(model, checkpoint)
    frozen_trainable = [
        name for name, parameter in model.net.named_parameters()
        if name.startswith('correspondence_pyramid.')
        and parameter.requires_grad]
    images = images.to(model._dev)
    targets = targets.to(model._dev)
    timesteps = timesteps.to(model._dev)

    # Scale=0 is exactly the historical flow path. Measure how far the safe
    # 0.01 initialization perturbs the inherited 0729 output.
    scales = [
        head.external_correlation_scale
        for head in model.net.block
        if head.external_correlation_scale is not None]
    initial_scales = [float(scale.detach()) for scale in scales]
    model.eval()
    with torch.no_grad(), torch.autocast(
            'cuda', dtype=torch.bfloat16, enabled=True):
        for scale in scales:
            scale.zero_()
        reference = model.net(
            images, timestep=timesteps, local=model.local)[-1].float()
        for scale, value in zip(scales, initial_scales):
            scale.fill_(value)
        injected = model.net(
            images, timestep=timesteps, local=model.local)[-1].float()
    identity_mean_abs = float((injected - reference).abs().mean())
    identity_max_abs = float((injected - reference).abs().max())
    del reference, injected

    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    step_seconds = []
    losses = []
    for step in range(args.steps):
        step_started = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            prediction, loss, flow_loss = model.update(
                images, targets, timestep=timesteps,
                learning_rate=float(config['optim']['lr_max']),
                training=True, scaler=None, flow_gt=flow_gt, has_mv=has_mv,
                loss_step=step, accumulation_steps=1,
                accumulation_index=0)
        torch.cuda.synchronize()
        losses.append(float(loss))
        step_seconds.append(time.perf_counter() - step_started)

    stages = []
    for index, head in enumerate(model.net.block):
        correlation = head.external_correlation
        if correlation is None:
            continue
        gradients = [
            parameter.grad.detach().abs().mean()
            for parameter in correlation.parameters()
            if parameter.grad is not None]
        scale_gradient = head.external_correlation_scale.grad
        stages.append({
            'stage': index,
            'radius': correlation.radius,
            'peak_probability': float(correlation.last_peak_probability),
            'normalized_entropy': float(
                correlation.last_normalized_entropy),
            'encoded_feature_abs': float(correlation.last_feature_abs),
            'adapter_mean_gradient_abs': (
                float(torch.stack(gradients).mean()) if gradients else 0.0),
            'scale': float(head.external_correlation_scale.detach()),
            'scale_gradient_abs': (
                float(scale_gradient.detach().abs())
                if scale_gradient is not None else 0.0),
        })

    peak_allocated = torch.cuda.max_memory_allocated() / 2 ** 30
    peak_reserved = torch.cuda.max_memory_reserved() / 2 ** 30
    failures = []
    if mismatches:
        failures.append(f'{len(mismatches)} inherited checkpoint mismatches')
    if frozen_trainable:
        failures.append('frozen correspondence parameters are trainable')
    if not torch.isfinite(prediction).all():
        failures.append('prediction contains non-finite values')
    if identity_mean_abs > 1e-3:
        failures.append(
            f'initial output delta {identity_mean_abs:.6g} exceeds 1e-3')
    if peak_allocated >= args.max_peak_gib:
        failures.append(
            f'peak allocated {peak_allocated:.2f} GiB exceeds gate')
    if len(stages) != 2:
        failures.append(f'expected 2 correspondence stages, got {len(stages)}')
    if any(stage['adapter_mean_gradient_abs'] <= 0.0
           or stage['scale_gradient_abs'] <= 0.0 for stage in stages):
        failures.append('correspondence adapter has zero/missing gradient')

    report = {
        'status': 'ok' if not failures else 'failed',
        'failures': failures,
        'device': torch.cuda.get_device_name(),
        'shape': list(images.shape),
        'checkpoint_base_mismatches': mismatches[:20],
        'frozen_trainable_parameters': frozen_trainable[:20],
        'identity_mean_abs': identity_mean_abs,
        'identity_max_abs': identity_max_abs,
        'losses': losses,
        'flow_loss': float(flow_loss),
        'prediction_finite': bool(torch.isfinite(prediction).all()),
        'step_seconds': step_seconds,
        'steady_step_seconds': (
            sum(step_seconds[1:]) / len(step_seconds[1:])
            if len(step_seconds) > 1 else step_seconds[0]),
        'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_gib': peak_allocated,
        'peak_reserved_gib': peak_reserved,
        'correspondence_stages': stages,
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
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
