#!/usr/bin/env python3
"""Preflight identity, compatibility and memory checks for explicit matching."""

import argparse
import gc
import inspect
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

import config as cfg
from Trainer import Model
from start import arch_from_model_section


def load_yaml(path):
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def build_model(config, checkpoint, training=False):
    model_config = config['model']
    cfg.MODEL_CONFIG['LOGNAME'] = config['exp_name']
    cfg.MODEL_CONFIG['MODEL_ARCH'] = arch_from_model_section(model_config)
    allowed = set(inspect.signature(Model).parameters)
    kwargs = {
        name: value for name, value in model_config.items()
        if name in allowed
    }
    model = Model(**kwargs)
    model.load_model(str(checkpoint), resume=False)
    model.loaded_checkpoint = None
    phase = config['phases'][0]
    model.set_local_enabled(phase.get('local', True))
    if training:
        model.set_trainable_scope(phase.get('trainable', 'all'))
        model.configure_optimizer(config['optim'])
        model.train()
    else:
        model.eval()
    return model


def random_inputs(batch, height, width, device):
    generator = torch.Generator(device='cpu').manual_seed(1234)
    frames = torch.rand(
        batch, 9, height, width, generator=generator).to(device)
    return frames[:, :6], frames[:, 6:9]


def identity_check(match_config, checkpoint, height, width):
    matching = build_model(match_config, checkpoint, training=False)
    images, _ = random_inputs(1, height, width, matching._dev)
    matcher = matching.net.sparse_matcher
    matching.net.sparse_matcher = None
    with torch.inference_mode():
        baseline = matching.inference(
            images[:, :3], images[:, 3:6], local=matching.local,
            TTA=False, return_debug=False).cpu()
        repeat = matching.inference(
            images[:, :3], images[:, 3:6], local=matching.local,
            TTA=False, return_debug=False).cpu()
    matching.net.sparse_matcher = matcher
    with torch.inference_mode():
        candidate = matching.inference(
            images[:, :3], images[:, 3:6], local=matching.local,
            TTA=False, return_debug=False).cpu()
    repeat_difference = (baseline - repeat).abs()
    branch_difference = (baseline - candidate).abs()
    result = {
        'matcher_residual_pixels': float(matcher.last_residual_abs),
        'matcher_exact_zero': float(matcher.last_residual_abs) == 0.0,
        'repeat_max_abs_difference': float(repeat_difference.max()),
        'repeat_mean_abs_difference': float(repeat_difference.mean()),
        'branch_max_abs_difference': float(branch_difference.max()),
        'branch_mean_abs_difference': float(branch_difference.mean()),
    }
    del matching, images, baseline, repeat, candidate
    gc.collect()
    torch.cuda.empty_cache()
    return result


def training_memory_check(config, checkpoint, batch, height, width):
    model = build_model(config, checkpoint, training=True)
    images, target = random_inputs(batch, height, width, model._dev)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
        _, loss, _ = model.update(
            images, target, timestep=0.5, learning_rate=1.0e-5,
            training=True, scaler=None, accumulation_steps=1,
            accumulation_index=0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    # The merger is zero at the start of the optimization step. Re-evaluate
    # after that step to prove it can become active.
    model.eval()
    with torch.inference_mode(), torch.autocast(
            'cuda', dtype=torch.bfloat16, enabled=True):
        model.inference(
            images[:1, :3], images[:1, 3:6], local=model.local,
            TTA=False, return_debug=False)
    matcher = model.net.sparse_matcher
    result = {
        'loss': float(loss),
        'elapsed_seconds': elapsed,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2 ** 30,
        'selected_ratio': float(matcher.last_selected_ratio),
        'confidence': float(matcher.last_confidence),
        'top1_margin': float(matcher.last_margin),
        'mutual_error': float(matcher.last_mutual_error),
        'mutual_ratio': float(matcher.last_mutual_ratio),
        'valid_ratio': float(matcher.last_valid_ratio),
        'proposal_pixels': float(matcher.last_proposal_abs),
        'applied_pixels_global': float(matcher.last_residual_abs),
        'applied_pixels_selected': float(
            matcher.last_residual_selected_abs),
    }
    del model, images, target
    gc.collect()
    torch.cuda.empty_cache()
    return result


def real_matching_check(config, checkpoint, data_root, list_path,
                        max_scenes, height, width):
    model = build_model(config, checkpoint, training=False)
    matcher = model.net.sparse_matcher
    rows = [
        line.split('\t')[0] for line in Path(list_path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    ][:max_scenes]
    results = []
    for relative in rows:
        frames = sorted(
            path for path in (Path(data_root) / relative).iterdir()
            if path.suffix.lower() in ('.png', '.jpg', '.jpeg'))
        if len(frames) < 3:
            continue
        endpoints = []
        for path in (frames[0], frames[-1]):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f'OpenCV cannot read {path}')
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            top = max((image.shape[0] - height) // 2, 0)
            left = max((image.shape[1] - width) // 2, 0)
            image = image[top:top + height, left:left + width]
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            tensor = torch.from_numpy(
                np.ascontiguousarray(image.transpose(2, 0, 1)))
            endpoints.append(
                tensor.float().div_(255.0).unsqueeze(0).to(model._dev))
        with torch.inference_mode(), torch.autocast(
                'cuda', dtype=torch.bfloat16, enabled=True):
            model.inference(
                endpoints[0], endpoints[1], local=model.local,
                TTA=False, return_debug=False)
        results.append({
            'scene': relative,
            'confidence': float(matcher.last_confidence),
            'top1_margin': float(matcher.last_margin),
            'mutual_error': float(matcher.last_mutual_error),
            'mutual_ratio': float(matcher.last_mutual_ratio),
            'valid_ratio': float(matcher.last_valid_ratio),
            'similarity_gain': float(matcher.last_similarity_gain),
            'similarity_improved_ratio': float(
                matcher.last_similarity_improved_ratio),
            'proposal_pixels': float(matcher.last_proposal_abs),
        })
    summary = {'scenes': results}
    if results:
        for key in (
                'confidence', 'top1_margin', 'mutual_error', 'mutual_ratio',
                'valid_ratio', 'similarity_gain',
                'similarity_improved_ratio', 'proposal_pixels'):
            summary[f'mean_{key}'] = sum(
                row[key] for row in results) / len(results)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control-config', required=True)
    parser.add_argument('--match-config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--width', type=int, default=704)
    parser.add_argument(
        '--data-root',
        default='/home/zhenying/qhong/data/ssd/vfi_database')
    parser.add_argument(
        '--xtrain-list',
        default='/home/zhenying/qhong/data/ssd/vfi_database/lists/xtrain_val.txt')
    parser.add_argument('--real-scenes', type=int, default=5)
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the explicit-matching smoke')
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    control_config = load_yaml(args.control_config)
    match_config = load_yaml(args.match_config)
    report = {
        'device': torch.cuda.get_device_name(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'shape': [args.batch, 6, args.height, args.width],
        'checkpoint': str(checkpoint),
        'identity': identity_check(
            match_config, checkpoint,
            min(args.height, 192), min(args.width, 320)),
        'training': training_memory_check(
            match_config, checkpoint, args.batch, args.height, args.width),
        'real_xtrain': real_matching_check(
            match_config, checkpoint, args.data_root, args.xtrain_list,
            args.real_scenes, args.height, args.width),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
