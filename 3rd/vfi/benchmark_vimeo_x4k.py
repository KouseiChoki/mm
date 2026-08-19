#!/usr/bin/env python3
"""Compare custom checkpoints and official VFIMamba on common data.

Protocols:
  * Vimeo90K: official tri_testlist, im1/im3 -> im2.
  * X4K holdout: xtrain_val scene list, two 8x recursive intervals per
    65-frame 768x768 clip (0->32 and 32->64), 14 predictions per scene.

PSNR is averaged per predicted frame. SSIM uses the exact ``ssim_matlab``
implementation bundled with the official VFIMamba benchmark. X4K predictions
are rounded to uint8 before scoring, matching the official XTEST script.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path('/home/zhenying/qhong/data/ssd/vfi_database')
DEFAULT_OFFICIAL_ROOT = Path('/home/zhenying/qhong/repo/VFIMamba_official_eval')
MODEL_SPECS = {
    '0729_lc_800': {
        'kind': 'custom',
        'ckpt': HERE / 'ckpt/0729_lc_v3s2/0729_lc_v3s2_800.pkl',
    },
    '0807_official_recipe_320': {
        'kind': 'custom',
        'ckpt': HERE / (
            'ckpt/0807_s2v3_official_tuesday/'
            '0807_s2v3_official_tuesday_320.pkl'),
    },
    'official_vfimamba': {
        'kind': 'official',
        'ckpt': DEFAULT_OFFICIAL_ROOT / 'ckpt/VFIMamba.pkl',
    },
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with open(temporary, 'w') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    os.replace(temporary, path)


def load_ssim(official_root: Path):
    source = official_root / 'benchmark/utils/pytorch_msssim.py'
    spec = importlib.util.spec_from_file_location('_vfimamba_ssim', source)
    if spec is None or spec.loader is None:
        raise ImportError(f'无法载入官方SSIM实现: {source}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ssim_matlab


def load_custom(ckpt_path: Path):
    sys.path.insert(0, str(HERE))
    import config as cfg
    from Trainer import Model

    yaml_path = ckpt_path.parent / 'model.yaml'
    if not yaml_path.is_file():
        raise FileNotFoundError(f'checkpoint缺少配套model.yaml: {yaml_path}')
    with open(yaml_path) as handle:
        model_cfg = yaml.safe_load(handle)['model']
    train_only = (
        'loss_type', 'flow_loss_weight', 'flow_stage_gamma',
        'flow_motion_threshold', 'flow_motion_balance', 'flow_motion_gain',
        'flow_motion_scale', 'flow_motion_weight_cap', 'flow_charbonnier_eps',
        'flow_loss_warmup_steps', 'merge_loss_gamma', 'merge_loss_weights',
        'normalize_pixel_loss', 'residual_loss_weight',
        'lc_charbonnier_eps', 'lc_census_weight', 'lc_lap_weight',
        'lc_warp_weight', 'pervfi_mask_loss_weight',
        'multi_hypothesis_oracle_weight',
        'multi_hypothesis_oracle_eps',
        'edge_loss_weight', 'edge_warp_loss_weight',
        'edge_motion_gain', 'edge_motion_scale', 'edge_weight_cap',
        'edge_charbonnier_eps')
    named = ('F', 'depth', 'M', 'version')
    extra = {key: value for key, value in model_cfg.items()
             if key not in named + train_only}
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=model_cfg.get('F', 32),
        depth=model_cfg.get('depth', [2, 2, 2, 3, 3]),
        M=model_cfg.get('M', False),
        version=model_cfg.get('version', 1),
        **extra)
    model = Model(loss_type=model_cfg.get('loss_type', 'lap'))
    model.load_model(str(ckpt_path))
    model.loaded_checkpoint = None
    model.eval()
    model.device()

    def infer(img0: torch.Tensor, img1: torch.Tensor, tta: bool) -> torch.Tensor:
        return model.inference(
            img0, img1, local=True, TTA=tta, timestep=0.5,
            return_debug=False)

    return infer, 'RGB'


def load_official(ckpt_path: Path, official_root: Path):
    # This runs in a dedicated child process, so official modules cannot
    # collide with the custom repository's modules of the same names.
    sys.path.insert(0, str(official_root))
    import config as cfg
    from Trainer_finetune import Model, convert

    cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=32, depth=[2, 2, 2, 3, 3])
    model = Model(-1)
    payload = torch.load(str(ckpt_path), map_location='cpu')
    model.net.load_state_dict(convert(payload), strict=True)
    del payload
    model.eval()
    model.device()

    def infer(img0: torch.Tensor, img1: torch.Tensor, tta: bool) -> torch.Tensor:
        return model.inference(
            img0, img1, local=True, TTA=tta, timestep=0.5)

    # Official benchmark scripts feed OpenCV BGR tensors directly.
    return infer, 'BGR'


def read_tensor(path: Path, color_order: str) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'OpenCV无法读取: {path}')
    if color_order == 'RGB':
        image = image[..., ::-1]
    array = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(array).float().div_(255.0).unsqueeze(0).cuda()


def score(pred: torch.Tensor, target: torch.Tensor,
          ssim_fn, quantize: bool) -> Tuple[float, float]:
    pred = pred.clamp(0, 1)
    if quantize:
        pred = torch.round(pred * 255.0) / 255.0
    mse = (pred - target).square().mean().clamp(min=1e-12)
    psnr = float((-10.0 * torch.log10(mse)).item())
    ssim = float(ssim_fn(target, pred).item())
    return psnr, ssim


def recursive_frames(infer: Callable, first: torch.Tensor,
                     last: torch.Tensor, levels: int,
                     tta: bool) -> List[torch.Tensor]:
    if levels <= 0:
        return []
    middle = infer(first, last, tta)
    return (recursive_frames(infer, first, middle, levels - 1, tta)
            + [middle]
            + recursive_frames(infer, middle, last, levels - 1, tta))


def load_progress(path: Path, model_name: str, resume: bool) -> dict:
    if resume and path.is_file():
        with open(path) as handle:
            state = json.load(handle)
        if state.get('model') != model_name:
            raise ValueError(f'结果文件模型不一致: {path}')
        return state
    return {
        'model': model_name,
        'datasets': {},
        'complete': False,
    }


def evaluate_vimeo(args, infer, color_order, ssim_fn,
                    state: dict, result_path: Path) -> None:
    list_path = Path(args.vimeo_list)
    sequence_root = Path(args.vimeo_root)
    names = [line.strip() for line in list_path.read_text().splitlines()
             if line.strip()]
    if args.max_vimeo is not None:
        names = names[:args.max_vimeo]
    saved = state['datasets'].get('vimeo90k', {}) if args.resume else {}
    done = min(int(saved.get('cases_done', 0)), len(names))
    psnr_sum = float(saved.get('psnr_sum', 0.0)) if done else 0.0
    ssim_sum = float(saved.get('ssim_sum', 0.0)) if done else 0.0
    started = time.time()
    for index in range(done, len(names)):
        folder = sequence_root / names[index]
        first = read_tensor(folder / 'im1.png', color_order)
        target = read_tensor(folder / 'im2.png', color_order)
        last = read_tensor(folder / 'im3.png', color_order)
        pred = infer(first, last, args.tta)
        psnr, ssim = score(pred, target, ssim_fn, quantize=False)
        psnr_sum += psnr
        ssim_sum += ssim
        done = index + 1
        if done % args.save_every == 0 or done == len(names):
            state['datasets']['vimeo90k'] = {
                'protocol': 'official tri_testlist; im1,im3 -> im2',
                'cases_done': done,
                'predicted_frames': done,
                'psnr_sum': psnr_sum,
                'ssim_sum': ssim_sum,
                'psnr': psnr_sum / done,
                'ssim': ssim_sum / done,
                'complete': done == len(names),
            }
            atomic_json(result_path, state)
            elapsed = time.time() - started
            print(f'[Vimeo90K] {done}/{len(names)} '
                  f'PSNR={psnr_sum/done:.4f} SSIM={ssim_sum/done:.6f} '
                  f'run_elapsed={elapsed/60:.1f}min', flush=True)


def x4k_cases(root: Path, list_path: Path,
              max_scenes: int | None) -> List[Tuple[str, Path, int]]:
    scenes = []
    for line in list_path.read_text().splitlines():
        if not line.strip():
            continue
        rel = line.split('\t', 1)[0]
        scenes.append((rel, root / rel))
    if max_scenes is not None:
        scenes = scenes[:max_scenes]
    cases = []
    for rel, folder in scenes:
        frames = sorted(folder.glob('*.png'))
        if len(frames) < 65:
            raise ValueError(f'X4K holdout场景不足65帧: {folder} ({len(frames)})')
        cases.extend((rel, folder, start) for start in (0, 32))
    return cases


def evaluate_x4k(args, infer, color_order, ssim_fn,
                  state: dict, result_path: Path) -> None:
    cases = x4k_cases(
        Path(args.data_root), Path(args.x4k_list), args.max_x4k_scenes)
    saved = state['datasets'].get('x4k1000fps_holdout', {}) if args.resume else {}
    done = min(int(saved.get('cases_done', 0)), len(cases))
    frames_done = int(saved.get('predicted_frames', 0)) if done else 0
    psnr_sum = float(saved.get('psnr_sum', 0.0)) if done else 0.0
    ssim_sum = float(saved.get('ssim_sum', 0.0)) if done else 0.0
    started = time.time()
    for index in range(done, len(cases)):
        _, folder, start = cases[index]
        paths = sorted(folder.glob('*.png'))
        first = read_tensor(paths[start], color_order)
        last = read_tensor(paths[start + 32], color_order)
        predictions = recursive_frames(infer, first, last, 3, args.tta)
        target_indices = range(start + 4, start + 32, 4)
        for pred, target_index in zip(predictions, target_indices):
            target = read_tensor(paths[target_index], color_order)
            psnr, ssim = score(pred, target, ssim_fn, quantize=True)
            psnr_sum += psnr
            ssim_sum += ssim
            frames_done += 1
        done = index + 1
        if done % max(args.save_every // 7, 1) == 0 or done == len(cases):
            state['datasets']['x4k1000fps_holdout'] = {
                'protocol': (
                    'xtrain_val; 65-frame 768x768 clips; recursive 8x on '
                    '0->32 and 32->64; uint8 metric'),
                'cases_done': done,
                'scene_count': len(cases) // 2,
                'predicted_frames': frames_done,
                'psnr_sum': psnr_sum,
                'ssim_sum': ssim_sum,
                'psnr': psnr_sum / frames_done,
                'ssim': ssim_sum / frames_done,
                'complete': done == len(cases),
            }
            atomic_json(result_path, state)
            elapsed = time.time() - started
            print(f'[X4K holdout] {done}/{len(cases)} cases, '
                  f'{frames_done} frames PSNR={psnr_sum/frames_done:.4f} '
                  f'SSIM={ssim_sum/frames_done:.6f} '
                  f'run_elapsed={elapsed/60:.1f}min', flush=True)


def run_one(args) -> None:
    if args.ckpt is not None:
        spec = {'kind': 'custom', 'ckpt': Path(args.ckpt).expanduser()}
        result_name = args.model_name or Path(args.ckpt).stem
    else:
        spec = MODEL_SPECS[args.model]
        result_name = args.model
    ckpt = Path(spec['ckpt'])
    official_root = Path(args.official_root)
    if not ckpt.is_file():
        raise FileNotFoundError(f'checkpoint不存在: {ckpt}')
    if spec['kind'] == 'custom':
        infer, color_order = load_custom(ckpt)
    else:
        infer, color_order = load_official(ckpt, official_root)
    ssim_fn = load_ssim(official_root)
    result_path = Path(args.output_dir) / f'{result_name}.json'
    state = load_progress(result_path, result_name, args.resume)
    state.update({
        'checkpoint': str(ckpt),
        'tta': bool(args.tta),
        'color_order': color_order,
        'metric': 'per-frame mean PSNR + official VFIMamba ssim_matlab',
    })
    if not state['datasets'].get('vimeo90k', {}).get('complete'):
        evaluate_vimeo(args, infer, color_order, ssim_fn, state, result_path)
    if not state['datasets'].get('x4k1000fps_holdout', {}).get('complete'):
        evaluate_x4k(args, infer, color_order, ssim_fn, state, result_path)
    state['complete'] = all(
        item.get('complete', False) for item in state['datasets'].values())
    atomic_json(result_path, state)
    print(f'[done] {result_name}: {result_path}', flush=True)


def print_table(output_dir: Path, names: Sequence[str]) -> None:
    print('\nmodel\tVimeo PSNR\tVimeo SSIM\tX4K PSNR\tX4K SSIM')
    for name in names:
        path = output_dir / f'{name}.json'
        if not path.is_file():
            continue
        with open(path) as handle:
            data = json.load(handle)['datasets']
        vim = data.get('vimeo90k', {})
        x4k = data.get('x4k1000fps_holdout', {})
        print(f'{name}\t{vim.get("psnr", float("nan")):.4f}\t'
              f'{vim.get("ssim", float("nan")):.6f}\t'
              f'{x4k.get("psnr", float("nan")):.4f}\t'
              f'{x4k.get("ssim", float("nan")):.6f}')


def run_all(args) -> None:
    names = list(MODEL_SPECS)
    for name in names:
        command = [
            sys.executable, str(Path(__file__).resolve()),
            '--model', name,
            '--data-root', args.data_root,
            '--vimeo-root', args.vimeo_root,
            '--vimeo-list', args.vimeo_list,
            '--x4k-list', args.x4k_list,
            '--official-root', args.official_root,
            '--output-dir', args.output_dir,
            '--save-every', str(args.save_every),
        ]
        command.append('--tta' if args.tta else '--no-tta')
        if args.resume:
            command.append('--resume')
        if args.max_vimeo is not None:
            command += ['--max-vimeo', str(args.max_vimeo)]
        if args.max_x4k_scenes is not None:
            command += ['--max-x4k-scenes', str(args.max_x4k_scenes)]
        print(f'\n[run] {name}', flush=True)
        subprocess.run(command, check=True, cwd=str(HERE))
    print_table(Path(args.output_dir), names)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['all', *MODEL_SPECS], default='all')
    parser.add_argument(
        '--ckpt', default=None,
        help='评测任意自定义checkpoint；设置后忽略--model')
    parser.add_argument(
        '--model-name', default=None,
        help='--ckpt结果名；默认使用checkpoint文件名')
    parser.add_argument('--data-root', default=str(DEFAULT_ROOT))
    parser.add_argument(
        '--vimeo-root',
        default=str(DEFAULT_ROOT / 'opensource/vimeo90k'))
    parser.add_argument(
        '--vimeo-list',
        default=str(DEFAULT_ROOT / 'vimeo_triplet/tri_testlist.txt'))
    parser.add_argument(
        '--x4k-list',
        default=str(DEFAULT_ROOT / 'lists/xtrain_val.txt'))
    parser.add_argument('--official-root', default=str(DEFAULT_OFFICIAL_ROOT))
    parser.add_argument(
        '--output-dir', default=str(HERE / 'benchmark_results/vimeo_x4k'))
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-vimeo', type=int, default=None)
    parser.add_argument('--max-x4k-scenes', type=int, default=None)
    tta = parser.add_mutually_exclusive_group()
    tta.add_argument('--tta', dest='tta', action='store_true')
    tta.add_argument('--no-tta', dest='tta', action='store_false')
    parser.set_defaults(tta=True)
    args = parser.parse_args()
    if args.save_every <= 0:
        parser.error('--save-every必须为正整数')
    return args


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if args.ckpt is not None:
        run_one(args)
    elif args.model == 'all':
        run_all(args)
    else:
        run_one(args)


if __name__ == '__main__':
    main()
