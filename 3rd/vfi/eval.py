"""使用训练 checkpoint 对固定测试集执行 2x 视频插帧。

默认扫描 ``/home/zhenying/qhong/data/testdata`` 下直接包含图像的目录，
对每对相邻帧预测 t=0.5 中间帧，并在输出目录中保存完整的 2N-1 帧序列。

示例：
    python eval.py \
        --ckpt ckpt/my_experiment/my_experiment_best.pkl
"""

import argparse
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

import config as cfg
from Trainer import Model
from file_utils import read, write
from model.warplayer import clear_warp_cache
from start import arch_from_model_section, extract_number, resize_if_needed, to_float


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.exr'}
DEFAULT_OUTPUT_ROOT = Path('/home/zhenying/qhong/sync/result')
DEFAULT_EXCLUDED_DIRS = {
    'vfi', 'pred', 'prediction', 'predictions',
    'output', 'outputs', 'result', 'results',
}


def resolve_model_yaml(ckpt_path: Path, explicit: str | None) -> Path:
    """按显式路径、checkpoint同名yaml、同目录model.yaml的顺序查找结构配置。"""
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'--model_yaml 不存在: {path}')
        return path

    candidates = (ckpt_path.with_suffix('.yaml'), ckpt_path.parent / 'model.yaml')
    for path in candidates:
        if path.is_file():
            return path
    expected = ' 或 '.join(str(path) for path in candidates)
    raise FileNotFoundError(
        f'未找到 checkpoint 对应的模型结构 YAML，请使用 --model_yaml；期望: {expected}')


def build_model(ckpt_path: Path, model_yaml: Path) -> Model:
    """按训练归档 YAML 构建同构模型并仅加载网络权重。"""
    with model_yaml.open() as file:
        config = yaml.safe_load(file)
    model_config = config.get('model', config)
    cfg.MODEL_CONFIG['LOGNAME'] = config.get('exp_name', ckpt_path.stem)
    cfg.MODEL_CONFIG['MODEL_ARCH'] = arch_from_model_section(model_config)

    print(f'[model] checkpoint: {ckpt_path}')
    print(f'[model] structure:  {model_yaml}')
    print(f'[model] version={model_config.get("version", 2)} '
          f'blend={model_config.get("blend_mode", "soft")}')

    model = Model()
    model.load_model(str(ckpt_path))
    # 评估不需要优化器以及 checkpoint 中保留的训练状态。
    model.loaded_checkpoint = None
    model.optimG = None
    model.eval()
    gc.collect()
    if model._dev.type == 'cuda':
        torch.cuda.empty_cache()
    return model


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def collect_sequences(root: Path, output: Path) -> list[tuple[Path, list[Path]]]:
    """递归发现直接包含图像的目录，同时排除历史输出和本次输出目录。"""
    sequences = []
    output = output.resolve()
    output_is_inside_root = is_within(output, root)
    for dirpath, dirnames, filenames in os.walk(root):
        folder = Path(dirpath).resolve()
        dirnames[:] = [
            name for name in dirnames
            if not name.startswith('.')
            and name.lower() not in DEFAULT_EXCLUDED_DIRS
            and not (
                output_is_inside_root
                and is_within((folder / name).resolve(), output)
            )
        ]
        if output_is_inside_root and is_within(folder, output):
            dirnames[:] = []
            continue

        frames = [
            folder / name for name in filenames
            if not name.startswith('.') and Path(name).suffix.lower() in IMAGE_EXTS
        ]
        frames.sort(key=lambda path: (extract_number(str(path)), path.name))
        if len(frames) >= 2:
            sequences.append((folder, frames))
    sequences.sort(key=lambda item: str(item[0].relative_to(root)))
    return sequences


def tensor_from_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(
        np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).to(
            device=device, dtype=torch.float32)


def output_indices(src0: Path, src1: Path, pair_index: int) -> tuple[int, int, int]:
    index0 = extract_number(str(src0))
    index1 = extract_number(str(src1))
    if index0 < 0 or index1 < 0 or index1 <= index0:
        index0, index1 = pair_index, pair_index + 1
    doubled0, doubled1 = index0 * 2, index1 * 2
    return doubled0, (doubled0 + doubled1) // 2, doubled1


def write_image(path: Path, image: np.ndarray) -> None:
    write(str(path), np.clip(image, 0.0, 1.0), dtype='image')


def evaluate_sequence(model: Model, frames: list[Path], save_dir: Path,
                      args) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)
    pair_times = []
    processed_pairs = min(
        len(frames) - 1,
        args.max_pairs_per_scene if args.max_pairs_per_scene > 0 else len(frames) - 1)

    first_extension = frames[0].suffix.lower()
    if first_extension == '.jpeg':
        first_extension = '.jpg'

    for pair_index in tqdm(
            range(processed_pairs), desc=save_dir.parent.name, unit='pair', leave=False):
        src0, src1 = frames[pair_index], frames[pair_index + 1]
        image0 = to_float(resize_if_needed(
            read(str(src0), type='image'), max_w=args.max_width, max_h=args.max_height))
        image1 = to_float(resize_if_needed(
            read(str(src1), type='image'), max_w=args.max_width, max_h=args.max_height))
        if image0.shape != image1.shape:
            raise ValueError(
                f'相邻帧尺寸不一致: {src0}={image0.shape}, {src1}={image1.shape}')

        tensor0 = tensor_from_image(image0, model._dev)
        tensor1 = tensor_from_image(image1, model._dev)
        if model._dev.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = model.inference(
                tensor0, tensor1, local=args.local,
                TTA=args.tta, fast_TTA=args.fast_tta,
                timestep=0.5, scale=args.scale, return_debug=False)
        if model._dev.type == 'cuda':
            torch.cuda.synchronize()
        pair_times.append(time.perf_counter() - started)

        prediction_np = prediction[0].detach().cpu().numpy().transpose(1, 2, 0)
        index0, middle_index, index1 = output_indices(src0, src1, pair_index)
        if pair_index == 0:
            write_image(save_dir / f'{index0:06d}{first_extension}', image0)
        write_image(save_dir / f'{middle_index:06d}{first_extension}', prediction_np)
        write_image(save_dir / f'{index1:06d}{first_extension}', image1)

        del tensor0, tensor1, prediction, prediction_np, image0, image1

    clear_warp_cache()
    if model._dev.type == 'cuda':
        torch.cuda.empty_cache()
    elif model._dev.type == 'mps':
        torch.mps.synchronize()
        torch.mps.empty_cache()

    return {
        'input_frames': len(frames),
        'processed_pairs': processed_pairs,
        'output_frames': processed_pairs * 2 + 1 if processed_pairs else 0,
        'mean_inference_ms': (
            1000.0 * sum(pair_times) / len(pair_times) if pair_times else 0.0),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='使用训练 checkpoint 对测试图像序列执行 2x 插帧')
    parser.add_argument('--ckpt', required=True, type=str,
                        help='训练生成的 .pkl/.pth checkpoint')
    parser.add_argument(
        '--output', type=str, default=None,
        help='测试结果输出目录；默认写入 '
             '/home/zhenying/qhong/sync/result/<当前日期时间>')
    parser.add_argument(
        '--root', type=str, default='/home/zhenying/qhong/data/testdata',
        help='测试集根目录（默认 /home/zhenying/qhong/data/testdata）')
    parser.add_argument('--model_yaml', '--model-yaml', default=None, type=str,
                        help='模型结构 YAML；默认查找 checkpoint 同名 YAML 或同目录 model.yaml')
    parser.add_argument('--scale', default=0.0, type=float)
    parser.add_argument('--max_width', '--max-width', default=4096, type=int)
    parser.add_argument('--max_height', '--max-height', default=4096, type=int)
    parser.add_argument(
        '--max_pairs_per_scene', '--max-pairs-per-scene', default=0, type=int,
        help='每个序列最多测试的相邻帧对数量；0=全部，用于快速检查')
    parser.add_argument('--dry_run', '--dry-run', action='store_true',
                        help='只扫描并打印测试集，不加载模型或写输出')

    local_group = parser.add_mutually_exclusive_group()
    local_group.add_argument('--local', dest='local', action='store_true')
    local_group.add_argument('--no-local', dest='local', action='store_false')
    parser.set_defaults(local=True)

    tta_group = parser.add_mutually_exclusive_group()
    tta_group.add_argument('--tta', dest='tta', action='store_true',
                           help='启用标准双次 TTA')
    tta_group.add_argument('--no-tta', dest='tta', action='store_false',
                           help='关闭标准 TTA')
    parser.set_defaults(tta=True)
    parser.add_argument('--fast_tta', '--fast-tta', action='store_true',
                        help='使用 batch 合并快速 TTA，优先于标准 TTA')
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else DEFAULT_OUTPUT_ROOT / datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'测试集目录不存在: {root}')
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'checkpoint 不存在: {ckpt_path}')
    if output == root:
        raise ValueError('--output 不能与 --root 相同')
    if args.max_width <= 0 or args.max_height <= 0:
        raise ValueError('--max_width 和 --max_height 必须为正整数')
    if args.max_pairs_per_scene < 0:
        raise ValueError('--max_pairs_per_scene 不能为负数')

    model_yaml = resolve_model_yaml(ckpt_path, args.model_yaml)
    sequences = collect_sequences(root, output)
    pair_count = sum(len(frames) - 1 for _, frames in sequences)
    if not sequences:
        raise ValueError(f'测试集下没有发现至少包含两帧的图像序列: {root}')
    print(f'[data] root={root}')
    print(f'[data] output={output}')
    print(f'[data] sequences={len(sequences)} adjacent_pairs={pair_count}')
    for folder, frames in sequences:
        print(f'  {folder.relative_to(root)}: {len(frames)} frames / {len(frames) - 1} pairs')
    if args.dry_run:
        print('[dry-run] 扫描完成，未加载模型或写入输出')
        return

    output.mkdir(parents=True, exist_ok=True)
    model = build_model(ckpt_path, model_yaml)
    tta_mode = 'fast' if args.fast_tta else ('standard' if args.tta else 'off')
    print(f'[eval] device={model._dev} local={args.local} TTA={tta_mode}')

    started = time.perf_counter()
    scene_results = {}
    for folder, frames in sequences:
        relative = folder.relative_to(root)
        scene_results[str(relative)] = evaluate_sequence(
            model, frames, output / relative / 'vfi', args)

    elapsed = time.perf_counter() - started
    processed_pairs = sum(item['processed_pairs'] for item in scene_results.values())
    summary = {
        'checkpoint': str(ckpt_path),
        'model_yaml': str(model_yaml),
        'test_root': str(root),
        'output': str(output),
        'device': str(model._dev),
        'local': args.local,
        'tta': tta_mode,
        'scale': args.scale,
        'sequence_count': len(sequences),
        'processed_pairs': processed_pairs,
        'elapsed_seconds': elapsed,
        'mean_seconds_per_pair': elapsed / processed_pairs if processed_pairs else 0.0,
        'scenes': scene_results,
    }
    with (output / 'summary.json').open('w') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f'[done] {processed_pairs} pairs, {elapsed / 60.0:.2f} min')
    print(f'[done] output:  {output}')
    print(f'[done] summary: {output / "summary.json"}')


if __name__ == '__main__':
    main()
