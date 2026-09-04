"""Batch evaluation adapter for the unmodified official VFIMamba repository.

Copy this file into the root of a clean MCG-NJU/VFIMamba checkout and run it
there.  It only adds dataset traversal and output handling; model construction,
checkpoint loading, padding, local refinement and inference all use official
VFIMamba modules.
"""

import argparse
import gc
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import numpy as np
import torch
from tqdm import tqdm

import config as cfg
from Trainer_finetune import Model
from benchmark.utils.padder import InputPadder


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.exr'}
EXCLUDED_DIRS = {
    'vfi', 'pred', 'prediction', 'predictions',
    'output', 'outputs', 'result', 'results',
}
DEFAULT_ROOT = Path('/home/zhenying/qhong/data/testdata')
DEFAULT_OUTPUT_ROOT = Path('/home/zhenying/qhong/sync/result')


def extract_number(path: Path) -> int:
    numbers = re.findall(r'(\d+)', path.name)
    return int(numbers[-1]) if numbers else -1


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def collect_sequences(root: Path, output: Path):
    sequences = []
    output_is_inside_root = is_within(output, root)
    for dirpath, dirnames, filenames in os.walk(root):
        folder = Path(dirpath).resolve()
        dirnames[:] = [
            name for name in dirnames
            if not name.startswith('.')
            and name.lower() not in EXCLUDED_DIRS
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
        frames.sort(key=lambda path: (extract_number(path), path.name))
        if len(frames) >= 2:
            sequences.append((folder, frames))
    sequences.sort(key=lambda item: str(item[0].relative_to(root)))
    return sequences


def read_openexr_bgr(path: Path) -> np.ndarray:
    """使用OpenEXR Python绑定读取RGB通道并返回BGR float32。"""
    try:
        import Imath
        import OpenEXR
    except ImportError as exc:
        raise RuntimeError(
            f'OpenCV无法读取EXR且OpenEXR Python模块不可用: {path}') from exc

    exr = OpenEXR.InputFile(str(path))
    try:
        header = exr.header()
        data_window = header['dataWindow']
        width = data_window.max.x - data_window.min.x + 1
        height = data_window.max.y - data_window.min.y + 1
        available = set(header['channels'])
        names = (
            ('B', 'G', 'R') if {'R', 'G', 'B'} <= available
            else ('b', 'g', 'r') if {'r', 'g', 'b'} <= available
            else None)
        if names is None:
            raise ValueError(
                f'EXR缺少RGB通道: {path}, channels={sorted(available)}')
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = [
            np.frombuffer(exr.channel(name, pixel_type), dtype=np.float32)
            .reshape(height, width)
            for name in names
        ]
        return np.stack(channels, axis=-1)
    finally:
        exr.close()


def to_model_float(image: np.ndarray) -> np.ndarray:
    """统一成模型输入的8-bit精度float32 [0,1]。"""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        normalized = image.astype(np.float32) / 65535.0
    else:
        normalized = image.astype(np.float32)
    quantized = (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)
    return quantized.astype(np.float32) / 255.0


def read_bgr(path: Path) -> np.ndarray:
    image = (
        read_openexr_bgr(path)
        if path.suffix.lower() == '.exr'
        else cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
    if image is None:
        raise FileNotFoundError(f'OpenCV无法读取: {path}')
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[..., :3]
    return to_model_float(image)


def resize_if_needed(image: np.ndarray, max_width: int,
                     max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    ratio = min(max_width / width, max_height / height)
    if ratio >= 1.0:
        return image
    size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def write_bgr(path: Path, image: np.ndarray, min_disk: bool) -> Path:
    clipped = np.clip(image, 0.0, 1.0)
    if min_disk:
        path = path.with_suffix('.webp')
        encoded = (clipped * 255).astype(np.uint8)
        params = [cv2.IMWRITE_WEBP_QUALITY, 101]
    elif path.suffix.lower() == '.exr':
        encoded = clipped.astype(np.float32)
        params = []
    elif path.suffix.lower() in ('.tif', '.tiff'):
        encoded = (clipped * 65535).astype(np.uint16)
        params = []
    else:
        encoded = (clipped * 255).astype(np.uint8)
        params = []
    if not cv2.imwrite(str(path), encoded, params):
        raise OSError(f'图像写入失败: {path}')
    return path


def output_indices(src0: Path, src1: Path, pair_index: int):
    index0, index1 = extract_number(src0), extract_number(src1)
    if index0 < 0 or index1 <= index0:
        index0, index1 = pair_index, pair_index + 1
    doubled0, doubled1 = index0 * 2, index1 * 2
    return doubled0, (doubled0 + doubled1) // 2, doubled1


def build_model(model_name: str) -> Model:
    if model_name == 'VFIMamba':
        cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F=32, depth=[2, 2, 2, 3, 3])
    else:
        cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba_S'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F=16, depth=[2, 2, 2, 3, 3])
    model = Model(-1)
    model.load_model()
    model.eval()
    return model


def evaluate_sequence(model: Model, frames, save_dir: Path, args):
    save_dir.mkdir(parents=True, exist_ok=True)
    pair_count = len(frames) - 1
    if args.max_pairs_per_scene > 0:
        pair_count = min(pair_count, args.max_pairs_per_scene)
    suffix = '.webp' if args.min_disk else frames[0].suffix.lower()
    if suffix == '.jpeg':
        suffix = '.jpg'
    timings = []

    for pair_index in tqdm(
            range(pair_count), desc=save_dir.parent.name,
            unit='pair', leave=False):
        src0, src1 = frames[pair_index], frames[pair_index + 1]
        image0 = resize_if_needed(
            read_bgr(src0), args.max_width, args.max_height)
        image1 = resize_if_needed(
            read_bgr(src1), args.max_width, args.max_height)
        if image0.shape != image1.shape:
            raise ValueError(
                f'相邻帧尺寸不一致: {src0}={image0.shape}, '
                f'{src1}={image1.shape}')

        tensor0 = torch.from_numpy(
            np.ascontiguousarray(image0.transpose(2, 0, 1))
        ).unsqueeze(0).cuda()
        tensor1 = torch.from_numpy(
            np.ascontiguousarray(image1.transpose(2, 0, 1))
        ).unsqueeze(0).cuda()
        padder = InputPadder(tensor0.shape, divisor=32)
        tensor0, tensor1 = padder.pad(tensor0, tensor1)

        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = model.inference(
                tensor0, tensor1, True,
                TTA=args.tta,
                fast_TTA=args.tta and args.fast_tta,
                timestep=0.5,
                scale=args.scale)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        prediction = padder.unpad(prediction)
        prediction_np = prediction[0].detach().cpu().numpy().transpose(1, 2, 0)

        index0, middle, index1 = output_indices(src0, src1, pair_index)
        if pair_index == 0:
            write_bgr(
                save_dir / f'{index0:06d}{suffix}', image0, args.min_disk)
        write_bgr(
            save_dir / f'{middle:06d}{suffix}', prediction_np, args.min_disk)
        write_bgr(
            save_dir / f'{index1:06d}{suffix}', image1, args.min_disk)
        del tensor0, tensor1, prediction, prediction_np, image0, image1

    torch.cuda.empty_cache()
    return {
        'input_frames': len(frames),
        'processed_pairs': pair_count,
        'output_frames': pair_count * 2 + 1 if pair_count else 0,
        'mean_inference_ms': (
            1000.0 * sum(timings) / len(timings) if timings else 0.0),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='使用未修改的官方VFIMamba代码批量评测固定测试集')
    parser.add_argument('--root', default=str(DEFAULT_ROOT))
    parser.add_argument('--output', default=None)
    parser.add_argument(
        '--model', choices=('VFIMamba', 'VFIMamba_S'), default='VFIMamba')
    parser.add_argument('--scale', type=float, default=0.0)
    parser.add_argument('--max-width', type=int, default=4096)
    parser.add_argument('--max-height', type=int, default=4096)
    parser.add_argument('--max-pairs-per-scene', type=int, default=0)
    parser.add_argument('--min-disk', action='store_true')
    tta = parser.add_mutually_exclusive_group()
    tta.add_argument('--tta', dest='tta', action='store_true')
    tta.add_argument('--no-tta', dest='tta', action='store_false')
    parser.set_defaults(tta=True)
    parser.add_argument(
        '--fast-tta', action='store_true',
        help='使用官方demo的batch合并TTA；不传则使用等价的串行TTA')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output else
        DEFAULT_OUTPUT_ROOT / (
            'VFIMamba_official_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    )
    if not root.is_dir():
        raise FileNotFoundError(f'测试集不存在: {root}')
    if min(args.max_width, args.max_height) <= 0:
        raise ValueError('--max-width和--max-height必须为正数')
    if args.max_pairs_per_scene < 0:
        raise ValueError('--max-pairs-per-scene不能为负数')

    sequences = collect_sequences(root, output)
    pair_count = sum(len(frames) - 1 for _, frames in sequences)
    print(f'[data] root={root}')
    print(f'[data] output={output}')
    print(f'[data] sequences={len(sequences)} adjacent_pairs={pair_count}')
    print('[data] output_format=' + (
        'lossless WebP' if args.min_disk else 'source extension'))
    print(f'[model] {args.model} local=True TTA={args.tta} '
          f'fast_TTA={args.fast_tta} scale={args.scale}')
    for folder, frames in sequences:
        print(f'  {folder.relative_to(root)}: {len(frames)} frames')
    if not sequences:
        raise ValueError('测试集下未找到有效图像序列')
    if args.dry_run:
        print('[dry-run] 未加载模型或写入结果')
        return

    output.mkdir(parents=True, exist_ok=True)
    model = build_model(args.model)
    started = time.perf_counter()
    results = {}
    for folder, frames in sequences:
        relative = folder.relative_to(root)
        results[str(relative)] = evaluate_sequence(
            model, frames, output / relative / 'vfi', args)
    elapsed = time.perf_counter() - started
    processed = sum(item['processed_pairs'] for item in results.values())
    summary = {
        'implementation': 'MCG-NJU/VFIMamba official',
        'model': args.model,
        'checkpoint': f'ckpt/{args.model}.pkl',
        'root': str(root),
        'output': str(output),
        'local': True,
        'tta': 'fast' if args.fast_tta and args.tta else (
            'standard' if args.tta else 'off'),
        'scale': args.scale,
        'min_disk': args.min_disk,
        'processed_pairs': processed,
        'elapsed_seconds': elapsed,
        'mean_seconds_per_pair': elapsed / processed if processed else 0.0,
        'scenes': results,
    }
    with (output / 'summary.json').open('w') as file:
        json.dump(
            summary, file,
            indent=None if args.min_disk else 2,
            separators=(',', ':') if args.min_disk else None,
            ensure_ascii=False)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f'[done] {processed} pairs, {elapsed / 60.0:.2f} min')
    print(f'[done] output: {output}')


if __name__ == '__main__':
    main()
