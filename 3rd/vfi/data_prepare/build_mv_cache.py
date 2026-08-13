#!/usr/bin/env python3
"""把teacher的mv1/mv0 EXR转换为训练专用的未压缩mmap缓存。

缓存布局（每个teacher scene内）::

    mv_cache_f16/
      <image_stem>.npy         # [H,W,4] float16, normalized, mv1再mv0
      <image_stem>.motion.npy  # [ceil(H/s),ceil(W/s)] float16, 像素运动残差

主缓存使用未压缩NPY，使Dataset可以通过mmap只读取随机crop对应的页面；
motion sidecar用于小物体感知裁剪，避免为了选crop先扫描整张flow。

用法::

    conda run -n vfi python data_prepare/build_mv_cache.py \
      --root /home/zhenying/qhong/data/ssd/vfi_database --workers 8
"""
import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from file_utils import read as fu_read  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s  %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
MV_EXTS = {'.exr'}
CACHE_VERSION = 1


def _frame_number(path: Path) -> Optional[int]:
    matches = re.findall(r'(\d+)', path.stem)
    return int(matches[-1]) if matches else None


def _numbered_files(directory: Path, extensions) -> Dict[int, Path]:
    result = {}
    if not directory.is_dir():
        return result
    for entry in os.scandir(directory):
        path = Path(entry.path)
        if (entry.is_file() and not entry.name.startswith('.')
                and path.suffix.lower() in extensions):
            number = _frame_number(path)
            if number is not None:
                result[number] = path
    return result


def _read_scene_lists(root: Path, list_names: Iterable[str]) -> List[str]:
    scenes = set()
    for value in list_names:
        list_path = Path(value)
        if not list_path.is_absolute():
            list_path = root / 'lists' / list_path
        if not list_path.is_file():
            logger.warning('清单不存在，跳过: %s', list_path)
            continue
        with open(list_path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                fields = line.rstrip('\n').split('\t')
                if len(fields) < 4 or fields[3] != '1':
                    continue
                scenes.add(fields[0])
    if not scenes:
        raise FileNotFoundError('没有从清单中找到has_mv=1的teacher scene')
    return sorted(scenes)


def _build_work(root: Path, scenes: Iterable[str],
                cache_dirname: str) -> Tuple[List[dict], List[str]]:
    work = []
    errors = []
    for rel in scenes:
        scene = root / rel
        images = _numbered_files(scene / 'image', IMG_EXTS)
        mv1 = _numbered_files(scene / 'mv1', MV_EXTS)
        mv0 = _numbered_files(scene / 'mv0', MV_EXTS)
        # s=1/t=0.5监督的GT不可能是scene首尾帧；很多数据也刻意不提供
        # 两个端点的MV，因此只生成训练真正会访问的中间帧。
        eligible = set(sorted(images)[1:-1])
        common = sorted(eligible & mv1.keys() & mv0.keys())
        missing = len(eligible) - len(common)
        if missing:
            errors.append(f'{rel}: {missing}/{len(eligible)} 可监督帧缺少mv1或mv0')
        output_dir = scene / cache_dirname
        for number in common:
            stem = images[number].stem
            work.append({
                'scene': rel,
                'image': str(images[number]),
                'mv1': str(mv1[number]),
                'mv0': str(mv0[number]),
                'cache': str(output_dir / f'{stem}.npy'),
                'preview': str(output_dir / f'{stem}.motion.npy'),
            })
    return work, errors


def _estimate_cache_bytes(work: List[dict], preview_stride: int) -> int:
    """每个scene读取一张图的header，估算完整未压缩缓存体积。"""
    scene_counts = {}
    for item in work:
        if item['scene'] not in scene_counts:
            scene_counts[item['scene']] = [0, item['image']]
        scene_counts[item['scene']][0] += 1
    total = 0
    for count, image_path in scene_counts.values():
        with Image.open(image_path) as image:
            width, height = image.size
        preview_h = (height + preview_stride - 1) // preview_stride
        preview_w = (width + preview_stride - 1) // preview_stride
        # 主缓存4通道float16 + motion float16 + 两个NPY header的裕量。
        total += count * (
            height * width * 4 * 2 + preview_h * preview_w * 2 + 512)
    return total


def _atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f'.{path.name}.tmp-{os.getpid()}-{os.urandom(4).hex()}')
    try:
        with open(temporary, 'wb') as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _motion_preview(normalized: np.ndarray, stride: int) -> np.ndarray:
    height, width = normalized.shape[:2]
    finite = np.isfinite(normalized).all(axis=-1)
    flow = normalized.astype(np.float32, copy=True)
    np.nan_to_num(flow, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    flow[..., (0, 2)] *= width
    flow[..., (1, 3)] *= height

    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    inside1 = (
        (xs + flow[..., 0] >= 0) & (xs + flow[..., 0] <= width - 1)
        & (ys + flow[..., 1] >= 0) & (ys + flow[..., 1] <= height - 1))
    inside0 = (
        (xs + flow[..., 2] >= 0) & (xs + flow[..., 2] <= width - 1)
        & (ys + flow[..., 3] >= 0) & (ys + flow[..., 3] <= height - 1))
    valid = finite & inside1 & inside0

    sparse_flow = flow[::8, ::8]
    sparse_valid = valid[::8, ::8]
    if sparse_valid.any():
        global_flow = np.median(sparse_flow[sparse_valid], axis=0)
    elif valid.any():
        global_flow = np.median(flow[valid], axis=0)
    else:
        global_flow = np.zeros(4, dtype=np.float32)
    residual1 = np.linalg.norm(
        flow[..., 0:2] - global_flow[0:2], axis=-1)
    residual0 = np.linalg.norm(
        flow[..., 2:4] - global_flow[2:4], axis=-1)
    motion = 0.5 * (residual1 + residual0)
    motion[~valid] = -np.inf

    preview_h = (height + stride - 1) // stride
    preview_w = (width + stride - 1) // stride
    padded = np.full(
        (preview_h * stride, preview_w * stride),
        -np.inf, dtype=np.float32)
    padded[:height, :width] = motion
    preview = padded.reshape(
        preview_h, stride, preview_w, stride).max(axis=(1, 3))
    preview[~np.isfinite(preview)] = np.nan
    np.clip(preview, 0, np.finfo(np.float16).max, out=preview)
    return preview.astype(np.float16)


def _convert_one(item: dict, preview_stride: int,
                 overwrite: bool) -> dict:
    cache_path = Path(item['cache'])
    preview_path = Path(item['preview'])
    if not overwrite and cache_path.is_file() and preview_path.is_file():
        if _verify_one(item, preview_stride) is None:
            return {'status': 'skipped', 'bytes': 0, 'scene': item['scene']}
    try:
        mv1 = fu_read(item['mv1'], type='flo')
        mv0 = fu_read(item['mv0'], type='flo')
        if mv1 is None or mv0 is None:
            raise ValueError('EXR解码结果为空')
        if (mv1.ndim != 3 or mv0.ndim != 3
                or mv1.shape[-1] < 2 or mv0.shape[-1] < 2
                or mv1.shape[:2] != mv0.shape[:2]):
            raise ValueError(f'MV尺寸/通道不匹配: mv1={mv1.shape}, mv0={mv0.shape}')

        normalized = np.concatenate(
            (mv1[..., :2], mv0[..., :2]), axis=-1).astype(np.float32)
        representable = (
            np.isfinite(normalized).all(axis=-1)
            & (np.abs(normalized) <= np.finfo(np.float16).max).all(axis=-1))
        normalized[~representable] = np.nan
        preview = _motion_preview(normalized, preview_stride)
        # HWC使随机crop的4个通道位于同一批磁盘页，比CHW mmap更适合训练访问。
        cache = normalized.astype(np.float16)
        _atomic_save(cache_path, cache)
        _atomic_save(preview_path, preview)
        written = cache_path.stat().st_size + preview_path.stat().st_size
        return {'status': 'written', 'bytes': written, 'scene': item['scene']}
    except Exception as exc:
        return {
            'status': 'failed', 'bytes': 0, 'scene': item['scene'],
            'error': f'{item["mv1"]}: {exc}',
        }


def _verify_one(item: dict, preview_stride: int) -> Optional[str]:
    try:
        cache = np.load(item['cache'], mmap_mode='r', allow_pickle=False)
        preview = np.load(item['preview'], mmap_mode='r', allow_pickle=False)
        if cache.ndim != 3 or cache.shape[-1] != 4 or cache.dtype != np.float16:
            raise ValueError(f'主缓存应为float16 [H,W,4]，实际{cache.dtype} {cache.shape}')
        expected = ((cache.shape[0] + preview_stride - 1) // preview_stride,
                    (cache.shape[1] + preview_stride - 1) // preview_stride)
        if preview.dtype != np.float16 or preview.shape != expected:
            raise ValueError(
                f'预览应为float16 {expected}，实际{preview.dtype} {preview.shape}')
        return None
    except Exception as exc:
        return f'{item["cache"]}: {exc}'


def _write_manifest(root: Path, cache_dirname: str, preview_stride: int,
                    scenes: int, files: int, written_bytes: int) -> Path:
    manifest = {
        'version': CACHE_VERSION,
        'cache_dirname': cache_dirname,
        'dtype': 'float16',
        'layout': 'HWC',
        'channels': ['mv1_x', 'mv1_y', 'mv0_x', 'mv0_y'],
        'units': 'normalized_by_source_width_height',
        'preview_stride': preview_stride,
        'scenes': scenes,
        'files': files,
        'written_bytes_this_run': written_bytes,
    }
    path = root / 'lists' / f'{cache_dirname}_manifest.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    with open(temporary, 'w') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    os.replace(temporary, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='将teacher EXR MV转换为训练专用的mmap float16 NPY缓存')
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument(
        '--lists', nargs='+', default=['teacher_train.txt', 'teacher_val.txt'],
        help='绝对路径或相对<root>/lists的清单名')
    parser.add_argument('--cache_dirname', default='mv_cache_f16')
    parser.add_argument('--preview_stride', type=int, default=4)
    parser.add_argument(
        '--workers', type=int,
        default=min(8, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verify', action='store_true',
                        help='生成结束后校验全部缓存header')
    parser.add_argument('--limit', type=int, default=None,
                        help='仅处理前N帧，用于首次小规模验证')
    args = parser.parse_args()
    if args.preview_stride <= 0 or args.workers <= 0:
        parser.error('--preview_stride和--workers必须为正整数')
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit必须为正整数')

    root = args.root.resolve()
    scenes = _read_scene_lists(root, args.lists)
    work, pairing_errors = _build_work(root, scenes, args.cache_dirname)
    if args.limit is not None:
        work = work[:args.limit]
    logger.info(
        '准备转换 %d scenes / %d frames，workers=%d，cache=%s',
        len(scenes), len(work), args.workers, args.cache_dirname)
    estimated_bytes = _estimate_cache_bytes(work, args.preview_stride)
    free_bytes = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
    logger.info(
        '预计完整缓存 %.1f GiB；当前磁盘可用 %.1f GiB（未扣除已存在缓存）',
        estimated_bytes / (1024 ** 3), free_bytes / (1024 ** 3))
    if estimated_bytes > free_bytes:
        logger.warning('完整缓存估算值大于当前可用空间，请先确认已有缓存占用或释放空间')
    for message in pairing_errors[:20]:
        logger.warning(message)
    if len(pairing_errors) > 20:
        logger.warning('另有 %d 个scene配对不完整', len(pairing_errors) - 20)

    totals = {'written': 0, 'skipped': 0, 'failed': 0, 'bytes': 0}
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(
            _convert_one, item, args.preview_stride, args.overwrite)
            for item in work]
        for future in tqdm(
                as_completed(futures), total=len(futures), unit='frame',
                desc='生成MV cache'):
            result = future.result()
            totals[result['status']] += 1
            totals['bytes'] += result['bytes']
            if result['status'] == 'failed':
                failures.append(result['error'])

    if args.verify:
        verify_errors = []
        for item in tqdm(work, unit='frame', desc='校验MV cache'):
            error = _verify_one(item, args.preview_stride)
            if error:
                verify_errors.append(error)
        failures.extend(verify_errors)
        logger.info('校验完成: ok=%d failed=%d',
                    len(work) - len(verify_errors), len(verify_errors))

    manifest = _write_manifest(
        root, args.cache_dirname, args.preview_stride,
        len(scenes), len(work) - totals['failed'], totals['bytes'])
    logger.info(
        '完成: written=%d skipped=%d failed=%d 本次写入=%.2f GiB',
        totals['written'], totals['skipped'], totals['failed'],
        totals['bytes'] / (1024 ** 3))
    logger.info('manifest: %s', manifest)
    if failures:
        failure_path = root / 'lists' / f'{args.cache_dirname}_failures.txt'
        with open(failure_path, 'w') as handle:
            handle.write('\n'.join(failures) + '\n')
        logger.error('失败明细: %s', failure_path)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
