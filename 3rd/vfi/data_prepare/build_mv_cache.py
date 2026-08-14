#!/usr/bin/env python3
"""把teacher的mv1/mv0 EXR转换为训练专用的未压缩mmap缓存。

缓存布局（每个teacher scene内）::

    mv_cache_f16/
      <image_stem>.npy         # [H,W,4] float16, normalized, mv1再mv0
      <image_stem>.motion.npy  # [ceil(H/s),ceil(W/s)] float16, 像素运动残差
      <image_stem>.cycle.npy   # [H,W] uint8, min(前/后向cycle置信度)

主缓存使用未压缩NPY，使Dataset可以通过mmap只读取随机crop对应的页面；
motion sidecar用于小物体感知裁剪，避免为了选crop先扫描整张flow。
cycle sidecar用当前帧到相邻帧的MV和相邻帧的反向MV做真正的
forward-backward consistency，训练时可用hard/soft方式过滤遮挡与错误标签。

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
import cv2
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
CACHE_VERSION = 2
cv2.setNumThreads(1)


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


def _build_work(root: Path, scenes: Iterable[str], cache_dirname: str,
                cycle_root: Optional[Path] = None
                ) -> Tuple[List[dict], List[str]]:
    work = []
    errors = []
    for rel in scenes:
        scene = root / rel
        images = _numbered_files(scene / 'image', IMG_EXTS)
        mv1 = _numbered_files(scene / 'mv1', MV_EXTS)
        mv0 = _numbered_files(scene / 'mv0', MV_EXTS)
        # s=1/t=0.5监督的GT不可能是scene首尾帧；很多数据也刻意不提供
        # 两个端点的MV，因此只生成训练真正会访问的中间帧。
        ordered = sorted(images)
        eligible = set(ordered[1:-1])
        common = sorted(eligible & mv1.keys() & mv0.keys())
        missing = len(eligible) - len(common)
        if missing:
            errors.append(f'{rel}: {missing}/{len(eligible)} 可监督帧缺少mv1或mv0')
        output_dir = scene / cache_dirname
        cycle_output_dir = (
            cycle_root / rel if cycle_root is not None else output_dir)
        positions = {number: index for index, number in enumerate(ordered)}
        for number in common:
            position = positions[number]
            previous_number = ordered[position - 1]
            next_number = ordered[position + 1]
            stem = images[number].stem
            work.append({
                'scene': rel,
                'image': str(images[number]),
                'mv1': str(mv1[number]),
                'mv0': str(mv0[number]),
                'cache': str(output_dir / f'{stem}.npy'),
                'preview': str(output_dir / f'{stem}.motion.npy'),
                'cycle': str(cycle_output_dir / f'{stem}.cycle.npy'),
                'previous_cache': str(
                    output_dir / f'{images[previous_number].stem}.npy'),
                'next_cache': str(
                    output_dir / f'{images[next_number].stem}.npy'),
            })
    return work, errors


def _estimate_cache_bytes(work: List[dict], preview_stride: int,
                          cycle_only: bool = False) -> int:
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
        if cycle_only:
            total += count * (height * width + 256)
        else:
            # 主缓存4通道float16 + cycle 1通道uint8 + motion float16
            # + 三个NPY header的裕量。
            total += count * (
                height * width * 9 + preview_h * preview_w * 2 + 768)
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


def _read_neighbor_direction(item: dict, direction: str,
                             expected_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """读取相邻帧指回当前帧的方向MV（归一化单位）。"""
    if direction == 'previous':
        cache_key, channels = 'previous_cache', slice(2, 4)
    elif direction == 'next':
        cache_key, channels = 'next_cache', slice(0, 2)
    else:
        raise ValueError(f'unknown cycle direction: {direction}')

    cache_path = Path(item[cache_key])
    # 不在cycle生成期回退读EXR：那会重新引入训练前已消除的
    # 随机长延迟。scene端点本来就不作teacher GT；相邻端点时
    # 保守返回None，使第二/倒数第二帧只参与图像重建。
    if not cache_path.is_file():
        return None
    cached = np.load(cache_path, mmap_mode='r', allow_pickle=False)
    if cached.ndim != 3 or cached.shape[-1] != 4:
        raise ValueError(f'相邻MV cache形状错误: {cache_path} {cached.shape}')
    flow = np.asarray(cached[..., channels], dtype=np.float32)
    if flow.shape[:2] != expected_shape:
        raise ValueError(
            f'相邻MV尺寸{flow.shape[:2]}与当前帧{expected_shape}不一致')
    return flow


def _direction_cycle_confidence(forward_normalized: np.ndarray,
                                backward_normalized: Optional[np.ndarray],
                                alpha: float, beta: float) -> np.ndarray:
    """Return threshold/(threshold+residual) so 0.5 is the hard pass boundary."""
    height, width = forward_normalized.shape[:2]
    if backward_normalized is None:
        return np.zeros((height, width), dtype=np.float32)

    forward = forward_normalized.astype(np.float32, copy=True)
    backward = backward_normalized.astype(np.float32, copy=True)
    forward[..., 0] *= width
    forward[..., 1] *= height
    backward[..., 0] *= width
    backward[..., 1] *= height
    finite_forward = np.isfinite(forward).all(axis=-1)
    finite_backward = np.isfinite(backward).all(axis=-1)
    np.nan_to_num(forward, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(backward, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32))
    map_x = xs + forward[..., 0]
    map_y = ys + forward[..., 1]
    inside = (
        (map_x >= 0) & (map_x <= width - 1)
        & (map_y >= 0) & (map_y <= height - 1))
    sampled_backward = cv2.remap(
        backward, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE)
    sampled_finite = np.isfinite(sampled_backward).all(axis=-1)
    sampled_source_finite = cv2.remap(
        finite_backward.astype(np.uint8), map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    np.nan_to_num(
        sampled_backward, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    residual_sq = np.square(forward + sampled_backward).sum(axis=-1)
    magnitude_sq = (
        np.square(forward).sum(axis=-1)
        + np.square(sampled_backward).sum(axis=-1))
    threshold = np.maximum(alpha * magnitude_sq + beta, 1e-6)
    confidence = threshold / (threshold + residual_sq)
    valid = (
        finite_forward & inside & sampled_finite & sampled_source_finite)
    confidence[~valid] = 0.0
    return np.clip(confidence, 0.0, 1.0)


def _build_cycle_one(item: dict, alpha: float, beta: float,
                     overwrite: bool) -> dict:
    cycle_path = Path(item['cycle'])
    if not overwrite and cycle_path.is_file():
        error = _verify_cycle_one(item)
        if error is None:
            cycle = np.load(cycle_path, mmap_mode='r', allow_pickle=False)
            confidence = cycle.astype(np.float32) / 255.0
            hard = cycle >= 128
            return {
                'status': 'skipped', 'bytes': 0, 'scene': item['scene'],
                'frame': Path(item['cache']).stem,
                'hard_valid_ratio': float(hard.mean()),
                'mean_confidence': float(confidence.mean()),
                'excluded': not bool(hard.any()),
            }
    try:
        current = np.load(item['cache'], mmap_mode='r', allow_pickle=False)
        if current.ndim != 3 or current.shape[-1] != 4:
            raise ValueError(f'当前MV cache形状错误: {current.shape}')
        shape = current.shape[:2]
        previous = _read_neighbor_direction(item, 'previous', shape)
        following = _read_neighbor_direction(item, 'next', shape)
        confidence_previous = _direction_cycle_confidence(
            np.asarray(current[..., 0:2]), previous, alpha, beta)
        confidence_next = _direction_cycle_confidence(
            np.asarray(current[..., 2:4]), following, alpha, beta)
        confidence = np.minimum(confidence_previous, confidence_next)
        cycle = np.rint(confidence * 255.0).astype(np.uint8)
        _atomic_save(cycle_path, cycle)
        hard = cycle >= 128
        return {
            'status': 'written', 'bytes': cycle_path.stat().st_size,
            'scene': item['scene'], 'frame': Path(item['cache']).stem,
            'hard_valid_ratio': float(hard.mean()),
            'mean_confidence': float((cycle.astype(np.float32) / 255.0).mean()),
            'excluded': not bool(hard.any()),
        }
    except Exception as exc:
        return {
            'status': 'failed', 'bytes': 0, 'scene': item['scene'],
            'frame': Path(item['cache']).stem,
            'hard_valid_ratio': 0.0, 'mean_confidence': 0.0,
            'excluded': True, 'error': f'{item["cache"]}: {exc}',
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


def _verify_cycle_one(item: dict) -> Optional[str]:
    try:
        cache = np.load(item['cache'], mmap_mode='r', allow_pickle=False)
        cycle = np.load(item['cycle'], mmap_mode='r', allow_pickle=False)
        expected = cache.shape[:2]
        if cycle.dtype != np.uint8 or cycle.shape != expected:
            raise ValueError(
                f'cycle应为uint8 {expected}，实际{cycle.dtype} {cycle.shape}')
        return None
    except Exception as exc:
        return f'{item["cycle"]}: {exc}'


def _write_manifest(root: Path, cache_dirname: str, preview_stride: int,
                    scenes: int, files: int, written_bytes: int,
                    cycle_alpha: float, cycle_beta: float,
                    cycle_root: Optional[Path]) -> Path:
    manifest = {
        'version': CACHE_VERSION,
        'cache_dirname': cache_dirname,
        'dtype': 'float16',
        'layout': 'HWC',
        'channels': ['mv1_x', 'mv1_y', 'mv0_x', 'mv0_y'],
        'units': 'normalized_by_source_width_height',
        'preview_stride': preview_stride,
        'cycle_channels': ['min_to_previous_to_next'],
        'cycle_dtype': 'uint8',
        'cycle_scale': 255,
        'cycle_formula': 'threshold/(threshold+fb_residual_squared)',
        'cycle_hard_threshold': 0.5,
        'cycle_alpha': cycle_alpha,
        'cycle_beta': cycle_beta,
        'cycle_root': str(cycle_root) if cycle_root is not None else None,
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


def _scene_group(scene: str) -> str:
    parts = Path(scene).parts
    domain = next(
        (name for name in ('clean', 'final')
         if any(part.lower().startswith(name) for part in parts)),
        'other')
    fps = next(
        (match.group(1) for part in reversed(parts)
         if (match := re.fullmatch(r'(\d+)fps', part.lower()))),
        'unknown')
    return f'{domain}/{fps}fps'


def _write_cycle_report(root: Path, cache_dirname: str, results: List[dict],
                        pairing_errors: List[str], alpha: float,
                        beta: float, report_root: Optional[Path] = None) -> Path:
    groups = {}
    excluded = []
    for result in results:
        group = groups.setdefault(_scene_group(result['scene']), {
            'frames': 0, 'hard_valid_ratio_sum': 0.0,
            'mean_confidence_sum': 0.0, 'excluded_frames': 0})
        group['frames'] += 1
        group['hard_valid_ratio_sum'] += result['hard_valid_ratio']
        group['mean_confidence_sum'] += result['mean_confidence']
        if result['excluded']:
            group['excluded_frames'] += 1
            excluded.append(result['scene'] + '/' + result['frame'])
    for group in groups.values():
        count = max(group['frames'], 1)
        group['hard_valid_ratio_mean'] = group.pop(
            'hard_valid_ratio_sum') / count
        group['mean_confidence'] = group.pop('mean_confidence_sum') / count
    report = {
        'cache_version': CACHE_VERSION,
        'cycle_alpha': alpha,
        'cycle_beta': beta,
        'hard_threshold': 0.5,
        'frames': len(results),
        'groups': groups,
        'fully_excluded_frames': excluded,
        'pairing_errors': pairing_errors,
    }
    report_directory = (
        report_root if report_root is not None else root / 'lists')
    report_directory.mkdir(parents=True, exist_ok=True)
    path = report_directory / f'{cache_dirname}_cycle_report.json'
    temporary = path.with_suffix('.json.tmp')
    with open(temporary, 'w') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
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
    parser.add_argument(
        '--cycle_root', type=Path, default=None,
        help='可选的cycle sidecar独立根目录；内部保留scene相对路径')
    parser.add_argument('--preview_stride', type=int, default=4)
    parser.add_argument('--cycle_alpha', type=float, default=0.05)
    parser.add_argument('--cycle_beta', type=float, default=1.0)
    parser.add_argument(
        '--workers', type=int,
        default=min(8, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--overwrite_cycle', action='store_true',
        help='仅重建cycle sidecar，不重写已有的大型主缓存')
    parser.add_argument(
        '--cycle_only', action='store_true',
        help='跳过主缓存生成/逐帧预检，仅在已有主缓存上生成cycle')
    parser.add_argument('--verify', action='store_true',
                        help='生成结束后校验全部缓存header')
    parser.add_argument('--limit', type=int, default=None,
                        help='仅处理前N帧，用于首次小规模验证')
    parser.add_argument(
        '--scene_contains', default=None,
        help='仅处理相对路径包含该字符串的scene（用于定点审计）')
    args = parser.parse_args()
    if args.preview_stride <= 0 or args.workers <= 0:
        parser.error('--preview_stride和--workers必须为正整数')
    if args.cycle_alpha < 0 or args.cycle_beta <= 0:
        parser.error('--cycle_alpha必须>=0，--cycle_beta必须>0')
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit必须为正整数')

    root = args.root.resolve()
    cycle_root = args.cycle_root.resolve() if args.cycle_root else None
    scenes = _read_scene_lists(root, args.lists)
    work, pairing_errors = _build_work(
        root, scenes, args.cache_dirname, cycle_root=cycle_root)
    if args.scene_contains:
        work = [item for item in work
                if args.scene_contains in item['scene']]
        if not work:
            parser.error(
                f'--scene_contains={args.scene_contains!r}未匹配任何帧')
    if args.limit is not None:
        work = work[:args.limit]
    logger.info(
        '准备转换 %d scenes / %d frames，workers=%d，cache=%s',
        len(scenes), len(work), args.workers, args.cache_dirname)
    estimated_bytes = _estimate_cache_bytes(
        work, args.preview_stride, cycle_only=args.cycle_only)
    output_filesystem = cycle_root if cycle_root is not None else root
    output_filesystem.mkdir(parents=True, exist_ok=True)
    free_bytes = (
        os.statvfs(output_filesystem).f_bavail
        * os.statvfs(output_filesystem).f_frsize)
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
    if args.cycle_only:
        # 不先在NTFS上串行stat数万个文件；各worker读当前帧时
        # 会精确报告任何缺失/损坏的主缓存。
        totals['skipped'] = len(work)
        logger.info('--cycle_only: 跳过主缓存生成与串行路径预检')
    else:
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

    # 主缓存全部就绪后再生成cycle，相邻帧只走NPY mmap。
    cycle_totals = {'written': 0, 'skipped': 0, 'failed': 0, 'bytes': 0}
    cycle_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(
            _build_cycle_one, item, args.cycle_alpha, args.cycle_beta,
            args.overwrite or args.overwrite_cycle) for item in work]
        for future in tqdm(
                as_completed(futures), total=len(futures), unit='frame',
                desc='生成cycle confidence'):
            result = future.result()
            cycle_totals[result['status']] += 1
            cycle_totals['bytes'] += result['bytes']
            cycle_results.append(result)
            if result['status'] == 'failed':
                failures.append(result['error'])

    if args.verify:
        verify_errors = []
        for item in tqdm(work, unit='frame', desc='校验MV cache'):
            error = _verify_one(item, args.preview_stride)
            if error:
                verify_errors.append(error)
            cycle_error = _verify_cycle_one(item)
            if cycle_error:
                verify_errors.append(cycle_error)
        failures.extend(verify_errors)
        logger.info('校验完成: ok=%d failed=%d',
                    len(work) - len(verify_errors), len(verify_errors))

    manifest = None
    if not args.cycle_only:
        manifest = _write_manifest(
            root, args.cache_dirname, args.preview_stride,
            len(scenes), len(work) - totals['failed'],
            totals['bytes'] + cycle_totals['bytes'],
            args.cycle_alpha, args.cycle_beta, cycle_root)
    cycle_report = _write_cycle_report(
        root, args.cache_dirname, cycle_results, pairing_errors,
        args.cycle_alpha, args.cycle_beta,
        report_root=cycle_root if cycle_root is not None else None)
    logger.info(
        '完成: written=%d skipped=%d failed=%d 本次写入=%.2f GiB',
        totals['written'], totals['skipped'], totals['failed'],
        totals['bytes'] / (1024 ** 3))
    if manifest is not None:
        logger.info('manifest: %s', manifest)
    logger.info(
        'cycle: written=%d skipped=%d failed=%d fully_excluded=%d report=%s',
        cycle_totals['written'], cycle_totals['skipped'],
        cycle_totals['failed'], sum(r['excluded'] for r in cycle_results),
        cycle_report)
    if failures:
        failure_path = root / 'lists' / f'{args.cache_dirname}_failures.txt'
        with open(failure_path, 'w') as handle:
            handle.write('\n'.join(failures) + '\n')
        logger.error('失败明细: %s', failure_path)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
