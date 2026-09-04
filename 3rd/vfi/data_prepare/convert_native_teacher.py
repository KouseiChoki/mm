#!/usr/bin/env python3
"""Convert native Sintel/FlyingThings3D flow into the VFI teacher cache.

The generated scene layout is directly consumable by ``KouseiDataset``::

    teacher/<dataset>/<split>/<pass>/<sequence>/<view>/
      image/*.png
      mv_cache_f16/
        <stem>.npy          # float16 HWC: mv1(x,y), mv0(x,y), normalized
        <stem>.motion.npy   # float16 motion-aware crop preview
        <stem>.cycle.npy    # uint8 forward/backward confidence
      teacher_scene.json

``mv1`` is middle->previous and ``mv0`` is middle->next.  FlyingThings has
both native directions.  Sintel only publishes current->next flow, so its
middle->previous direction is obtained by conservative bilinear inversion;
holes, occlusions and conflicting splats remain NaN and are excluded by the
dataset valid mask.

No EXR intermediate is written.  This saves space and gives training the
fast mmap format it would otherwise build from EXR in a second pass.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


logger = logging.getLogger(__name__)
cv2.setNumThreads(1)

CACHE_DIRNAME = 'mv_cache_f16'
CONVERSION_VERSION = 1
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
PASS_DIRS = {
    'clean': 'frames_cleanpass',
    'final': 'frames_finalpass',
}


def _frame_number(path: Path) -> Optional[int]:
    matches = re.findall(r'(\d+)', path.stem)
    return int(matches[-1]) if matches else None


def _numbered_files(directory: Path, extensions: Iterable[str]) -> Dict[int, Path]:
    result: Dict[int, Path] = {}
    if not directory.is_dir():
        return result
    allowed = {value.lower() for value in extensions}
    for entry in os.scandir(directory):
        if entry.name.startswith('.') or not entry.is_file():
            continue
        path = Path(entry.path)
        if path.suffix.lower() not in allowed:
            continue
        number = _frame_number(path)
        if number is not None:
            result[number] = path
    return result


def read_flo(path: Path) -> np.ndarray:
    with open(path, 'rb') as handle:
        magic = np.fromfile(handle, np.float32, count=1)
        if len(magic) != 1 or not np.isclose(magic[0], 202021.25):
            raise ValueError(f'invalid FLO magic: {path}')
        width_height = np.fromfile(handle, np.int32, count=2)
        if len(width_height) != 2:
            raise ValueError(f'truncated FLO header: {path}')
        width, height = map(int, width_height)
        data = np.fromfile(handle, np.float32, count=height * width * 2)
    if data.size != height * width * 2:
        raise ValueError(f'truncated FLO payload: {path}')
    return data.reshape(height, width, 2)


def read_pfm(path: Path) -> np.ndarray:
    with open(path, 'rb') as handle:
        header = handle.readline().rstrip()
        if header not in {b'PF', b'Pf'}:
            raise ValueError(f'invalid PFM header: {path}')
        dimensions = handle.readline().strip().split()
        if len(dimensions) != 2:
            raise ValueError(f'invalid PFM dimensions: {path}')
        width, height = map(int, dimensions)
        scale = float(handle.readline().strip())
        dtype = np.dtype('<f4' if scale < 0 else '>f4')
        channels = 3 if header == b'PF' else 1
        data = np.fromfile(handle, dtype=dtype)
    expected = height * width * channels
    if data.size != expected:
        raise ValueError(
            f'truncated PFM payload: {path} ({data.size}/{expected})')
    return np.flipud(data.reshape(height, width, channels)).astype(
        np.float32, copy=False)


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


def _atomic_copy(source: Path, target: Path, overwrite: bool) -> None:
    if target.is_file() and not overwrite:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f'.{target.name}.tmp-{os.getpid()}-{os.urandom(4).hex()}')
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _link_or_copy(source: Path, target: Path, overwrite: bool) -> None:
    if target.exists():
        if not overwrite:
            return
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        _atomic_copy(source, target, overwrite=True)


def _valid_flow(flow: np.ndarray, max_abs_flow: float) -> np.ndarray:
    return (
        np.isfinite(flow).all(axis=-1)
        & (np.abs(flow) <= max_abs_flow).all(axis=-1))


def invert_forward_flow(
        forward: np.ndarray, source_valid: Optional[np.ndarray] = None,
        max_abs_flow: float = 10000.0,
        variance_alpha: float = 0.01, variance_beta: float = 0.5,
        cycle_alpha: float = 0.01, cycle_beta: float = 0.5) -> np.ndarray:
    """Conservatively invert source->target flow with bilinear splatting.

    Returned target->source vectors are NaN in disocclusions, outside the
    image, at ambiguous many-to-one splats, or where inverse cycle fails.
    """
    if forward.ndim != 3 or forward.shape[-1] != 2:
        raise ValueError(f'flow must be [H,W,2], got {forward.shape}')
    height, width = forward.shape[:2]
    valid = _valid_flow(forward, max_abs_flow)
    if source_valid is not None:
        if source_valid.shape != (height, width):
            raise ValueError(
                f'source_valid shape mismatch: {source_valid.shape}')
        valid &= source_valid.astype(bool)

    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.full((height, width, 2), np.nan, dtype=np.float32)
    vectors = forward[ys, xs].astype(np.float32, copy=False)
    qx = xs.astype(np.float32) + vectors[:, 0]
    qy = ys.astype(np.float32) + vectors[:, 1]
    x0 = np.floor(qx).astype(np.int32)
    y0 = np.floor(qy).astype(np.int32)

    weight_sum = np.zeros(height * width, dtype=np.float32)
    vector_sum = np.zeros((height * width, 2), dtype=np.float32)
    square_sum = np.zeros(height * width, dtype=np.float32)
    vector_square = np.square(vectors).sum(axis=-1)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        tx = x0 + dx
        ty = y0 + dy
        weights = ((1.0 - np.abs(qx - tx))
                   * (1.0 - np.abs(qy - ty))).astype(np.float32)
        inside = (
            (tx >= 0) & (tx < width) & (ty >= 0) & (ty < height)
            & (weights > 1e-6))
        if not inside.any():
            continue
        indices = ty[inside] * width + tx[inside]
        selected_weights = weights[inside]
        np.add.at(weight_sum, indices, selected_weights)
        np.add.at(
            vector_sum, indices,
            vectors[inside] * selected_weights[:, None])
        np.add.at(
            square_sum, indices,
            vector_square[inside] * selected_weights)

    covered = weight_sum > 0.05
    mean = np.zeros_like(vector_sum)
    mean[covered] = vector_sum[covered] / weight_sum[covered, None]
    variance = np.full(height * width, np.inf, dtype=np.float32)
    variance[covered] = np.maximum(
        0.0,
        square_sum[covered] / weight_sum[covered]
        - np.square(mean[covered]).sum(axis=-1))
    magnitude_sq = np.square(mean).sum(axis=-1)
    unambiguous = variance <= variance_alpha * magnitude_sq + variance_beta

    backward = -mean.reshape(height, width, 2)
    base_valid = (covered & unambiguous).reshape(height, width)
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32))
    map_x = grid_x + backward[..., 0]
    map_y = grid_y + backward[..., 1]
    sampled_forward = cv2.remap(
        forward.astype(np.float32, copy=False), map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sampled_valid = cv2.remap(
        valid.astype(np.uint8), map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    residual_sq = np.square(backward + sampled_forward).sum(axis=-1)
    cycle_magnitude_sq = (
        np.square(backward).sum(axis=-1)
        + np.square(sampled_forward).sum(axis=-1))
    cycle_ok = residual_sq <= cycle_alpha * cycle_magnitude_sq + cycle_beta
    target_valid = (
        base_valid & sampled_valid & np.isfinite(sampled_forward).all(axis=-1)
        & cycle_ok)
    backward[~target_valid] = np.nan
    return backward.astype(np.float32, copy=False)


def _normalize_pair(past: np.ndarray, future: np.ndarray) -> np.ndarray:
    if past.shape != future.shape or past.ndim != 3 or past.shape[-1] != 2:
        raise ValueError(f'flow pair shape mismatch: {past.shape}/{future.shape}')
    height, width = past.shape[:2]
    normalized = np.concatenate((past, future), axis=-1).astype(np.float32)
    normalized[..., (0, 2)] /= width
    normalized[..., (1, 3)] /= height
    # Keep direction validity independent.  Sintel's first/last cache has only
    # one native direction; neighbour cycle checks still need that direction.
    representable = (
        np.isfinite(normalized)
        & (np.abs(normalized) <= np.finfo(np.float16).max))
    normalized[~representable] = np.nan
    return normalized.astype(np.float16)


def motion_preview(normalized: np.ndarray, stride: int) -> np.ndarray:
    height, width = normalized.shape[:2]
    flow = normalized.astype(np.float32, copy=True)
    finite = np.isfinite(flow).all(axis=-1)
    np.nan_to_num(flow, copy=False)
    flow[..., (0, 2)] *= width
    flow[..., (1, 3)] *= height
    sparse_valid = finite[::8, ::8]
    if sparse_valid.any():
        global_flow = np.median(flow[::8, ::8][sparse_valid], axis=0)
    elif finite.any():
        global_flow = np.median(flow[finite], axis=0)
    else:
        global_flow = np.zeros(4, dtype=np.float32)
    motion = 0.5 * (
        np.linalg.norm(flow[..., :2] - global_flow[:2], axis=-1)
        + np.linalg.norm(flow[..., 2:] - global_flow[2:], axis=-1))
    motion[~finite] = -np.inf
    preview_h = (height + stride - 1) // stride
    preview_w = (width + stride - 1) // stride
    padded = np.full(
        (preview_h * stride, preview_w * stride), -np.inf, np.float32)
    padded[:height, :width] = motion
    preview = padded.reshape(
        preview_h, stride, preview_w, stride).max(axis=(1, 3))
    preview[~np.isfinite(preview)] = np.nan
    np.clip(preview, 0, np.finfo(np.float16).max, out=preview)
    return preview.astype(np.float16)


def direction_cycle_confidence(
        forward_normalized: np.ndarray,
        backward_normalized: Optional[np.ndarray],
        alpha: float = 0.05, beta: float = 1.0) -> np.ndarray:
    height, width = forward_normalized.shape[:2]
    if backward_normalized is None:
        return np.zeros((height, width), np.float32)
    forward = forward_normalized.astype(np.float32, copy=True)
    backward = backward_normalized.astype(np.float32, copy=True)
    forward[..., 0] *= width
    forward[..., 1] *= height
    backward[..., 0] *= width
    backward[..., 1] *= height
    finite_forward = np.isfinite(forward).all(axis=-1)
    finite_backward = np.isfinite(backward).all(axis=-1)
    np.nan_to_num(forward, copy=False)
    np.nan_to_num(backward, copy=False)
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32))
    map_x = xs + forward[..., 0]
    map_y = ys + forward[..., 1]
    inside = (
        (map_x >= 0) & (map_x <= width - 1)
        & (map_y >= 0) & (map_y <= height - 1))
    sampled = cv2.remap(
        backward, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sampled_valid = cv2.remap(
        finite_backward.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    residual_sq = np.square(forward + sampled).sum(axis=-1)
    magnitude_sq = (
        np.square(forward).sum(axis=-1)
        + np.square(sampled).sum(axis=-1))
    threshold = np.maximum(alpha * magnitude_sq + beta, 1e-6)
    confidence = threshold / (threshold + residual_sq)
    confidence[~(finite_forward & sampled_valid & inside)] = 0
    return np.clip(confidence, 0, 1)


def _write_cycle_sidecars(cache_dir: Path, stems: Sequence[str],
                          overwrite: bool) -> None:
    caches = {
        stem: np.load(cache_dir / f'{stem}.npy', mmap_mode='r',
                      allow_pickle=False)
        for stem in stems
    }
    for index, stem in enumerate(stems):
        output = cache_dir / f'{stem}.cycle.npy'
        if output.is_file() and not overwrite:
            continue
        current = caches[stem]
        previous = caches[stems[index - 1]] if index > 0 else None
        following = caches[stems[index + 1]] if index + 1 < len(stems) else None
        to_previous = direction_cycle_confidence(
            current[..., 0:2], None if previous is None else previous[..., 2:4])
        to_next = direction_cycle_confidence(
            current[..., 2:4], None if following is None else following[..., 0:2])
        confidence = np.minimum(to_previous, to_next)
        _atomic_save(output, np.rint(confidence * 255).astype(np.uint8))


def _copy_images(images: Dict[int, Path], target_scene: Path,
                 overwrite: bool) -> List[str]:
    names = []
    for number in sorted(images):
        source = images[number]
        target = target_scene / 'image' / source.name
        _atomic_copy(source, target, overwrite)
        names.append(source.stem)
    return names


def _write_metadata(target_scene: Path, metadata: dict) -> None:
    path = target_scene / 'teacher_scene.json'
    temporary = path.with_suffix('.json.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(temporary, 'w') as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    os.replace(temporary, path)


def _sintel_masks(training: Path, scene: str, number: int,
                  shape: Tuple[int, int]) -> np.ndarray:
    valid = np.ones(shape, dtype=bool)
    for directory in ('invalid', 'occlusions'):
        path = training / directory / scene / f'frame_{number:04d}.png'
        if not path.is_file():
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != shape:
            raise ValueError(f'invalid Sintel mask: {path}')
        valid &= mask == 0
    return valid


def convert_sintel_scene(item: dict) -> dict:
    source = Path(item['source'])
    target_root = Path(item['target_root'])
    scene = item['scene']
    passes = item['passes']
    overwrite = item['overwrite']
    preview_stride = item['preview_stride']
    training = source / 'Sintel' / 'training'
    flows = _numbered_files(training / 'flow' / scene, {'.flo'})
    images_by_pass = {
        render_pass: _numbered_files(
            training / render_pass / scene, IMAGE_EXTENSIONS)
        for render_pass in passes
    }
    image_numbers = sorted(set.intersection(
        *(set(value) for value in images_by_pass.values())))
    if len(image_numbers) < 3:
        raise ValueError(f'Sintel/{scene}: fewer than 3 common images')

    reference_scene = target_root / 'teacher' / 'Sintel' / passes[0] / scene
    stems = _copy_images(images_by_pass[passes[0]], reference_scene, overwrite)
    cache_dir = reference_scene / CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    forward_cache: Dict[int, np.ndarray] = {}
    for number in image_numbers:
        if number in flows:
            forward_cache[number] = read_flo(flows[number])
    for number in image_numbers:
        stem = images_by_pass[passes[0]][number].stem
        output = cache_dir / f'{stem}.npy'
        preview_output = cache_dir / f'{stem}.motion.npy'
        if output.is_file() and preview_output.is_file() and not overwrite:
            continue
        with Image.open(images_by_pass[passes[0]][number]) as image:
            width, height = image.size
        future = forward_cache.get(number)
        previous_forward = forward_cache.get(number - 1)
        if future is None:
            future = np.full((height, width, 2), np.nan, np.float32)
        if previous_forward is None:
            past = np.full((height, width, 2), np.nan, np.float32)
        else:
            source_valid = _sintel_masks(
                training, scene, number - 1, (height, width))
            past = invert_forward_flow(previous_forward, source_valid)
        cache = _normalize_pair(past, future)
        _atomic_save(output, cache)
        _atomic_save(preview_output, motion_preview(cache, preview_stride))
    _write_cycle_sidecars(cache_dir, stems, overwrite)
    metadata = {
        'conversion_version': CONVERSION_VERSION,
        'dataset': 'Sintel', 'source_split': 'training',
        'render_pass': passes[0],
        'mv1': 'conservative inverse of previous native forward flow',
        'mv0': 'native current-to-next .flo',
        'cache_units': 'normalized_by_width_height',
        'cache_channels': ['mv1_x', 'mv1_y', 'mv0_x', 'mv0_y'],
    }
    _write_metadata(reference_scene, metadata)

    for render_pass in passes[1:]:
        target_scene = target_root / 'teacher' / 'Sintel' / render_pass / scene
        _copy_images(images_by_pass[render_pass], target_scene, overwrite)
        target_cache = target_scene / CACHE_DIRNAME
        for source_file in cache_dir.iterdir():
            if source_file.is_file():
                _link_or_copy(source_file, target_cache / source_file.name, overwrite)
        pass_metadata = dict(metadata, render_pass=render_pass)
        _write_metadata(target_scene, pass_metadata)
    return {'dataset': 'Sintel', 'scene': scene, 'frames': len(image_numbers)}


def _flying_flow_files(source_scene: Path, direction: str,
                       view: str) -> Dict[int, Path]:
    return _numbered_files(source_scene / direction / view, {'.pfm'})


def convert_flying_scene(item: dict) -> dict:
    source = Path(item['source'])
    target_root = Path(item['target_root'])
    split, letter, sequence, view = (
        item['split'], item['letter'], item['sequence'], item['view'])
    passes = item['passes']
    overwrite = item['overwrite']
    preview_stride = item['preview_stride']
    dataset = source / 'flyingthings'
    images_by_pass = {
        render_pass: _numbered_files(
            dataset / PASS_DIRS[render_pass] / split / letter / sequence / view,
            IMAGE_EXTENSIONS)
        for render_pass in passes
    }
    image_numbers = sorted(set.intersection(
        *(set(value) for value in images_by_pass.values())))
    flow_scene = dataset / 'optical_flow' / split / letter / sequence
    past_files = _flying_flow_files(flow_scene, 'into_past', view)
    future_files = _flying_flow_files(flow_scene, 'into_future', view)
    if len(image_numbers) < 3:
        raise ValueError(
            f'FlyingThings/{split}/{letter}/{sequence}/{view}: <3 images')

    reference_scene = (
        target_root / 'teacher' / 'FlyingThings3D' / split / passes[0]
        / letter / sequence / view)
    stems = _copy_images(images_by_pass[passes[0]], reference_scene, overwrite)
    cache_dir = reference_scene / CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    for number in image_numbers:
        if number not in past_files or number not in future_files:
            raise FileNotFoundError(
                f'missing native flow for {split}/{letter}/{sequence}/{view}/{number}')
        stem = images_by_pass[passes[0]][number].stem
        output = cache_dir / f'{stem}.npy'
        preview_output = cache_dir / f'{stem}.motion.npy'
        if output.is_file() and preview_output.is_file() and not overwrite:
            continue
        past = read_pfm(past_files[number])[..., :2]
        future = read_pfm(future_files[number])[..., :2]
        cache = _normalize_pair(past, future)
        _atomic_save(output, cache)
        _atomic_save(preview_output, motion_preview(cache, preview_stride))
    _write_cycle_sidecars(cache_dir, stems, overwrite)
    metadata = {
        'conversion_version': CONVERSION_VERSION,
        'dataset': 'FlyingThings3D', 'source_split': split,
        'render_pass': passes[0], 'view': view,
        'mv1': 'native into_past PFM',
        'mv0': 'native into_future PFM',
        'cache_units': 'normalized_by_width_height',
        'cache_channels': ['mv1_x', 'mv1_y', 'mv0_x', 'mv0_y'],
    }
    _write_metadata(reference_scene, metadata)

    for render_pass in passes[1:]:
        target_scene = (
            target_root / 'teacher' / 'FlyingThings3D' / split / render_pass
            / letter / sequence / view)
        _copy_images(images_by_pass[render_pass], target_scene, overwrite)
        target_cache = target_scene / CACHE_DIRNAME
        for source_file in cache_dir.iterdir():
            if source_file.is_file():
                _link_or_copy(source_file, target_cache / source_file.name, overwrite)
        _write_metadata(target_scene, dict(metadata, render_pass=render_pass))
    return {
        'dataset': 'FlyingThings3D',
        'scene': f'{split}/{letter}/{sequence}/{view}',
        'frames': len(image_numbers),
    }


def discover_sintel(source: Path, args) -> List[dict]:
    scenes = sorted(
        path.name for path in (source / 'Sintel' / 'training' / 'flow').iterdir()
        if path.is_dir() and not path.name.startswith('.'))
    if args.limit_scenes is not None:
        scenes = scenes[:args.limit_scenes]
    return [{
        'source': str(source), 'target_root': str(args.root), 'scene': scene,
        'passes': args.passes, 'overwrite': args.overwrite,
        'preview_stride': args.preview_stride,
    } for scene in scenes]


def discover_flying(source: Path, args) -> List[dict]:
    base = source / 'flyingthings' / PASS_DIRS[args.passes[0]] / args.flying_split
    work = []
    for letter in sorted(path for path in base.iterdir()
                         if path.is_dir() and not path.name.startswith('.')):
        for sequence in sorted(path for path in letter.iterdir()
                               if path.is_dir() and not path.name.startswith('.')):
            for view in ('left', 'right'):
                if (sequence / view).is_dir():
                    work.append({
                        'source': str(source), 'target_root': str(args.root),
                        'split': args.flying_split, 'letter': letter.name,
                        'sequence': sequence.name, 'view': view,
                        'passes': args.passes, 'overwrite': args.overwrite,
                        'preview_stride': args.preview_stride,
                    })
    if args.limit_scenes is not None:
        work = work[:args.limit_scenes]
    return work


def _run_dataset(name: str, work: List[dict], worker, workers: int) -> List[dict]:
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, item): item for item in work}
        with tqdm(total=len(futures), desc=f'convert {name}', unit='scene') as bar:
            for future in as_completed(futures):
                item = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    failures.append(f'{item}: {exc}')
                bar.update(1)
    if failures:
        preview = '\n'.join(failures[:20])
        raise RuntimeError(
            f'{name}: {len(failures)}/{len(work)} scenes failed:\n{preview}')
    return results


def _write_manifest(root: Path, results: List[dict], args) -> Path:
    manifest = {
        'conversion_version': CONVERSION_VERSION,
        'source': str(args.source), 'target_root': str(root),
        'datasets': args.datasets, 'passes': args.passes,
        'flying_split': args.flying_split,
        'scenes': len(results),
        'frames': sum(item['frames'] for item in results),
        'results_by_dataset': {
            name: sum(1 for item in results if item['dataset'] == name)
            for name in sorted({item['dataset'] for item in results})
        },
    }
    output = root / 'lists' / 'native_teacher_manifest.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.json.tmp')
    with open(temporary, 'w') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    os.replace(temporary, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert native Sintel/FlyingThings into teacher mmap cache')
    parser.add_argument('--source', required=True, type=Path,
                        help='directory containing Sintel/ and flyingthings/')
    parser.add_argument('--root', required=True, type=Path,
                        help='vfi_database root')
    parser.add_argument('--datasets', nargs='+',
                        choices=('sintel', 'flyingthings'),
                        default=['sintel', 'flyingthings'])
    parser.add_argument('--passes', nargs='+', choices=('clean', 'final'),
                        default=['clean', 'final'])
    parser.add_argument('--flying_split', choices=('TRAIN', 'TEST'),
                        default='TRAIN')
    parser.add_argument('--workers', type=int,
                        default=min(8, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument('--preview_stride', type=int, default=4)
    parser.add_argument('--limit_scenes', type=int, default=None,
                        help='per dataset limit for smoke tests')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.workers <= 0 or args.preview_stride <= 0:
        parser.error('--workers and --preview_stride must be positive')
    if args.limit_scenes is not None and args.limit_scenes <= 0:
        parser.error('--limit_scenes must be positive')
    args.source = args.source.resolve()
    args.root = args.root.resolve()
    if not args.source.is_dir():
        parser.error(f'--source does not exist: {args.source}')
    args.root.mkdir(parents=True, exist_ok=True)

    all_results = []
    if 'sintel' in args.datasets:
        work = discover_sintel(args.source, args)
        logger.info('Sintel: %d scenes, passes=%s', len(work), args.passes)
        all_results.extend(_run_dataset(
            'Sintel', work, convert_sintel_scene, args.workers))
    if 'flyingthings' in args.datasets:
        work = discover_flying(args.source, args)
        logger.info('FlyingThings3D: %d views, split=%s, passes=%s',
                    len(work), args.flying_split, args.passes)
        all_results.extend(_run_dataset(
            'FlyingThings3D', work, convert_flying_scene, args.workers))
    manifest = _write_manifest(args.root, all_results, args)
    logger.info('done: %d scenes / %d frames; manifest=%s',
                len(all_results), sum(item['frames'] for item in all_results),
                manifest)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s  %(message)s',
        datefmt='%H:%M:%S')
    main()
