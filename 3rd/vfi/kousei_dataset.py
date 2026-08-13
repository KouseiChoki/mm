'''
KouseiDataset v2 — 多tier清单 + 运行时 framestep/timestep 采样 + teacher flow监督
==================================================================================
配合 build_lists.py 生成的 scene 级清单使用。

清单行格式 (tab分隔): scene相对路径  帧数  tier  has_mv  fps

采样逻辑 (train):
  1. index → scene (按各scene可采三元组数加权)
  2. 随机 framestep s ∈ framesteps (窗口 (i, i+2s))
  3. timestep: 以 t_half_prob 概率取 t=0.5 (gt=i+s);
     否则在窗口内随机取 gt=j, t=(j-i)/(2s)  ← 任意timestep的真实GT监督
  4. teacher scene: 以 mv_prob 概率强制 s=1/t=0.5 并加载 mv0/mv1 exr 作 flow 监督
     (mv 是1帧位移, 仅 s=1 & t=0.5 时与 gt→img0/img1 语义对齐, 其余采样 has_mv=0)

flow_gt 语义对齐 (按项目约定 mv1=gt→上一帧, mv0=gt→下一帧):
  网络 flow[:, :2] = F_t→0 (gt→img0) = mv1@gt
  网络 flow[:,2:4] = F_t→1 (gt→img1) = mv0@gt
  → flow_gt = cat(mv1, mv0), 像素单位 (读入的归一化值 × 原图宽高还原)
  符号由 mv_sign 配置 (先用 verify_mv_convention.py 实证后填入!)

增强与 mv 的联动:
  crop      : mv 空间切片, 数值不变 (像素位移与crop无关)
  rotate180 : mv 四通道全部取负
  时序翻转  : img0↔img1, t→1-t, mv前两通道↔后两通道 (F_t→0 ↔ F_t→1)

__getitem__ 返回:
  frames  : [9, H, W] uint8   (img0, img1, gt 依旧cat, 与旧Trainer一致)
  timestep: [1, 1, 1] float
  flow_gt : [5, H, W] float32 (前4通道位移+第5通道valid mask;
                                has_mv=0 时为全零占位)
  has_mv  : [] float  (1=本样本flow_gt有效, 0=无效, loss侧用它mask)

用法:
  train_ds = MixedTierDataset(root, lists={'easy':..., 'normal':..., 'hard':...,
                              'opensource':..., 'illumination':..., 'noise':..., 'teacher':...},
                              ratios={'easy':0.3,'normal':0.3,'hard':0.1,'opensource':0.1,
                                      'illumination':0.05,'noise':0.05,'teacher':0.1},
                              crop_hw=(256,448), framesteps=(1,2), ...)
  val_ds   = TierDataset(root, list_file=root/'lists/val.txt', split='val')
'''
import os
import sys
import random
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(1)
logger = logging.getLogger(__name__)

# ── exr flow 读取: file_utils.read ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')
try:
    from file_utils import read as fu_read
    _HAS_FILE_UTILS = True
except ImportError:
    _HAS_FILE_UTILS = False
    logger.warning('未找到 file_utils, teacher flow 读取不可用')

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
MV_EXTS = {'.exr'}
DEFAULT_TRAIN_TIERS = (
    'easy', 'normal', 'hard',
    'opensource', 'illumination', 'noise',
    'teacher',
)


def resolve_train_lists(lists_dir, phases, tiers=None) -> Dict[str, str]:
    """Resolve configured category names to existing ``*_train.txt`` files.

    Positive-ratio categories are mandatory. Zero-ratio categories may be
    absent, allowing old datasets/configurations to keep working unchanged.
    """
    configured = list(tiers or DEFAULT_TRAIN_TIERS)
    positive = {
        name
        for phase in phases
        for name, ratio in phase.get('ratios', {}).items()
        if float(ratio) > 0
    }
    for name in positive:
        if name not in configured:
            configured.append(name)

    base = Path(lists_dir)
    resolved = {
        name: str(base / f'{name}_train.txt')
        for name in configured
        if name in positive and (base / f'{name}_train.txt').is_file()
    }
    missing = sorted(positive - resolved.keys())
    if missing:
        expected = ', '.join(str(base / f'{name}_train.txt') for name in missing)
        raise FileNotFoundError(
            f'以下正权重数据分类缺少清单: {expected}. '
            f'请先运行 data_prepare/build_lists.py 或将该分类ratio设为0.')
    if not resolved:
        raise FileNotFoundError(f'未在 {base} 找到任何 *_train.txt 训练清单')
    return resolved


def _list_frames(d: Path, exts) -> List[Path]:
    def num(p):
        m = re.findall(r'(\d+)', p.stem)
        return int(m[-1]) if m else -1
    return sorted((p for p in d.iterdir()
                   if p.suffix.lower() in exts and not p.name.startswith('.')),
                  key=num)


# ─────────────────────────────────────────────────────────────────────────────
# 单清单 Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TierDataset(Dataset):
    """
    Parameters
    ----------
    root        : 数据根目录 (清单中相对路径基于此)
    list_file   : build_lists.py 生成的清单 txt
    split       : 'train' | 'val'  (val: 确定性枚举/抽样 s=1/t=0.5 三元组, 无增强无裁剪)
    crop_hw     : (h, w) 训练随机裁剪, None=不裁剪
    framesteps  : 可采样的 framestep 集合
    t_half_prob : t=0.5 的采样概率 (其余概率均匀采窗口内任意t)
    mv_prob     : teacher scene 采样为 flow监督样本(s=1,t=0.5,带mv) 的概率
    mv_sign     : (sx, sy) mv符号修正, 用 verify_mv_convention.py 的结论填
    mv_symmetry_confidence : 是否将 |mv0+mv1| 作为软置信度; 默认关闭,
                             不再把非匀速运动误判为无效flow
    motion_aware_crop_prob : teacher样本以小运动连通域为中心裁剪的概率
    mv_cache_dirname : scene内离线MV缓存目录；None表示只读取EXR
    mv_cache_required : 为True时缓存缺失/损坏直接报错，禁止静默回退慢速EXR
    mv_cache_preview_stride : 小运动裁剪预览图的降采样步长
    val_samples_per_scene : 验证时每个scene最多均匀抽取的三元组数；None=全部
    """

    def __init__(self, root, list_file, split='train',
                 crop_hw: Optional[Tuple[int, int]] = (256, 448),
                 resize_hw: Optional[Tuple[int, int]] = None,
                 framesteps: Sequence[int] = (1, 2),
                 t_half_prob: float = 0.6,
                 mv_prob: float = 0.5,
                 mv_sign: Tuple[int, int] = (1, 1),
                 occ_alpha: float = 0.05,
                 occ_beta: float = 1.0,
                 mv_symmetry_confidence: bool = False,
                 motion_aware_crop_prob: float = 0.0,
                 motion_crop_threshold: float = 1.0,
                 small_motion_min_pixels: int = 8,
                 small_motion_max_ratio: float = 0.05,
                 motion_crop_jitter: float = 0.2,
                 mv_cache_dirname: Optional[str] = None,
                 mv_cache_required: bool = False,
                 mv_cache_preview_stride: int = 4,
                 val_with_mv: bool = False,
                 val_samples_per_scene: Optional[int] = None,
                 augment_profile: str = 'legacy',
                 augment: bool = True):
        self.root = Path(root)
        self.split = split
        self.crop_hw = crop_hw if split == 'train' else None
        self.resize_hw = (
            tuple(int(value) for value in resize_hw)
            if resize_hw is not None and split == 'train' else None)
        self.framesteps = tuple(int(value) for value in framesteps)
        if not self.framesteps or any(value <= 0 for value in self.framesteps):
            raise ValueError(f'framesteps必须包含正整数, got {framesteps!r}')
        if (self.resize_hw is not None
                and (len(self.resize_hw) != 2
                     or min(self.resize_hw) <= 0)):
            raise ValueError(f'resize_hw必须为两个正整数, got {resize_hw!r}')
        self.t_half_prob = t_half_prob
        self.mv_prob = mv_prob
        self.mv_sign = mv_sign
        self.occ_alpha = occ_alpha
        self.occ_beta = occ_beta
        self.mv_symmetry_confidence = bool(mv_symmetry_confidence)
        self.motion_aware_crop_prob = float(motion_aware_crop_prob)
        self.motion_crop_threshold = float(motion_crop_threshold)
        self.small_motion_min_pixels = int(small_motion_min_pixels)
        self.small_motion_max_ratio = float(small_motion_max_ratio)
        self.motion_crop_jitter = float(motion_crop_jitter)
        self.mv_cache_dirname = (
            str(mv_cache_dirname).strip() if mv_cache_dirname else None)
        self.mv_cache_required = bool(mv_cache_required)
        if self.mv_cache_required and self.mv_cache_dirname is None:
            raise ValueError('mv_cache_required=true时必须配置mv_cache_dirname')
        self.mv_cache_preview_stride = int(mv_cache_preview_stride)
        if self.mv_cache_preview_stride <= 0:
            raise ValueError(
                'mv_cache_preview_stride必须为正整数，'
                f'got {mv_cache_preview_stride!r}')
        self.val_with_mv = bool(val_with_mv)
        self.val_samples_per_scene = None
        if val_samples_per_scene is not None:
            try:
                val_sample_cap = int(val_samples_per_scene)
                is_integer = float(val_samples_per_scene) == val_sample_cap
            except (TypeError, ValueError):
                is_integer = False
                val_sample_cap = 0
            if not is_integer or val_sample_cap <= 0:
                raise ValueError(
                    'val_samples_per_scene必须为正整数或null，'
                    f'got {val_samples_per_scene!r}')
            self.val_samples_per_scene = val_sample_cap
        self.augment_profile = str(augment_profile).lower()
        if self.augment_profile not in ('legacy', 'vimeo'):
            raise ValueError(
                'augment_profile须为legacy或vimeo，'
                f'got {augment_profile!r}')
        self.do_augment = augment and split == 'train'

        self.scenes: List[dict] = []
        with open(list_file) as f:
            for line in f:
                if not line.strip():
                    continue
                rel, n, tier, has_mv, fps = line.rstrip('\n').split('\t')
                self.scenes.append({'rel': rel, 'n': int(n), 'tier': tier,
                                    'has_mv': int(has_mv), 'fps': int(fps)})
        if not self.scenes:
            raise ValueError(f'清单为空: {list_file}')

        # 帧文件名索引 (build_lists 生成的 .frames.json): 训练时零目录列举
        self._frame_index: Dict[str, List[str]] = {}
        idx_path = Path(list_file).with_suffix('.frames.json')
        if idx_path.exists():
            import json
            with open(idx_path) as f:
                self._frame_index = json.load(f)
            logger.info(f'帧索引已加载: {idx_path.name} ({len(self._frame_index)} scenes)')
        else:
            logger.warning(f'未找到帧索引 {idx_path.name}, 回退运行时目录列举 '
                           f'(建议重跑 build_lists.py 生成)')

        max_s = max(self.framesteps)
        # train: 每scene权重 = s=1可采三元组数；val可确定性均匀限量，
        # 避免少数长scene产生数百次完整分辨率前向。
        self._weights = [max(s['n'] - 2 * max_s, 1) for s in self.scenes]
        self._cum = np.cumsum(self._weights)
        if split == 'val':
            self._val_index = []
            for si, scene in enumerate(self.scenes):
                sample_count = scene['n'] - 2
                cap = self.val_samples_per_scene
                if cap is None or sample_count <= cap:
                    indices = range(sample_count)
                elif cap == 1:
                    indices = (sample_count // 2,)
                else:
                    # 固定包含首尾，并在中间等距取点；不依赖随机种子。
                    indices = tuple(round(
                        index * (sample_count - 1) / (cap - 1))
                        for index in range(cap))
                self._val_index.extend((si, index) for index in indices)
        self._frame_cache: Dict[str, List[Path]] = {}
        self._mv_path_cache: Dict[Tuple[str, str], Dict[int, Path]] = {}
        self._cache_warnings = set()
        if (self.mv_cache_required
                and (self.split == 'train' or self.val_with_mv)
                and any(scene['has_mv'] for scene in self.scenes)):
            self._validate_required_mv_cache()

    # ── 帧文件定位 (优先帧索引, 无索引才回退listdir并惰性缓存) ─────────────
    def _frames_of(self, scene: dict) -> List[Path]:
        rel = scene['rel']
        if rel not in self._frame_cache:
            d = self.root / rel
            if scene['has_mv']:
                d = d / 'image'
            names = self._frame_index.get(rel)
            if names is not None:
                self._frame_cache[rel] = [d / n for n in names]   # 零listdir
            else:
                self._frame_cache[rel] = _list_frames(d, IMG_EXTS)
        return self._frame_cache[rel]

    def _mv_path(self, scene: dict, frame_path: Path,
                 which: str) -> Optional[Path]:
        """按帧号匹配MV，兼容image与flow使用不同文件名前缀的Spring数据。"""
        directory = self.root / scene['rel'] / which
        exact = (directory / frame_path.name).with_suffix('.exr')
        if exact.is_file():
            return exact
        numbers = re.findall(r'(\d+)', frame_path.stem)
        if not numbers:
            return None
        key = (scene['rel'], which)
        if key not in self._mv_path_cache:
            mapping = {}
            for path in _list_frames(directory, MV_EXTS):
                matches = re.findall(r'(\d+)', path.stem)
                if matches:
                    mapping[int(matches[-1])] = path
            self._mv_path_cache[key] = mapping
        return self._mv_path_cache[key].get(int(numbers[-1]))

    def _mv_cache_paths(self, scene: dict,
                        frame_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
        if self.mv_cache_dirname is None:
            return None, None
        directory = self.root / scene['rel'] / self.mv_cache_dirname
        return (directory / f'{frame_path.stem}.npy',
                directory / f'{frame_path.stem}.motion.npy')

    def _validate_required_mv_cache(self) -> None:
        """训练开始前一次性检查，避免跑了数小时后才遇到缺失缓存。"""
        missing = []
        checked = 0
        need_preview = self.split == 'train' and self.motion_aware_crop_prob > 0
        for scene in self.scenes:
            if not scene['has_mv']:
                continue
            # teacher监督的GT固定为三元组中间帧，首尾永远不会访问且通常无MV。
            for frame_path in self._frames_of(scene)[1:-1]:
                cache_path, preview_path = self._mv_cache_paths(scene, frame_path)
                checked += 1
                if cache_path is None or not cache_path.is_file():
                    missing.append(str(cache_path))
                if (need_preview and preview_path is not None
                        and not preview_path.is_file()):
                    missing.append(str(preview_path))
                if len(missing) >= 10:
                    break
            if len(missing) >= 10:
                break
        if missing:
            examples = '\n  '.join(missing)
            raise FileNotFoundError(
                f'MV快速缓存不完整（前{len(missing)}个缺失项）:\n  {examples}\n'
                '请先运行 data_prepare/build_mv_cache.py --verify')
        logger.info(
            'MV快速缓存预检通过: %d frames, cache=%s',
            checked, self.mv_cache_dirname)

    def _warn_cache_once(self, key: str, message: str) -> None:
        if key not in self._cache_warnings:
            self._cache_warnings.add(key)
            logger.warning(message)

    def _normalized_mv_to_flow(self, normalized: np.ndarray,
                               source_h: int, source_w: int) -> np.ndarray:
        """将缓存/EXR中的归一化 [H,W,4] MV 转为像素 flow+valid。"""
        if normalized.ndim != 3 or normalized.shape[-1] != 4:
            raise ValueError(f'MV缓存应为[H,W,4], got {normalized.shape}')
        flow = normalized.astype(np.float32, copy=True)
        finite = np.isfinite(flow).all(axis=-1)
        np.nan_to_num(flow, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        sx, sy = self.mv_sign
        flow[..., (0, 2)] *= sx * source_w
        flow[..., (1, 3)] *= sy * source_h

        region_h, region_w = flow.shape[:2]
        xs = np.arange(region_w, dtype=np.float32)[None, :]
        ys = np.arange(region_h, dtype=np.float32)[:, None]
        inside1 = (
            (xs + flow[..., 0] >= 0) & (xs + flow[..., 0] <= region_w - 1)
            & (ys + flow[..., 1] >= 0) & (ys + flow[..., 1] <= region_h - 1))
        inside0 = (
            (xs + flow[..., 2] >= 0) & (xs + flow[..., 2] <= region_w - 1)
            & (ys + flow[..., 3] >= 0) & (ys + flow[..., 3] <= region_h - 1))
        valid = (finite & inside1 & inside0).astype(np.float32)

        if self.mv_symmetry_confidence:
            sym = np.linalg.norm(flow[..., 0:2] + flow[..., 2:4], axis=-1)
            mag = (np.linalg.norm(flow[..., 0:2], axis=-1)
                   + np.linalg.norm(flow[..., 2:4], axis=-1))
            denom = np.maximum(self.occ_alpha * mag + self.occ_beta, 1e-6)
            valid *= 1.0 / (1.0 + sym / denom)
        return np.concatenate((flow, valid[..., None]), axis=-1)

    def _read_mv_cache(self, scene: dict, gt_path: Path,
                       image_h: int, image_w: int,
                       crop: Optional[Tuple[int, int, int, int]] = None
                       ) -> Optional[np.ndarray]:
        """mmap未压缩float16缓存；有crop时只触发对应页面的磁盘读取。"""
        cache_path, _ = self._mv_cache_paths(scene, gt_path)
        if cache_path is None:
            return None
        if not cache_path.is_file():
            message = (
                f'MV缓存缺失: {cache_path}. 请先运行 '
                'data_prepare/build_mv_cache.py。')
            if self.mv_cache_required:
                raise FileNotFoundError(message)
            self._warn_cache_once('missing', message + ' 回退EXR读取。')
            return None
        try:
            cached = np.load(cache_path, mmap_mode='r', allow_pickle=False)
            if cached.ndim != 3 or cached.shape[-1] != 4:
                raise ValueError(
                    f'期望[H,W,4]，实际{cached.shape}')
            if cached.shape[:2] != (image_h, image_w):
                raise ValueError(
                    f'缓存尺寸{cached.shape[:2]}与图像{(image_h, image_w)}不一致')
            if crop is None:
                top, left, region_h, region_w = 0, 0, image_h, image_w
            else:
                top, left, region_h, region_w = crop
                if (top < 0 or left < 0 or top + region_h > image_h
                        or left + region_w > image_w):
                    raise ValueError(f'非法MV cache crop: {crop}')
            # 保持mmap view；唯一一次float32拷贝在_normalized_mv_to_flow完成。
            normalized = cached[
                top:top + region_h, left:left + region_w, :]
            return self._normalized_mv_to_flow(
                normalized, source_h=image_h, source_w=image_w)
        except Exception as exc:
            message = f'MV缓存读取失败: {cache_path} ({exc})'
            if self.mv_cache_required:
                raise RuntimeError(message) from exc
            self._warn_cache_once('broken', message + '，回退EXR读取。')
            return None

    # ── 读取 ────────────────────────────────────────────────────────────────
    @staticmethod
    def _read_img(p: Path) -> np.ndarray:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f'无法读取: {p}')
        return img[..., ::-1]                                   # BGR→RGB

    def _read_mv_pair(self, scene: dict, gt_path: Path,
                      h: int, w: int) -> Optional[np.ndarray]:
        """读取 gt 帧的 mv1/mv0，反归一化到像素并生成有效性/置信度mask。

        返回 [H,W,5] = (mv1(2), mv0(2), valid(1)); 损坏/缺失返回 None。
        valid 基础条件为数值有限且两个目标坐标都在图像内。可选的
        mv_symmetry_confidence 只把双向非对称性作为软权重，不再把加速、
        转向等合法的非匀速运动硬判为无效。"""
        if not _HAS_FILE_UTILS:
            return None
        mv1_path = self._mv_path(scene, gt_path, 'mv1')
        mv0_path = self._mv_path(scene, gt_path, 'mv0')
        if mv1_path is None or mv0_path is None:
            logger.warning(
                f'mv文件缺失: {scene["rel"]}/{gt_path.name}')
            return None
        try:
            mv1 = fu_read(str(mv1_path), type='flo')
            mv0 = fu_read(str(mv0_path), type='flo')
        except Exception as e:                      # 截断exr等解码失败
            logger.warning(f'mv读取失败(损坏?): {scene["rel"]}/{gt_path.name} ({e})')
            return None
        if mv1 is None or mv0 is None:
            return None
        if (mv1.ndim != 3 or mv0.ndim != 3
                or mv1.shape[-1] < 2 or mv0.shape[-1] < 2
                or mv1.shape[:2] != (h, w) or mv0.shape[:2] != (h, w)):
            logger.warning(
                f'mv尺寸/通道不匹配: {scene["rel"]}/{gt_path.name} '
                f'mv1={getattr(mv1, "shape", None)} '
                f'mv0={getattr(mv0, "shape", None)} image={(h, w)}')
            return None
        # 网络约定: flow[:, :2]=F_t→0=mv1(gt→上一帧),  flow[:,2:4]=F_t→1=mv0(gt→下一帧)
        normalized = np.concatenate((mv1[..., :2], mv0[..., :2]), axis=-1)
        return self._normalized_mv_to_flow(normalized, h, w)   # [H,W,5]

    # ── 采样 ────────────────────────────────────────────────────────────────
    def _pick_scene(self, index: int) -> dict:
        pos = index % self._cum[-1]
        si = int(np.searchsorted(self._cum, pos, side='right'))
        return self.scenes[si]

    def _sample_window(self, scene: dict):
        """返回 (i0, ig, i1, t, want_mv)。"""
        n = scene['n']
        want_mv = scene['has_mv'] and random.random() < self.mv_prob
        if want_mv:
            s = 1                                               # mv仅在s=1,t=0.5对齐
            i0 = random.randint(0, n - 3)
            return i0, i0 + 1, i0 + 2, 0.5, True
        valid = [s for s in self.framesteps if n >= 2 * s + 1]
        s = random.choice(valid)
        i0 = random.randint(0, n - 2 * s - 1)
        if random.random() < self.t_half_prob or 2 * s < 2:
            ig = i0 + s
        else:
            ig = i0 + random.randint(1, 2 * s - 1)              # 任意t, 真实GT
        t = (ig - i0) / (2 * s)
        return i0, ig, i0 + 2 * s, t, False

    # ── 增强 (与 mv 联动) ────────────────────────────────────────────────────
    def _motion_crop_origin(self, mv: np.ndarray, crop_h: int, crop_w: int,
                            image_h: int, image_w: int) -> Optional[Tuple[int, int]]:
        """返回以小运动连通域为中心的 (top, left), 找不到时返回None。

        使用相对全局中位flow的残差而非绝对位移，避免摄像机平移时
        整张图被视为一个巨大运动物体。
        """
        if mv is None or mv.shape[-1] < 5:
            return None
        valid = mv[..., 4] > 0.5
        if valid.sum() < self.small_motion_min_pixels:
            return None

        flow = mv[..., :4]
        # 全局运动只需稀疏估计, 避免对大分辨率valid像素做大量拷贝。
        flow_sparse = flow[::8, ::8]
        valid_sparse = valid[::8, ::8]
        if valid_sparse.any():
            global_flow = np.median(flow_sparse[valid_sparse], axis=0)
        else:
            global_flow = np.median(flow[valid], axis=0)
        residual0 = np.linalg.norm(flow[..., 0:2] - global_flow[0:2], axis=-1)
        residual1 = np.linalg.norm(flow[..., 2:4] - global_flow[2:4], axis=-1)
        motion = 0.5 * (residual0 + residual1)
        motion_mask = (valid & (motion >= self.motion_crop_threshold)).astype(np.uint8)

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            motion_mask, connectivity=8)
        max_area = max(
            self.small_motion_min_pixels,
            int(crop_h * crop_w * self.small_motion_max_ratio))
        candidates = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if self.small_motion_min_pixels <= area <= max_area:
                candidates.append((label, area))
        if not candidates:
            return None

        # 小连通域抽到的概率更高, 但不完全排除稍大的小物体。
        label, _ = random.choices(
            candidates, weights=[1.0 / np.sqrt(a) for _, a in candidates], k=1)[0]
        center_x, center_y = centroids[label]                       # OpenCV: (x, y)
        jitter_y = int(random.uniform(-1, 1) * crop_h * self.motion_crop_jitter)
        jitter_x = int(random.uniform(-1, 1) * crop_w * self.motion_crop_jitter)
        top = int(round(center_y - crop_h / 2 + jitter_y))
        left = int(round(center_x - crop_w / 2 + jitter_x))
        top = min(max(top, 0), image_h - crop_h)
        left = min(max(left, 0), image_w - crop_w)
        return top, left

    def _motion_crop_origin_preview(
            self, scene: dict, gt_path: Path, crop_h: int, crop_w: int,
            image_h: int, image_w: int) -> Optional[Tuple[int, int]]:
        """从低分辨率motion sidecar选择小物体crop，避免读取整帧MV。"""
        _, preview_path = self._mv_cache_paths(scene, gt_path)
        if preview_path is None:
            return None
        if not preview_path.is_file():
            message = (
                f'MV motion预览缺失: {preview_path}. 请重新运行 '
                'data_prepare/build_mv_cache.py。')
            if self.mv_cache_required:
                raise FileNotFoundError(message)
            self._warn_cache_once('preview_missing', message + ' 回退随机裁剪。')
            return None
        try:
            motion = np.load(preview_path, allow_pickle=False)
            if motion.ndim != 2:
                raise ValueError(f'期望二维数组，实际{motion.shape}')
            stride = self.mv_cache_preview_stride
            expected = ((image_h + stride - 1) // stride,
                        (image_w + stride - 1) // stride)
            if motion.shape != expected:
                raise ValueError(
                    f'预览尺寸{motion.shape}与期望{expected}不一致；'
                    '生成缓存和训练配置的preview_stride必须相同')
        except Exception as exc:
            message = f'MV motion预览读取失败: {preview_path} ({exc})'
            if self.mv_cache_required:
                raise RuntimeError(message) from exc
            self._warn_cache_once('preview_broken', message + '，回退随机裁剪。')
            return None

        motion_mask = (
            np.isfinite(motion)
            & (motion >= self.motion_crop_threshold)).astype(np.uint8)
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            motion_mask, connectivity=8)
        max_area = max(
            self.small_motion_min_pixels,
            int(crop_h * crop_w * self.small_motion_max_ratio))
        cell_area = stride * stride
        candidates = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA]) * cell_area
            if self.small_motion_min_pixels <= area <= max_area:
                candidates.append((label, area))
        if not candidates:
            return None

        label, _ = random.choices(
            candidates, weights=[1.0 / np.sqrt(a) for _, a in candidates], k=1)[0]
        center_x, center_y = centroids[label]
        center_x = (center_x + 0.5) * stride
        center_y = (center_y + 0.5) * stride
        jitter_y = int(random.uniform(-1, 1) * crop_h * self.motion_crop_jitter)
        jitter_x = int(random.uniform(-1, 1) * crop_w * self.motion_crop_jitter)
        top = int(round(center_y - crop_h / 2 + jitter_y))
        left = int(round(center_x - crop_w / 2 + jitter_x))
        return (min(max(top, 0), image_h - crop_h),
                min(max(left, 0), image_w - crop_w))

    def _cached_crop_origin(self, scene: dict, gt_path: Path,
                            crop_h: int, crop_w: int,
                            image_h: int, image_w: int) -> Tuple[int, int]:
        origin = None
        if (self.motion_aware_crop_prob > 0
                and random.random() < self.motion_aware_crop_prob):
            origin = self._motion_crop_origin_preview(
                scene, gt_path, crop_h, crop_w, image_h, image_w)
        if origin is not None:
            return origin
        return (np.random.randint(0, image_h - crop_h + 1),
                np.random.randint(0, image_w - crop_w + 1))

    @staticmethod
    def _pad_to_crop(arrs: List[np.ndarray], crop_h: int,
                     crop_w: int) -> List[np.ndarray]:
        """尺寸不足时先补到 crop 大小，保证 DataLoader 可以稳定堆叠。

        前三个数组是 img0/gt/img1，使用反射补边；第四个是 flow GT，
        补0后 valid 通道也为0，因此人工边界不参与 flow loss。补边位置
        随机分配到两侧，避免原图永远固定在 crop 中心。
        """
        image_h, image_w = arrs[0].shape[:2]
        for arr in arrs[1:]:
            if arr.shape[:2] != (image_h, image_w):
                raise ValueError(
                    f'同一样本的图像/flow尺寸不一致: '
                    f'{(image_h, image_w)} vs {arr.shape[:2]}')

        pad_h = max(crop_h - image_h, 0)
        pad_w = max(crop_w - image_w, 0)
        if pad_h == 0 and pad_w == 0:
            return arrs

        pad_top = np.random.randint(0, pad_h + 1) if pad_h else 0
        pad_left = np.random.randint(0, pad_w + 1) if pad_w else 0
        pad_bottom = pad_h - pad_top
        pad_right = pad_w - pad_left
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

        padded = []
        for index, arr in enumerate(arrs):
            if index < 3:
                # 极端的1像素宽/高无法reflect，此时回退复制边界。
                mode = 'reflect' if image_h > 1 and image_w > 1 else 'edge'
                padded.append(np.pad(arr, pad_width, mode=mode))
            else:
                padded.append(np.pad(arr, pad_width, mode='constant',
                                     constant_values=0))
        return padded

    def _crop(self, arrs: List[np.ndarray]) -> List[np.ndarray]:
        if self.crop_hw is None:
            return arrs
        h, w = self.crop_hw
        arrs = self._pad_to_crop(arrs, h, w)
        ih, iw = arrs[0].shape[:2]
        if ih == h and iw == w:
            return arrs

        origin = None
        mv = arrs[3] if len(arrs) > 3 else None
        if (mv is not None and self.motion_aware_crop_prob > 0
                and random.random() < self.motion_aware_crop_prob):
            origin = self._motion_crop_origin(mv, h, w, ih, iw)
        if origin is None:
            top = np.random.randint(0, ih - h + 1)
            left = np.random.randint(0, iw - w + 1)
        else:
            top, left = origin
        cropped = [a[top:top + h, left:left + w] for a in arrs]
        if len(cropped) > 3:
            # 原图内有效并不代表 crop 内仍有对应点。若终点被裁掉，网络输入
            # 中已经没有可用于该像素监督的源内容，必须再次清除 valid。
            flow = cropped[3]
            xs = np.arange(w, dtype=np.float32)[None, :]
            ys = np.arange(h, dtype=np.float32)[:, None]
            inside0 = (
                (xs + flow[..., 0] >= 0) & (xs + flow[..., 0] <= w - 1)
                & (ys + flow[..., 1] >= 0) & (ys + flow[..., 1] <= h - 1))
            inside1 = (
                (xs + flow[..., 2] >= 0) & (xs + flow[..., 2] <= w - 1)
                & (ys + flow[..., 3] >= 0) & (ys + flow[..., 3] <= h - 1))
            flow[..., 4] *= (inside0 & inside1).astype(flow.dtype)
        return cropped

    def _resize_before_crop(self, arrs: List[np.ndarray]) -> List[np.ndarray]:
        """按curriculum尺寸缩放整帧，再执行固定大小随机裁剪。

        图像使用面积/线性插值；若包含teacher flow，则同步缩放位移数值并
        使用最近邻处理valid通道。官方X-TRAIN不含flow，但这里保持通用。
        """
        if self.resize_hw is None:
            return arrs
        target_h, target_w = self.resize_hw
        source_h, source_w = arrs[0].shape[:2]
        if (source_h, source_w) == (target_h, target_w):
            return arrs
        sx, sy = target_w / source_w, target_h / source_h
        interpolation = (
            cv2.INTER_AREA if target_h < source_h or target_w < source_w
            else cv2.INTER_LINEAR)
        resized = []
        for index, array in enumerate(arrs):
            if index < 3:
                resized.append(cv2.resize(
                    array, (target_w, target_h), interpolation=interpolation))
                continue
            flow = cv2.resize(
                array[..., :4], (target_w, target_h),
                interpolation=cv2.INTER_LINEAR)
            flow[..., (0, 2)] *= sx
            flow[..., (1, 3)] *= sy
            valid = cv2.resize(
                array[..., 4], (target_w, target_h),
                interpolation=cv2.INTER_NEAREST)[..., None]
            resized.append(np.concatenate((flow, valid), axis=-1))
        return resized

    def _augment(self, img0, gt, img1, t, mv):
        if self.augment_profile == 'vimeo':
            if mv is not None:
                raise ValueError(
                    'vimeo增强模式仅用于无MV数据；'
                    'teacher数据请使用augment_profile=legacy')
            if random.random() < 0.5:                           # horizontal flip
                img0 = np.flip(img0, axis=1)
                gt = np.flip(gt, axis=1)
                img1 = np.flip(img1, axis=1)
            if random.random() < 0.5:                           # vertical flip
                img0 = np.flip(img0, axis=0)
                gt = np.flip(gt, axis=0)
                img1 = np.flip(img1, axis=0)
            rotate_k = random.randrange(4)                      # 0/90/180/270°
            if rotate_k:
                if img0.shape[0] != img0.shape[1]:
                    raise ValueError(
                        'vimeo随机90°旋转要求方形crop，'
                        f'当前为{img0.shape[:2]}')
                img0 = np.rot90(img0, rotate_k)
                gt = np.rot90(gt, rotate_k)
                img1 = np.rot90(img1, rotate_k)
        elif random.random() < 0.5:                             # legacy rotate180
            img0 = cv2.rotate(img0, cv2.ROTATE_180)
            gt = cv2.rotate(gt, cv2.ROTATE_180)
            img1 = cv2.rotate(img1, cv2.ROTATE_180)
            if mv is not None:
                mv = cv2.rotate(mv, cv2.ROTATE_180)
                mv[..., :4] = mv[..., :4] * -1.0                # 位移反转; valid通道(第5)不变
        if random.random() < 0.5:                               # 时序翻转
            img0, img1 = img1, img0
            t = 1.0 - t
            if mv is not None:
                mv = mv[..., [2, 3, 0, 1, 4]]                   # F_t→0 ↔ F_t→1; valid不动
        return img0, gt, img1, t, mv

    # ── Dataset 接口 ────────────────────────────────────────────────────────
    def __len__(self) -> int:
        if self.split == 'val':
            return len(self._val_index)
        return int(self._cum[-1])

    def __getitem__(self, index: int):
        if self.split == 'val':
            si, i0 = self._val_index[index]
            scene = self.scenes[si]
            ig, i1, t = i0 + 1, i0 + 2, 0.5
            want_mv = self.val_with_mv and bool(scene['has_mv'])
        else:
            scene = self._pick_scene(index)
            i0, ig, i1, t, want_mv = self._sample_window(scene)

        frames = self._frames_of(scene)
        img0 = self._read_img(frames[i0])
        gt = self._read_img(frames[ig])
        img1 = self._read_img(frames[i1])

        mv = None
        used_direct_cache_crop = False
        if want_mv:
            h, w = gt.shape[:2]
            cache_path, _ = self._mv_cache_paths(scene, frames[ig])
            can_direct_crop = (
                cache_path is not None and cache_path.is_file()
                and self.resize_hw is None and self.crop_hw is not None
                and h >= self.crop_hw[0] and w >= self.crop_hw[1])
            if can_direct_crop:
                crop_h, crop_w = self.crop_hw
                top, left = self._cached_crop_origin(
                    scene, frames[ig], crop_h, crop_w, h, w)
                mv = self._read_mv_cache(
                    scene, frames[ig], h, w,
                    crop=(top, left, crop_h, crop_w))
                if mv is not None:
                    img0 = img0[top:top + crop_h, left:left + crop_w]
                    gt = gt[top:top + crop_h, left:left + crop_w]
                    img1 = img1[top:top + crop_h, left:left + crop_w]
                    used_direct_cache_crop = True
            elif self.mv_cache_dirname is not None:
                mv = self._read_mv_cache(scene, frames[ig], h, w)
            if mv is None:
                mv = self._read_mv_pair(scene, frames[ig], h, w)  # EXR回退

        if not used_direct_cache_crop:
            arrs = [img0, gt, img1] + ([mv] if mv is not None else [])
            arrs = self._resize_before_crop(arrs)
            arrs = self._crop(arrs)
            img0, gt, img1 = arrs[0], arrs[1], arrs[2]
            mv = arrs[3] if mv is not None else None

        if self.do_augment:
            img0, gt, img1, t, mv = self._augment(img0, gt, img1, t, mv)

        h, w = gt.shape[:2]
        frames_t = torch.cat([
            torch.from_numpy(img0.copy()).permute(2, 0, 1),
            torch.from_numpy(img1.copy()).permute(2, 0, 1),
            torch.from_numpy(gt.copy()).permute(2, 0, 1),
        ], dim=0)                                               # [9,H,W] uint8
        timestep = torch.tensor(float(t)).reshape(1, 1, 1)
        if mv is not None:
            flow_gt = torch.from_numpy(np.ascontiguousarray(mv)).permute(2, 0, 1).float()
            has_mv = torch.tensor(1.0)
        else:
            flow_gt = torch.zeros(5, h, w)                      # 第5通道valid=0
            has_mv = torch.tensor(0.0)

        return frames_t, timestep, flow_gt, has_mv


# ─────────────────────────────────────────────────────────────────────────────
# 多tier配比混合
# ─────────────────────────────────────────────────────────────────────────────

class MixedTierDataset(Dataset):
    """
    按配比在多个 TierDataset 间采样。所有tier共用同一 crop 尺寸,
    因此 batch 内可自由混合, 默认 collate 即可。
    配比可训练中途通过 set_ratios 切换 (phase 1 → phase 2)。
    多尺度训练: set_crop_size 建议按 epoch 切换 (worker副本延迟一个epoch生效)。
    """

    def __init__(self, root, lists: Dict[str, str], ratios: Dict[str, float],
                 source_options: Optional[Dict[str, dict]] = None,
                 **tier_kwargs):
        source_options = source_options or {}
        self.datasets: Dict[str, TierDataset] = {}
        for name, list_file in lists.items():
            options = dict(tier_kwargs)
            options.update(source_options.get(name, {}))
            self.datasets[name] = TierDataset(
                root, list_file, split='train', **options)
        self._batch_pattern = None
        self.set_ratios(ratios)
        self._epoch_size = sum(len(d) for d in self.datasets.values())

    def set_ratios(self, ratios: Dict[str, float],
                   batch_counts: Optional[Dict[str, int]] = None) -> None:
        negative = [n for n, value in ratios.items() if float(value) < 0]
        if negative:
            raise ValueError(f'tier权重不能为负数: {negative}')
        missing = [n for n, value in ratios.items()
                   if float(value) > 0 and n not in self.datasets]
        if missing:
            raise ValueError(f'正权重tier未加载清单: {missing}')
        names = [n for n in ratios if n in self.datasets and ratios[n] > 0]
        total = sum(ratios[n] for n in names)
        if total <= 0:
            raise ValueError('tier配比中至少需要一个正权重分类')
        self._names = names
        self._probs = [ratios[n] / total for n in names]
        self._batch_pattern = None
        if batch_counts is not None:
            invalid = {
                name: value for name, value in batch_counts.items()
                if int(value) != value or int(value) < 0
            }
            missing_counts = [
                name for name, value in batch_counts.items()
                if int(value) > 0 and name not in self.datasets]
            if invalid:
                raise ValueError(f'batch_counts必须为非负整数: {invalid}')
            if missing_counts:
                raise ValueError(f'batch_counts引用了未加载tier: {missing_counts}')
            pattern = [
                name for name, value in batch_counts.items()
                for _ in range(int(value))
            ]
            if not pattern:
                raise ValueError('batch_counts至少需要一个正数')
            self._batch_pattern = pattern
        logger.info(f'tier配比: {dict(zip(self._names, [round(p,3) for p in self._probs]))}')
        if self._batch_pattern is not None:
            logger.info(f'固定batch组成: {batch_counts}')

    def set_crop_size(self, crop_hw: Tuple[int, int]) -> None:
        for d in self.datasets.values():
            d.crop_hw = crop_hw

    def configure_source(self, name: str, framesteps=None,
                         resize_hw=None) -> None:
        if name not in self.datasets:
            raise ValueError(f'curriculum引用了未加载tier: {name}')
        dataset = self.datasets[name]
        if framesteps is not None:
            values = tuple(int(value) for value in framesteps)
            if not values or any(value <= 0 for value in values):
                raise ValueError(f'framesteps必须包含正整数: {framesteps!r}')
            minimum_frames = 2 * max(values) + 1
            too_short = sum(
                scene['n'] < minimum_frames for scene in dataset.scenes)
            if too_short:
                raise ValueError(
                    f'{name}有{too_short}个scene不足{minimum_frames}帧，'
                    '请重新生成对应训练清单')
            dataset.framesteps = values
            # curriculum改变窗口跨度后同步更新scene采样权重；否则65帧
            # X-TRAIN在s=32时仍会沿用s=1的窗口数量。
            max_s = max(values)
            dataset._weights = [
                max(scene['n'] - 2 * max_s, 1)
                for scene in dataset.scenes]
            dataset._cum = np.cumsum(dataset._weights)
            self._epoch_size = sum(
                len(source) for source in self.datasets.values())
        if resize_hw is not None:
            values = tuple(int(value) for value in resize_hw)
            if len(values) != 2 or min(values) <= 0:
                raise ValueError(f'resize_hw必须为两个正整数: {resize_hw!r}')
            dataset.resize_hw = values

    @property
    def batch_pattern_size(self) -> Optional[int]:
        return len(self._batch_pattern) if self._batch_pattern is not None else None

    def __len__(self) -> int:
        return self._epoch_size

    def __getitem__(self, index: int):
        name = (
            self._batch_pattern[index % len(self._batch_pattern)]
            if self._batch_pattern is not None
            else random.choices(self._names, weights=self._probs, k=1)[0])
        ds = self.datasets[name]
        return ds[random.randrange(len(ds))]


# ─────────────────────────────────────────────────────────────────────────────
# 快速自检
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--lists_dir', default=None, help='默认 <root>/lists')
    args = p.parse_args()

    lists_dir = Path(args.lists_dir) if args.lists_dir else Path(args.root) / 'lists'
    demo_ratios = {
        'easy': .3, 'normal': .3, 'hard': .1, 'opensource': .1,
        'illumination': .05, 'noise': .05, 'teacher': .1,
    }
    lists = resolve_train_lists(
        lists_dir, phases=[{'ratios': demo_ratios}])

    ds = MixedTierDataset(args.root, lists,
                          ratios=demo_ratios,
                          crop_hw=(256, 448), framesteps=(1, 2))
    print(f'train混合集: {len(ds)} (名义长度)')
    for i in range(4):
        frames, t, flow_gt, has_mv = ds[i]
        print(f'  sample{i}: frames{tuple(frames.shape)} t={t.item():.3f} '
              f'flow{tuple(flow_gt.shape)} has_mv={has_mv.item()}')

    val_f = lists_dir / 'val.txt'
    if val_f.exists():
        vd = TierDataset(args.root, val_f, split='val')
        print(f'val: {len(vd)} 三元组')
        frames, t, flow_gt, has_mv = vd[0]
        print(f'  val sample: frames{tuple(frames.shape)} t={t.item():.2f}')
