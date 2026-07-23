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
    import re
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
    split       : 'train' | 'val'  (val: 确定性枚举 s=1/t=0.5 三元组, 无增强无裁剪)
    crop_hw     : (h, w) 训练随机裁剪, None=不裁剪
    framesteps  : 可采样的 framestep 集合
    t_half_prob : t=0.5 的采样概率 (其余概率均匀采窗口内任意t)
    mv_prob     : teacher scene 采样为 flow监督样本(s=1,t=0.5,带mv) 的概率
    mv_sign     : (sx, sy) mv符号修正, 用 verify_mv_convention.py 的结论填
    motion_aware_crop_prob : teacher样本以小运动连通域为中心裁剪的概率
    """

    def __init__(self, root, list_file, split='train',
                 crop_hw: Optional[Tuple[int, int]] = (256, 448),
                 framesteps: Sequence[int] = (1, 2),
                 t_half_prob: float = 0.6,
                 mv_prob: float = 0.5,
                 mv_sign: Tuple[int, int] = (1, 1),
                 occ_alpha: float = 0.05,
                 occ_beta: float = 1.0,
                 motion_aware_crop_prob: float = 0.0,
                 motion_crop_threshold: float = 1.0,
                 small_motion_min_pixels: int = 8,
                 small_motion_max_ratio: float = 0.05,
                 motion_crop_jitter: float = 0.2,
                 augment: bool = True):
        self.root = Path(root)
        self.split = split
        self.crop_hw = crop_hw if split == 'train' else None
        self.framesteps = tuple(framesteps)
        self.t_half_prob = t_half_prob
        self.mv_prob = mv_prob
        self.mv_sign = mv_sign
        self.occ_alpha = occ_alpha          # 遮挡判定: |mv0+mv1| < alpha*(|mv0|+|mv1|) + beta
        self.occ_beta = occ_beta
        self.motion_aware_crop_prob = float(motion_aware_crop_prob)
        self.motion_crop_threshold = float(motion_crop_threshold)
        self.small_motion_min_pixels = int(small_motion_min_pixels)
        self.small_motion_max_ratio = float(small_motion_max_ratio)
        self.motion_crop_jitter = float(motion_crop_jitter)
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
        # train: 每scene权重 = s=1可采三元组数; val: 确定性展开全部 s=1 三元组
        self._weights = [max(s['n'] - 2 * max_s, 1) for s in self.scenes]
        self._cum = np.cumsum(self._weights)
        if split == 'val':
            self._val_index = [(si, i) for si, s in enumerate(self.scenes)
                               for i in range(s['n'] - 2)]
        self._frame_cache: Dict[str, List[Path]] = {}

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

    def _mv_path(self, scene: dict, frame_path: Path, which: str) -> Path:
        return (self.root / scene['rel'] / which / frame_path.name).with_suffix('.exr')

    # ── 读取 ────────────────────────────────────────────────────────────────
    @staticmethod
    def _read_img(p: Path) -> np.ndarray:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f'无法读取: {p}')
        return img[..., ::-1]                                   # BGR→RGB

    def _read_mv_pair(self, scene: dict, gt_path: Path,
                      h: int, w: int) -> Optional[np.ndarray]:
        """读取 gt 帧的 mv1/mv0, 反归一化到像素, 并由双向对称性生成遮挡有效性mask。
        返回 [H,W,5] = (mv1(2), mv0(2), valid(1)); 损坏/缺失返回 None。
        valid=0 的像素为遮挡/uncover区: 该处flow无合法对应点, 监督在物理上病态,
        由 Trainer 的逐像素加权EPE跳过。"""
        if not _HAS_FILE_UTILS:
            return None
        try:
            mv1 = fu_read(str(self._mv_path(scene, gt_path, 'mv1')), type='flo')
            mv0 = fu_read(str(self._mv_path(scene, gt_path, 'mv0')), type='flo')
        except Exception as e:                      # 截断exr等解码失败
            logger.warning(f'mv读取失败(损坏?): {scene["rel"]}/{gt_path.name} ({e})')
            return None
        if mv1 is None or mv0 is None:
            return None
        mv1 = mv1[..., :2].astype(np.float32)
        mv0 = mv0[..., :2].astype(np.float32)
        sx, sy = self.mv_sign
        for mv in (mv1, mv0):
            mv[..., 0] *= sx * w                                # 归一化 → 像素
            mv[..., 1] *= sy * h

        # 遮挡mask: t=0.5线性运动下 mv0 ≈ -mv1; 不对称像素即遮挡/uncover区
        sym = np.linalg.norm(mv1 + mv0, axis=-1)                # [H,W]
        mag = np.linalg.norm(mv1, axis=-1) + np.linalg.norm(mv0, axis=-1)
        valid = (sym < self.occ_alpha * mag + self.occ_beta).astype(np.float32)

        # 网络约定: flow[:, :2]=F_t→0=mv1(gt→上一帧),  flow[:,2:4]=F_t→1=mv0(gt→下一帧)
        return np.concatenate([mv1, mv0, valid[..., None]], axis=-1)   # [H,W,5]

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
        return [a[top:top + h, left:left + w] for a in arrs]

    def _augment(self, img0, gt, img1, t, mv):
        if random.random() < 0.5:                               # rotate180
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
            ig, i1, t, want_mv = i0 + 1, i0 + 2, 0.5, False
        else:
            scene = self._pick_scene(index)
            i0, ig, i1, t, want_mv = self._sample_window(scene)

        frames = self._frames_of(scene)
        img0 = self._read_img(frames[i0])
        gt = self._read_img(frames[ig])
        img1 = self._read_img(frames[i1])

        mv = None
        if want_mv:
            h, w = gt.shape[:2]
            mv = self._read_mv_pair(scene, frames[ig], h, w)    # 读取失败→None

        arrs = [img0, gt, img1] + ([mv] if mv is not None else [])
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
                 **tier_kwargs):
        self.datasets: Dict[str, TierDataset] = {
            name: TierDataset(root, lf, split='train', **tier_kwargs)
            for name, lf in lists.items()
        }
        self.set_ratios(ratios)
        self._epoch_size = sum(len(d) for d in self.datasets.values())

    def set_ratios(self, ratios: Dict[str, float]) -> None:
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
        logger.info(f'tier配比: {dict(zip(self._names, [round(p,3) for p in self._probs]))}')

    def set_crop_size(self, crop_hw: Tuple[int, int]) -> None:
        for d in self.datasets.values():
            d.crop_hw = crop_hw

    def __len__(self) -> int:
        return self._epoch_size

    def __getitem__(self, index: int):
        name = random.choices(self._names, weights=self._probs, k=1)[0]
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
