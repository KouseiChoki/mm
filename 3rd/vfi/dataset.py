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
  flow_gt : [4, H, W] float32 (像素位移; has_mv=0 时为全零占位, 保证collate形状统一)
  has_mv  : [] float  (1=本样本flow_gt有效, 0=无效, loss侧用它mask)

用法:
  train_ds = MixedTierDataset(root, lists={'easy':..., 'normal':..., 'hard':..., 'teacher':...},
                              ratios={'easy':0.4,'normal':0.4,'hard':0.1,'teacher':0.1},
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

IMG_EXTS = {'.png', '.jpg', '.jpeg'}
MV_EXTS = {'.exr'}


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
    """

    def __init__(self, root, list_file, split='train',
                 crop_hw: Optional[Tuple[int, int]] = (256, 448),
                 framesteps: Sequence[int] = (1, 2),
                 t_half_prob: float = 0.6,
                 mv_prob: float = 0.5,
                 mv_sign: Tuple[int, int] = (1, 1),
                 augment: bool = True):
        self.root = Path(root)
        self.split = split
        self.crop_hw = crop_hw if split == 'train' else None
        self.framesteps = tuple(framesteps)
        self.t_half_prob = t_half_prob
        self.mv_prob = mv_prob
        self.mv_sign = mv_sign
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
        """读取 gt 帧的 mv1/mv0, 反归一化到像素。损坏/缺失返回 None (由上层降级处理)。"""
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
        # 网络约定: flow[:, :2]=F_t→0=mv1(gt→上一帧),  flow[:,2:4]=F_t→1=mv0(gt→下一帧)
        return np.concatenate([mv1, mv0], axis=-1)              # [H,W,4]

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
    def _crop(self, arrs: List[np.ndarray], has_flow: bool = False):
        """随机裁剪到 self.crop_hw; 源图不足时reflect-pad补齐。
        返回 (arrs, mv_still_valid): pad过的样本flow监督失效。"""
        if self.crop_hw is None:
            return arrs, True
        h, w = self.crop_hw
        ih, iw = arrs[0].shape[:2]
        mv_ok = True
        if ih < h or iw < w:
            ph, pw = max(h - ih, 0), max(w - iw, 0)
            arrs = [cv2.copyMakeBorder(a, 0, ph, 0, pw, cv2.BORDER_REFLECT)
                    for a in arrs]
            if has_flow:
                mv_ok = False                      # pad区flow无效, 丢弃监督
            ih, iw = arrs[0].shape[:2]
        if ih == h and iw == w:
            return arrs, mv_ok
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        return [a[x:x + h, y:y + w] for a in arrs], mv_ok

    def _augment(self, img0, gt, img1, t, mv):
        if random.random() < 0.5:                               # rotate180
            img0 = cv2.rotate(img0, cv2.ROTATE_180)
            gt = cv2.rotate(gt, cv2.ROTATE_180)
            img1 = cv2.rotate(img1, cv2.ROTATE_180)
            if mv is not None:
                mv = cv2.rotate(mv, cv2.ROTATE_180)
                mv = mv * -1.0                                  # 位移方向全部反转
        if random.random() < 0.5:                               # 时序翻转
            img0, img1 = img1, img0
            t = 1.0 - t
            if mv is not None:
                mv = mv[..., [2, 3, 0, 1]]                      # F_t→0 ↔ F_t→1
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
        arrs, mv_ok = self._crop(arrs, has_flow=(mv is not None))
        img0, gt, img1 = arrs[0], arrs[1], arrs[2]
        mv = arrs[3] if (mv is not None and mv_ok) else None

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
            flow_gt = torch.zeros(4, h, w)
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
        names = [n for n in ratios if n in self.datasets and ratios[n] > 0]
        total = sum(ratios[n] for n in names)
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
    lists = {t: str(lists_dir / f'{t}_train.txt')
             for t in ('easy', 'normal', 'hard', 'teacher')
             if (lists_dir / f'{t}_train.txt').exists()}

    ds = MixedTierDataset(args.root, lists,
                          ratios={'easy': .4, 'normal': .4, 'hard': .1, 'teacher': .1},
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