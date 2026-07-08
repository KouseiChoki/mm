'''
VFI 训练数据指标统计脚本 (仅输出指标, 不做阈值筛选)
=====================================================
扫描 scenes/<video>/scene_XXXX/ 目录结构, 对每个三元组 (img0, gt, img1) 计算:

  运动幅度   : flow_mean / flow_p95        (img0→img1 光流位移统计, 全分辨率像素)
  运动占比   : moving_ratio                (位移 > 1px 的像素占比)
  遮挡/非线性: fb_ratio / fb_epe           (forward-backward consistency)
  重复帧     : ssim_g0 / ssim_g1           (GT 与两输入帧的 SSIM)
  静止       : ssim_01                     (img0 与 img1 的 SSIM)
  亮度突变   : luma_jump                   (闪光/叠化/漏切检测)
  线性度     : linearity                   (可选, |F(0→g) - F(g→1)| 归一化, 需 --linearity)

输出:
  <output>/metadata.csv      每个三元组一行的完整指标 (后续分档/筛选在此基础上进行)
  <output>/stats.csv         各指标的分布统计 (min/P10/P30/P50/P75/P90/P99/max/mean)
  <output>/summary.txt       同 stats 的可读版本

用法:
  python measure_vfi_data.py --scenes_root /data/out/scenes --output /data/out/clean \
      --stride 1 --flow_max_w 512 --linearity

依赖: pip install opencv-python-headless numpy tqdm
'''
import re
import csv
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

IMG_EXTS = {'.png', '.jpg', '.jpeg'}


# ─────────────────────────────────────────────────────────────────────────────
# 基础工具
# ─────────────────────────────────────────────────────────────────────────────

def extract_number(p: Path) -> int:
    nums = re.findall(r'(\d+)', p.stem)
    return int(nums[-1]) if nums else -1


def list_frames(scene_dir: Path) -> List[Path]:
    return sorted((p for p in scene_dir.iterdir()
                   if p.suffix.lower() in IMG_EXTS and not p.name.startswith('.')),
                  key=extract_number)


def load_gray_small(path: Path, max_w: int) -> Optional[np.ndarray]:
    """读图转灰度, 等比缩到不超过 max_w 宽 (用于光流/SSIM, 控制计算量)。"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_w:
        s = max_w / w
        img = cv2.resize(img, (max_w, int(round(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """轻量 SSIM (灰度, 高斯窗), 避免引入 skimage 依赖。输入 float32 [0,255]。"""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu_a = cv2.filter2D(a, -1, window)[5:-5, 5:-5]
    mu_b = cv2.filter2D(b, -1, window)[5:-5, 5:-5]
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sig_a2 = cv2.filter2D(a * a, -1, window)[5:-5, 5:-5] - mu_a2
    sig_b2 = cv2.filter2D(b * b, -1, window)[5:-5, 5:-5] - mu_b2
    sig_ab = cv2.filter2D(a * b, -1, window)[5:-5, 5:-5] - mu_ab

    s = ((2 * mu_ab + C1) * (2 * sig_ab + C2)) / \
        ((mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2))
    return float(s.mean())


def farneback(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Farneback 光流 a→b, 输入 float32 灰度, 返回 (H,W,2) float32 像素位移。"""
    return cv2.calcOpticalFlowFarneback(
        a.astype(np.uint8), b.astype(np.uint8), None,
        pyr_scale=0.5, levels=5, winsize=21,
        iterations=3, poly_n=7, poly_sigma=1.5, flags=0)


def warp_flow(flow_src: np.ndarray, flow_by: np.ndarray) -> np.ndarray:
    """按 flow_by 对 flow_src 做 backward warp (采样 flow_src 在 x+flow_by 处的值)。"""
    h, w = flow_by.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    map_x = gx + flow_by[..., 0]
    map_y = gy + flow_by[..., 1]
    return cv2.remap(flow_src, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def fb_consistency(f01: np.ndarray, f10: np.ndarray,
                   alpha: float = 0.01, beta: float = 0.5) -> Tuple[float, float]:
    """
    forward-backward consistency (Sundaram et al. 阈值形式):
      |F01 + warp(F10 by F01)|^2 > alpha*(|F01|^2+|warp F10|^2) + beta  → 不一致
    返回 (不一致像素占比, fb_epe 均值)。
    """
    f10w = warp_flow(f10, f01)
    diff = f01 + f10w
    diff_sq = (diff ** 2).sum(-1)
    mag_sq = (f01 ** 2).sum(-1) + (f10w ** 2).sum(-1)
    incons = diff_sq > (alpha * mag_sq + beta)
    fb_epe = float(np.sqrt(diff_sq).mean())
    return float(incons.mean()), fb_epe


# ─────────────────────────────────────────────────────────────────────────────
# 逐三元组指标
# ─────────────────────────────────────────────────────────────────────────────

METRIC_FIELDS = ['flow_mean', 'flow_p95', 'moving_ratio', 'fb_ratio', 'fb_epe',
                 'ssim_01', 'ssim_g0', 'ssim_g1', 'luma_jump', 'linearity']


def measure_triplet(g0: np.ndarray, gm: np.ndarray, g1: np.ndarray,
                    full_w: int, compute_linearity: bool) -> dict:
    """g0/gm/g1: 缩小后的灰度帧 (img0 / gt / img1)。full_w: 原图宽, 用于把位移换算回全分辨率。"""
    scale_back = full_w / g0.shape[1]

    f01 = farneback(g0, g1)
    f10 = farneback(g1, g0)

    mag = np.sqrt((f01 ** 2).sum(-1))
    flow_mean = float(mag.mean()) * scale_back
    flow_p95 = float(np.percentile(mag, 95)) * scale_back
    moving_ratio = float((mag * scale_back > 1.0).mean())   # 位移>1px(全分辨率)的像素占比

    fb_ratio, fb_epe = fb_consistency(f01, f10)
    fb_epe *= scale_back

    m = {
        'flow_mean': flow_mean,
        'flow_p95': flow_p95,
        'moving_ratio': moving_ratio,
        'fb_ratio': fb_ratio,
        'fb_epe': fb_epe,
        'ssim_01': ssim_gray(g0, g1),
        'ssim_g0': ssim_gray(gm, g0),
        'ssim_g1': ssim_gray(gm, g1),
        'luma_jump': max(abs(float(gm.mean()) - float(g0.mean())),
                         abs(float(g1.mean()) - float(gm.mean()))),
        'linearity': -1.0,
    }

    if compute_linearity:
        f0m = farneback(g0, gm)
        fm1 = farneback(gm, g1)
        num = np.sqrt(((f0m - fm1) ** 2).sum(-1)).mean()
        den = np.sqrt((f0m ** 2).sum(-1)).mean() + np.sqrt((fm1 ** 2).sum(-1)).mean() + 1e-6
        m['linearity'] = float(num / den)   # 0=完全匀速, 越大越非线性

    return m


def collect(scenes_root: Path, stride: int, flow_max_w: int,
            compute_linearity: bool) -> List[dict]:
    rows = []
    scene_dirs = sorted(p for v in sorted(scenes_root.iterdir()) if v.is_dir()
                        for p in sorted(v.iterdir()) if p.is_dir())
    if not scene_dirs:
        logger.error(f'未找到场景目录: {scenes_root}/*/scene_*')
        return rows

    logger.info(f'{len(scene_dirs)} 个场景目录, stride={stride}, linearity={compute_linearity}')

    for sdir in tqdm(scene_dirs, desc='指标计算', unit='scene'):
        frames = list_frames(sdir)
        if len(frames) < 2 * stride + 1:
            continue

        # 缓存缩小灰度帧, 避免同一帧重复读取解码
        cache: dict = {}
        full_w = None

        def get_gray(idx: int) -> Optional[np.ndarray]:
            nonlocal full_w
            if idx not in cache:
                if full_w is None:
                    probe = cv2.imread(str(frames[idx]), cv2.IMREAD_COLOR)
                    if probe is None:
                        cache[idx] = None
                        return None
                    full_w = probe.shape[1]
                    if probe.shape[1] > flow_max_w:
                        s = flow_max_w / probe.shape[1]
                        probe = cv2.resize(probe, (flow_max_w, int(round(probe.shape[0] * s))),
                                           interpolation=cv2.INTER_AREA)
                    cache[idx] = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY).astype(np.float32)
                else:
                    cache[idx] = load_gray_small(frames[idx], flow_max_w)
            return cache[idx]

        for i in range(0, len(frames) - 2 * stride):
            i0, im, i1 = i, i + stride, i + 2 * stride
            g0, gm, g1 = get_gray(i0), get_gray(im), get_gray(i1)
            if g0 is None or gm is None or g1 is None:
                continue

            m = measure_triplet(g0, gm, g1, full_w, compute_linearity)
            m.update({
                'scene': str(sdir.relative_to(scenes_root)),
                'img0': str(frames[i0]),
                'gt': str(frames[im]),
                'img1': str(frames[i1]),
            })
            rows.append(m)

            # 滑动窗口: 只保留后续还会用到的帧
            for k in list(cache.keys()):
                if k <= i:
                    del cache[k]

    logger.info(f'完成: {len(rows)} 个三元组')
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 输出: 元数据 + 分布统计
# ─────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = ['scene', 'img0', 'gt', 'img1'] + METRIC_FIELDS

PERCENTILES = [0, 10, 30, 50, 75, 90, 99, 100]
STAT_COLS = ['metric', 'mean'] + [f'P{p}' for p in PERCENTILES]


def compute_stats(rows: List[dict], compute_linearity: bool) -> List[dict]:
    stats = []
    for k in METRIC_FIELDS:
        if k == 'linearity' and not compute_linearity:
            continue
        vals = np.array([m[k] for m in rows], dtype=np.float64)
        row = {'metric': k, 'mean': float(vals.mean())}
        for p in PERCENTILES:
            row[f'P{p}'] = float(np.percentile(vals, p))
        stats.append(row)
    return stats


def save_outputs(rows: List[dict], out_dir: Path, compute_linearity: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 逐三元组元数据
    csv_path = out_dir / 'metadata.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for m in rows:
            w.writerow({k: (f'{m[k]:.4f}' if isinstance(m[k], float) else m[k])
                        for k in CSV_FIELDS})
    logger.info(f'元数据: {csv_path}')

    # 分布统计
    stats = compute_stats(rows, compute_linearity)
    stats_path = out_dir / 'stats.csv'
    with open(stats_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=STAT_COLS)
        w.writeheader()
        for s in stats:
            w.writerow({k: (f'{s[k]:.4f}' if isinstance(s[k], float) else s[k])
                        for k in STAT_COLS})
    logger.info(f'分布统计: {stats_path}')

    # 可读 summary
    lines = [f'总三元组数: {len(rows)}', '']
    header = f"{'metric':<14}" + f"{'mean':>10}" + ''.join(f"{f'P{p}':>10}" for p in PERCENTILES)
    lines.append(header)
    lines.append('-' * len(header))
    for s in stats:
        lines.append(f"{s['metric']:<14}" + f"{s['mean']:>10.4f}"
                     + ''.join(f"{s[f'P{p}']:>10.4f}" for p in PERCENTILES))
    summary = '\n'.join(lines)
    (out_dir / 'summary.txt').write_text(summary)
    logger.info('\n' + summary)


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='VFI 数据指标统计 (仅输出指标与分布, 不做筛选)')
    p.add_argument('--scenes_root', type=str, required=True,
                   help='scenes 根目录 (scenes/<video>/scene_XXXX/*.png)')
    p.add_argument('--output', type=str, required=True,
                   help='输出目录 (metadata.csv / stats.csv / summary.txt)')
    p.add_argument('--stride', type=int, default=1,
                   help='三元组帧间隔: (i, i+s, i+2s), 默认 1')
    p.add_argument('--flow_max_w', type=int, default=512,
                   help='光流/SSIM 计算用的缩小宽度上限 (默认 512, 位移统计会换算回全分辨率)')
    p.add_argument('--linearity', action='store_true',
                   help='额外计算窗口线性度 (多两次光流, 慢约一倍; 任意timestep训练建议开)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect(Path(args.scenes_root), args.stride,
                   args.flow_max_w, args.linearity)
    if not rows:
        return
    save_outputs(rows, Path(args.output), args.linearity)


if __name__ == '__main__':
    main()