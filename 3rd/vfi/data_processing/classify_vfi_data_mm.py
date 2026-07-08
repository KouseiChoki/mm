'''
VFI 数据难度分档脚本 (载入现成光流 EXR 版)
============================================
不计算光流, 直接读取模型 dump 出的 mv0/mv1 光流文件。

输入目录结构 (dump_debug_data 的输出):
    root/
    ├── scene1/
    │   ├── xxx_mv0/  (或 mv0/, 名字包含 "mv0" 即可)
    │   │   ├── 000001.exr
    │   │   └── ...
    │   └── xxx_mv1/
    │       ├── 000001.exr
    │       └── ...
    └── scene2/ ...

mv0/mv1 语义: 中间帧 → img0 / img1 的光流。
t=0.5 线性运动下应满足 mv0 ≈ -mv1, 因此用 |mv0+mv1| 的对称性残差
作为遮挡/非线性代理指标 (sym_ratio), 替代原先的 forward-backward 检查。

指标 (逐帧计算, 逐 scene 聚合):
  flow_mean / flow_p95 : mv0,mv1 合并的位移统计 (像素)
  moving_ratio         : 位移 > 1px 的像素占比
  sym_epe              : |mv0 + mv1| 均值 (像素)
  sym_ratio            : sym_epe / (mean|mv0| + mean|mv1|), 0=完全对称匀速

分档 (scene 级):
  NG   绝对规则: static (moving_ratio 过低) / flow_failure (sym_ratio 过高)
  easy / medium / hard: 非 NG scene 按 flow_p95 与 sym_ratio 的分位数 (默认 P30/P75)

输出:
  <output>/metadata_frames.csv   逐帧指标
  <output>/metadata_scenes.csv   逐 scene 聚合指标 + 档位
  <output>/summary.txt           档位统计

用法:
  python classify_vfi_flow.py --root /data/dump_out --output /data/clean
  # 若 exr 中存的是归一化流(除过宽高, 同 evaluate 的 norm=True), 加 --normalized
  python classify_vfi_flow.py --root ... --output ... --normalized
'''
import os
import sys
import re
import csv
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── file_utils.read 导入 (与 evaluate 脚本同款路径处理), 失败则回退 cv2 ──────
cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cur_path + '/../algo')
try:
    from file_utils import read          # noqa: E402
    _HAS_FILE_UTILS = True
except ImportError:
    _HAS_FILE_UTILS = False
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    import cv2

    def read(path, type='flo', **kwargs):        # 最小回退实现, 仅支持 exr flo
        if path is None or not os.path.isfile(path):
            return None
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        return np.ascontiguousarray(img.astype(np.float32))

    logger.warning('未找到 file_utils, 使用 cv2 回退读取 exr (通道顺序可能与 mvread 不同, 建议在正式环境运行)')


# ─────────────────────────────────────────────────────────────────────────────
# 目录/文件配对 (沿用 evaluate 脚本的序号匹配思路)
# ─────────────────────────────────────────────────────────────────────────────

def extract_number(p: Path) -> Optional[int]:
    nums = re.findall(r'(\d+)', p.name)
    return int(nums[-1]) if nums else None


def find_mv_dirs(scene_dir: Path):
    """在 scene 目录下找名字包含 mv0 / mv1 的子文件夹。"""
    mv0_dir = mv1_dir = None
    for d in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
        if 'mv0' in d.name:
            mv0_dir = d
        elif 'mv1' in d.name:
            mv1_dir = d
    return mv0_dir, mv1_dir


def list_exrs(d: Path) -> Dict[int, Path]:
    """返回 {帧序号: 路径}, 便于 mv0/mv1 按序号配对。"""
    out = {}
    for p in sorted(d.iterdir()):
        if p.suffix.lower() == '.exr' and not p.name.startswith('.'):
            n = extract_number(p)
            if n is not None:
                out[n] = p
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 逐帧指标
# ─────────────────────────────────────────────────────────────────────────────

def measure_frame(mv0: np.ndarray, mv1: np.ndarray, normalized: bool) -> Optional[dict]:
    mv0 = mv0[..., :2].astype(np.float32)
    mv1 = mv1[..., :2].astype(np.float32)
    if mv0.shape != mv1.shape:
        return None

    if normalized:                       # 归一化流 → 像素 (同 evaluate 的 norm=True)
        h, w = mv0.shape[:2]
        scale = np.array([w, h], dtype=np.float32)
        mv0 = mv0 * scale
        mv1 = mv1 * scale

    mag0 = np.sqrt((mv0 ** 2).sum(-1))
    mag1 = np.sqrt((mv1 ** 2).sum(-1))
    mag_all = np.concatenate([mag0.ravel(), mag1.ravel()])

    sym = np.sqrt(((mv0 + mv1) ** 2).sum(-1))     # t=0.5 线性运动 → mv0≈-mv1 → sym≈0
    sym_epe = float(sym.mean())
    denom = float(mag0.mean() + mag1.mean()) + 1e-6

    return {
        'flow_mean': float(mag_all.mean()),
        'flow_p95': float(np.percentile(mag_all, 95)),
        'moving_ratio': float((mag_all > 1.0).mean()),
        'sym_epe': sym_epe,
        'sym_ratio': sym_epe / denom,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1: 逐帧 → 逐 scene 聚合
# ─────────────────────────────────────────────────────────────────────────────

FRAME_METRICS = ['flow_mean', 'flow_p95', 'moving_ratio', 'sym_epe', 'sym_ratio']


def pass1_collect(root: Path, normalized: bool, skip: int):
    frame_rows, scene_rows = [], []
    scene_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not scene_dirs:
        logger.error(f'未找到 scene 目录: {root}')
        return frame_rows, scene_rows

    logger.info(f'Pass1: {len(scene_dirs)} 个 scene, normalized={normalized}, skip={skip}')

    for sdir in tqdm(scene_dirs, desc='Pass1 读取mv统计', unit='scene'):
        mv0_dir, mv1_dir = find_mv_dirs(sdir)
        if mv0_dir is None or mv1_dir is None:
            logger.warning(f'  {sdir.name}: 缺少 mv0/mv1 子目录, 跳过')
            continue

        mv0_files = list_exrs(mv0_dir)
        mv1_files = list_exrs(mv1_dir)
        common = sorted(set(mv0_files) & set(mv1_files))
        if skip > 0:
            common = common[skip:-skip] if len(common) > 2 * skip else []
        if not common:
            logger.warning(f'  {sdir.name}: 无可配对的 mv0/mv1 帧, 跳过')
            continue

        per_scene = {k: [] for k in FRAME_METRICS}
        for n in common:
            mv0 = read(str(mv0_files[n]), type='flo')
            mv1 = read(str(mv1_files[n]), type='flo')
            if mv0 is None or mv1 is None:
                continue
            m = measure_frame(mv0, mv1, normalized)
            if m is None:
                logger.warning(f'  {sdir.name} 帧{n}: mv0/mv1 尺寸不一致, 跳过')
                continue
            m.update({'scene': sdir.name, 'frame': n})
            frame_rows.append(m)
            for k in FRAME_METRICS:
                per_scene[k].append(m[k])

        if not per_scene['flow_p95']:
            continue

        scene_rows.append({
            'scene': sdir.name,
            'n_frames': len(per_scene['flow_p95']),
            'flow_mean': float(np.mean(per_scene['flow_mean'])),
            'flow_p95': float(np.percentile(per_scene['flow_p95'], 95)),   # 帧级p95的p95: 捕捉峰值运动
            'moving_ratio': float(np.mean(per_scene['moving_ratio'])),
            'sym_epe': float(np.mean(per_scene['sym_epe'])),
            'sym_ratio': float(np.mean(per_scene['sym_ratio'])),
        })

    logger.info(f'Pass1 完成: {len(scene_rows)} 个 scene, {len(frame_rows)} 帧')
    return frame_rows, scene_rows


# ─────────────────────────────────────────────────────────────────────────────
# Pass 2: scene 级 NG + 分位数分档
# ─────────────────────────────────────────────────────────────────────────────

def ng_reason(s: dict, args) -> str:
    if s['moving_ratio'] < args.static_moving:
        return 'static'                  # 几乎无运动 → 无监督价值
    if s['sym_ratio'] > args.ng_sym_ratio:
        return 'flow_failure'            # mv0/mv1 大面积不对称 → 流不可信/强非线性
    return ''


def pass2_classify(scene_rows: List[dict], args) -> None:
    for s in scene_rows:
        s['ng_reason'] = ng_reason(s, args)

    valid = [s for s in scene_rows if not s['ng_reason']]
    if not valid:
        logger.warning('全部 scene 被判为 NG, 请检查阈值')
        for s in scene_rows:
            s['tier'] = 'ng'
        return

    p_lo_flow = np.percentile([s['flow_p95'] for s in valid], args.p_easy)
    p_hi_flow = np.percentile([s['flow_p95'] for s in valid], args.p_hard)
    p_lo_sym = np.percentile([s['sym_ratio'] for s in valid], args.p_easy)
    p_hi_sym = np.percentile([s['sym_ratio'] for s in valid], args.p_hard)

    logger.info(f'分位阈值: flow_p95 P{args.p_easy}={p_lo_flow:.2f}px  '
                f'P{args.p_hard}={p_hi_flow:.2f}px | '
                f'sym_ratio P{args.p_easy}={p_lo_sym:.4f}  P{args.p_hard}={p_hi_sym:.4f}')

    for s in scene_rows:
        if s['ng_reason']:
            s['tier'] = 'ng'
        elif s['flow_p95'] > p_hi_flow or s['sym_ratio'] > p_hi_sym:
            s['tier'] = 'hard'
        elif s['flow_p95'] <= p_lo_flow and s['sym_ratio'] <= p_lo_sym:
            s['tier'] = 'easy'
        else:
            s['tier'] = 'medium'


# ─────────────────────────────────────────────────────────────────────────────
# 输出
# ─────────────────────────────────────────────────────────────────────────────

FRAME_FIELDS = ['scene', 'frame'] + FRAME_METRICS
SCENE_FIELDS = ['scene', 'n_frames'] + FRAME_METRICS + ['ng_reason', 'tier']


def _write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{r[k]:.4f}' if isinstance(r[k], float) else r[k])
                        for k in fields})


def save_outputs(frame_rows, scene_rows, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(out_dir / 'metadata_frames.csv', frame_rows, FRAME_FIELDS)
    _write_csv(out_dir / 'metadata_scenes.csv', scene_rows, SCENE_FIELDS)
    logger.info(f'元数据: {out_dir}/metadata_frames.csv, metadata_scenes.csv')

    tiers = ['easy', 'medium', 'hard', 'ng']
    counts = {t: sum(1 for s in scene_rows if s['tier'] == t) for t in tiers}
    lines = [f'总 scene 数: {len(scene_rows)}']
    for t in tiers:
        names = [s['scene'] for s in scene_rows if s['tier'] == t]
        lines.append(f'  {t:<7}: {counts[t]:>5}  ({counts[t] / max(len(scene_rows), 1):.1%})'
                     + (f'  → {", ".join(names)}' if names else ''))
    ng_breakdown: dict = {}
    for s in scene_rows:
        if s['ng_reason']:
            ng_breakdown[s['ng_reason']] = ng_breakdown.get(s['ng_reason'], 0) + 1
    if ng_breakdown:
        lines.append('NG 明细:')
        for k, v in sorted(ng_breakdown.items(), key=lambda x: -x[1]):
            lines.append(f'  {k:<20}: {v}')
    summary = '\n'.join(lines)
    (out_dir / 'summary.txt').write_text(summary)
    logger.info('\n' + summary)


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='VFI 数据分档 (载入 mv0/mv1 光流 EXR, scene 级统计)')
    p.add_argument('--root', type=str, required=True,
                   help='根目录: root/<scene>/<*mv0*>/*.exr 与 <*mv1*>/*.exr')
    p.add_argument('--output', type=str, required=True,
                   help='输出目录')
    p.add_argument('--normalized', action='store_true',
                   help='exr 中的流为归一化值(已除以宽高, 同 evaluate 的 norm=True), 统计前换算回像素')
    p.add_argument('--skip', type=int, default=0,
                   help='每个 scene 掐头去尾跳过的帧数 (同 evaluate 的 skip, 默认 0)')

    g_ng = p.add_argument_group('NG 绝对阈值 (scene 级)')
    g_ng.add_argument('--static_moving', type=float, default=0.005,
                      help='运动像素占比低于此值判静止 (默认 0.5%%)')
    g_ng.add_argument('--ng_sym_ratio', type=float, default=0.6,
                      help='mv0/mv1 对称残差比超过此值判光流失败/强非线性 (默认 0.6)')

    g_q = p.add_argument_group('分位数分档')
    g_q.add_argument('--p_easy', type=float, default=30, help='easy 上界分位数 (默认 P30)')
    g_q.add_argument('--p_hard', type=float, default=75, help='hard 下界分位数 (默认 P75)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frame_rows, scene_rows = pass1_collect(Path(args.root), args.normalized, args.skip)
    if not scene_rows:
        return
    pass2_classify(scene_rows, args)
    save_outputs(frame_rows, scene_rows, Path(args.output))


if __name__ == '__main__':
    main()