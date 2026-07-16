'''
VFI 训练数据清单生成脚本 (build_lists.py)
==========================================
扫描数据根目录, 生成 scene 级训练/验证清单 (txt), 供 Dataset 运行时按
framestep/timestep 动态采样三元组使用。大数据集只需扫描一次, 之后训练只读txt。

输入目录结构:
    root/
    ├── easy/   电影名/scene0001/*.png|jpg        ← tier 数据
    ├── normal/ 电影名/scene0001/*.png|jpg
    ├── hard/   电影名/scene0001/*.png|jpg
    ├── val/    电影名/scene0001/*.png|jpg        ← 预划分的验证集
    └── teacher/ ...任意深度.../12fps|24fps|48fps/{image, mv0, mv1}/
                 image/*.png|jpg  mv0/*.exr  mv1/*.exr
                 (同时包含 image+mv0+mv1 三个子目录的文件夹即为一个 teacher scene,
                  fps 从路径中的 "12fps/24fps/48fps" 目录名解析)

输出 (默认写到 <root>/lists/):
    easy_train.txt / normal_train.txt / hard_train.txt / teacher_train.txt / val.txt
        每行 (tab分隔):  scene相对路径\t帧数\ttier\thas_mv\tfps
        - scene相对路径: 相对 root, 训练时 root+相对路径 定位
        - has_mv: teacher=1, 其余=0
        - fps: teacher 从路径解析 (12/24/48), 其余=0
    broken_scenes.txt   被剔除的 scene 及原因 (帧号断号 / 帧数不足 / mv不齐)
    summary.txt         各清单统计

校验规则:
    - 帧号必须连续 (按文件名末尾数字), 断号 scene 剔除并记录 (--allow_gaps 保留)
    - 帧数 < 2*max_framestep+1 的 scene 剔除 (采不出最大间隔的三元组)
    - teacher: image/mv0/mv1 三方按帧号配对, 配对率 < 90% 的 scene 剔除并记录

用法:
    python build_lists.py --root /home/zhenying/qhong/data/ssd/vfi_database --max_framestep 2
'''
import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

IMG_EXTS = {'.png', '.jpg', '.jpeg'}
MV_EXTS = {'.exr'}
TIERS = ('easy', 'normal', 'hard')          # 常规tier目录
VAL_DIR = 'val'
TEACHER_DIR = 'teacher'

_FPS_RE = re.compile(r'^(\d+)\s*fps$', re.IGNORECASE)


def extract_number(name: str) -> Optional[int]:
    nums = re.findall(r'(\d+)', Path(name).stem)
    return int(nums[-1]) if nums else None


def frame_entries(d: Path, exts) -> List[Tuple[int, str]]:
    """目录下帧文件的 (帧号, 文件名) 列表 (按帧号升序), 无法解析帧号的文件忽略。
    用 scandir 的文件名直接判断, 不对每帧做 stat (NFS/大目录下快一个量级)。"""
    entries = []
    for e in os.scandir(d):
        name = e.name
        if name.startswith('.'):
            continue
        dot = name.rfind('.')
        if dot < 0 or name[dot:].lower() not in exts:
            continue
        n = extract_number(name)
        if n is not None:
            entries.append((n, name))
    return sorted(entries)


def frame_numbers(d: Path, exts) -> List[int]:
    return [n for n, _ in frame_entries(d, exts)]


def check_contiguous(nums: List[int]) -> Tuple[bool, int]:
    """帧号是否连续; 返回 (连续?, 缺口数)。"""
    if len(nums) < 2:
        return True, 0
    gaps = sum(1 for a, b in zip(nums, nums[1:]) if b - a != 1)
    return gaps == 0, gaps


def fps_from_path(p: Path) -> int:
    """从路径各级目录名解析 fps (如 '12fps'), 找不到返回 0。"""
    for part in reversed(p.parts):
        m = _FPS_RE.match(part.strip())
        if m:
            return int(m.group(1))
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 常规 tier / val 扫描:  <dir>/电影名/sceneXXXX/*.png
# ─────────────────────────────────────────────────────────────────────────────

def scan_tier(root: Path, tier_dir: str, tier_label: str,
              min_frames: int, allow_gaps: bool, broken: list) -> List[dict]:
    base = root / tier_dir
    rows = []
    if not base.is_dir():
        logger.warning(f'目录不存在, 跳过: {base}')
        return rows

    # scandir 避免逐条目 stat (NFS/大目录下 iterdir+is_dir 极慢); 电影层带进度
    movies = sorted(e.path for e in os.scandir(base) if e.is_dir())
    scene_dirs = []
    for movie in tqdm(movies, desc=f'收集 {tier_label} scene目录', unit='movie'):
        scene_dirs.extend(sorted(Path(e.path) for e in os.scandir(movie)
                                 if e.is_dir()))
    for sdir in tqdm(scene_dirs, desc=f'扫描 {tier_label}', unit='scene'):
        rel = sdir.relative_to(root)
        entries = frame_entries(sdir, IMG_EXTS)
        nums = [n for n, _ in entries]
        if len(nums) < min_frames:
            broken.append(f'{rel}\t帧数不足({len(nums)}<{min_frames})')
            continue
        ok, gaps = check_contiguous(nums)
        if not ok and not allow_gaps:
            broken.append(f'{rel}\t帧号断号({gaps}处缺口, 共{len(nums)}帧)')
            continue
        rows.append({'rel': str(rel), 'n': len(nums), 'tier': tier_label,
                     'has_mv': 0, 'fps': 0,
                     'frames': [name for _, name in entries]})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# teacher 扫描: 任意深度下同时包含 image/mv0/mv1 的目录
# ─────────────────────────────────────────────────────────────────────────────

def scan_teacher(root: Path, min_frames: int, allow_gaps: bool,
                 pair_ratio: float, broken: list) -> List[dict]:
    base = root / TEACHER_DIR
    rows = []
    
    if not base.is_dir():
        logger.warning(f'目录不存在, 跳过: {base}')
        return rows

    pbar = tqdm(desc='扫描 teacher (遍历目录)', unit='dir')
    for dirpath, dirnames, _ in os.walk(base):
        pbar.update(1)
        if not {'image', 'mv0', 'mv1'}.issubset(set(dirnames)):
            continue
        sdir = Path(dirpath)
        fps = fps_from_path(sdir)
        if args.teacher_fps is not None and fps not in args.teacher_fps:
            continue

        dirnames[:] = []                      # 命中即剪枝, 不再深入
        rel = sdir.relative_to(root)
        pbar.set_postfix(scenes=len(rows) + 1)

        img_entries = frame_entries(sdir / 'image', IMG_EXTS)
        img_nums = [n for n, _ in img_entries]
        mv0_nums = set(frame_numbers(sdir / 'mv0', MV_EXTS))
        mv1_nums = set(frame_numbers(sdir / 'mv1', MV_EXTS))

        if len(img_nums) < min_frames:
            broken.append(f'{rel}\t帧数不足({len(img_nums)}<{min_frames})')
            continue
        ok, gaps = check_contiguous(img_nums)
        if not ok and not allow_gaps:
            broken.append(f'{rel}\t帧号断号({gaps}处缺口)')
            continue

        paired = sum(1 for n in img_nums if n in mv0_nums and n in mv1_nums)
        ratio = paired / max(len(img_nums), 1)
        if ratio < pair_ratio:
            broken.append(f'{rel}\tmv配对率过低({paired}/{len(img_nums)}={ratio:.0%})')
            continue

        rows.append({'rel': str(rel), 'n': len(img_nums), 'tier': 'teacher',
                     'has_mv': 1, 'fps': fps_from_path(sdir),
                     'frames': [name for _, name in img_entries]})
    pbar.close()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 输出
# ─────────────────────────────────────────────────────────────────────────────

def write_list(path: Path, rows: List[dict]) -> None:
    with open(path, 'w') as f:
        for r in rows:
            f.write(f"{r['rel']}\t{r['n']}\t{r['tier']}\t{r['has_mv']}\t{r['fps']}\n")
    # 帧文件名索引: Dataset 直接读, 训练时零目录列举
    idx_path = path.with_suffix('.frames.json')
    with open(idx_path, 'w') as f:
        json.dump({r['rel']: r['frames'] for r in rows}, f)
    logger.info(f'  {path.name}: {len(rows)} scenes, {sum(r["n"] for r in rows)} 帧 '
                f'(+帧索引 {idx_path.name})')


def main() -> None:
    p = argparse.ArgumentParser(description='VFI 数据清单生成 (scene级txt)')
    p.add_argument('--root', type=str, required=True,
                   help='数据根目录 (含 easy/normal/hard/val/teacher)')
    p.add_argument('--output', type=str, default=None,
                   help='清单输出目录 (默认 <root>/lists)')
    p.add_argument('--max_framestep', type=int, default=4,
                   help='训练最大framestep, 决定最小帧数 2*s+1 (默认 4 → 最少9帧)')
    p.add_argument('--allow_gaps', action='store_true',
                   help='允许帧号断号的scene入库 (默认剔除并记录)')
    p.add_argument('--pair_ratio', type=float, default=0.9,
                   help='teacher scene 的 image/mv 配对率下限 (默认 0.9)')
    p.add_argument('--teacher_fps', type=int, nargs='+', default=None,
               help='仅保留指定fps的teacher scene, 如 --teacher_fps 24 48 (默认全收)')
    args = p.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output) if args.output else root / 'lists'
    out_dir.mkdir(parents=True, exist_ok=True)
    min_frames = 2 * args.max_framestep + 1
    logger.info(f'最小帧数要求: {min_frames} (max_framestep={args.max_framestep})')

    broken: List[str] = []
    summary_lines = []

    # 常规 tier
    for tier in TIERS:
        rows = scan_tier(root, tier, tier, min_frames, args.allow_gaps, broken)
        write_list(out_dir / f'{tier}_train.txt', rows)
        summary_lines.append(f'{tier:>8}: {len(rows):>6} scenes  '
                             f'{sum(r["n"] for r in rows):>9} 帧')

    # teacher
    rows = scan_teacher(root, min_frames, args.allow_gaps, args.pair_ratio, broken)
    write_list(out_dir / 'teacher_train.txt', rows)
    fps_dist = {}
    for r in rows:
        fps_dist[r['fps']] = fps_dist.get(r['fps'], 0) + 1
    summary_lines.append(f'{"teacher":>8}: {len(rows):>6} scenes  '
                         f'{sum(r["n"] for r in rows):>9} 帧  '
                         f'fps分布={dict(sorted(fps_dist.items()))}')

    # val (预划分验证集, tier标签记为 val)
    rows = scan_tier(root, VAL_DIR, 'val', min_frames, args.allow_gaps, broken)
    write_list(out_dir / 'val.txt', rows)
    summary_lines.append(f'{"val":>8}: {len(rows):>6} scenes  '
                         f'{sum(r["n"] for r in rows):>9} 帧')

    # broken 记录
    if broken:
        bp = out_dir / 'broken_scenes.txt'
        bp.write_text('\n'.join(broken) + '\n')
        summary_lines.append(f'剔除 {len(broken)} 个scene → broken_scenes.txt')
        logger.warning(f'剔除 {len(broken)} 个scene, 明细: {bp}')

    summary = '\n'.join(summary_lines)
    (out_dir / 'summary.txt').write_text(summary + '\n')
    logger.info('\n════════ 清单统计 ════════\n' + summary)


if __name__ == '__main__':
    main()