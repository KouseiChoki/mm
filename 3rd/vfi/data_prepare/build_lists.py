'''
VFI 训练数据清单生成脚本 (build_lists.py)
==========================================
扫描数据根目录, 按 scene 从每个训练分类抽取验证集, 再与 root/val 中的
预划分数据合并生成训练/验证清单。大数据集只需扫描一次, 之后训练只读txt。

输入目录结构:
    root/
    ├── easy/        .../scene0001/*.png|jpg       ← 常规难度数据
    ├── normal/      .../scene0001/*.png|jpg
    ├── hard/        .../scene0001/*.png|jpg
    ├── opensource/  数据集名/.../scene/*.png|jpg ← Vimeo90K/X4K/SNU-FILM等
    ├── illumination/.../scene/*.png|jpg            ← 光暗/曝光变化
    ├── noise/       .../scene/*.png|jpg            ← 噪点/低信噪比
    ├── val/    电影名/scene0001/*.png|jpg        ← 预划分的验证集
    └── teacher/ ...任意深度.../12fps|24fps|48fps/{image, mv0, mv1}/
                 image/*.png|jpg  mv0/*.exr  mv1/*.exr
                 (同时包含 image+mv0+mv1 三个子目录的文件夹即为一个 teacher scene,
                  fps 从路径中的 "12fps/24fps/48fps" 目录名解析)

输出 (默认写到 <root>/lists/):
    <tier>_train.txt / <tier>_val.txt / teacher_train.txt / teacher_val.txt /
    val.txt
        每行 (tab分隔):  scene相对路径\t帧数\ttier\thas_mv\tfps
        - scene相对路径: 相对 root, 训练时 root+相对路径 定位
        - has_mv: teacher=1, 其余=0
        - fps: teacher 从路径解析 (12/24/48), 其余=0
    val.txt             所有分类抽样验证集 + root/val 预划分验证集
    broken_scenes.txt   被剔除的 scene 及原因 (帧号断号 / 帧数不足 / mv不齐)
    summary.txt         各清单统计

校验规则:
    - 帧号必须连续 (按文件名末尾数字), 断号 scene 剔除并记录 (--allow_gaps 保留)
    - 帧数 < 2*max_framestep+1 的 scene 剔除 (采不出最大间隔的三元组)
    - teacher: image/mv0/mv1 三方按帧号配对, 配对率 < 90% 的 scene 剔除并记录
    - 默认按固定 seed 抽取 0.5% scene；同一 teacher 内容的多 FPS、clean/final、
      left/right 作为一个组切分，避免跨训练/验证集泄漏

用法:
    python build_lists.py --root /home/zhenying/qhong/data/ssd/vfi_database --max_framestep 2
'''
import os
import re
import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
MV_EXTS = {'.exr'}
TIERS = (
    'easy', 'normal', 'hard',
    'opensource', 'illumination', 'noise',
)                                           # 无MV的常规分类目录
SHORT_SEQUENCE_TIERS = ('opensource', 'illumination', 'noise')
OFFICIAL_OPENSOURCE_SUBSETS = {
    # VFIMamba官方训练的两个数据源。xtrain后期使用framestep=32，
    # 因此必须具有完整的65帧窗口。
    'vimeo': ('opensource/vimeo90k/', 3),
    'xtrain': ('opensource/X4K1000FPS/', 65),
}
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


def teacher_group_key(row: dict) -> str:
    """将同一 teacher 内容的渲染/FPS/视角归为同组，防止切分泄漏。"""
    parts = Path(row['rel']).parts
    if len(parts) >= 3 and parts[0:2] == ('teacher', 'Unreal'):
        scene = re.sub(r'^(?:clean|final)', '', parts[2], flags=re.IGNORECASE)
        return '/'.join((*parts[:2], scene))

    # Spring 等无 fps 数据通常以 scene/left、scene/right 组织。
    for index, part in enumerate(parts):
        if _FPS_RE.match(part):
            return '/'.join(parts[:index])
    return str(Path(row['rel']).parent)


def split_rows(rows: List[dict], val_ratio: float, seed: int,
               group_key=None) -> Tuple[List[dict], List[dict]]:
    """按组做确定性 scene 切分；小数据集至少保留一个训练组。"""
    if not rows or val_ratio <= 0:
        return rows, []
    key_fn = group_key or (lambda row: row['rel'])
    groups = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    keys = sorted(groups)
    if len(keys) < 2:
        return rows, []

    rng = random.Random(seed)
    rng.shuffle(keys)
    val_group_count = min(
        len(keys) - 1, max(1, round(len(keys) * val_ratio)))
    val_keys = set(keys[:val_group_count])
    train_rows = [row for row in rows if key_fn(row) not in val_keys]
    val_rows = [row for row in rows if key_fn(row) in val_keys]
    return train_rows, val_rows


# ─────────────────────────────────────────────────────────────────────────────
# 常规 tier / val 扫描: 支持任意目录深度, 含图像的目录视为scene
# ─────────────────────────────────────────────────────────────────────────────

def scan_tier(root: Path, tier_dir: str, tier_label: str,
              min_frames: int, allow_gaps: bool, broken: list) -> List[dict]:
    base = root / tier_dir
    rows = []
    if not base.is_dir():
        logger.warning(f'目录不存在, 跳过: {base}')
        return rows

    # 开源数据集常有 dataset/split/sequence 等多层结构，因此递归
    # 发现直接包含帧文件的目录，不再假设固定的“电影/scene”层级。
    scene_dirs = []
    for dirpath, _, filenames in os.walk(base):
        if any(Path(name).suffix.lower() in IMG_EXTS
               for name in filenames if not name.startswith('.')):
            scene_dirs.append(Path(dirpath))
    scene_dirs.sort()
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
                 pair_ratio: float, broken: list,
                 teacher_fps: Optional[List[int]]) -> List[dict]:
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
        if teacher_fps is not None and fps not in teacher_fps:
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
    rows.sort(key=lambda row: row['rel'])
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


def write_val_list(path: Path, rows: List[dict]) -> None:
    """空分类不生成验证清单，并清除旧结果，避免 Dataset 读取空文件报错。"""
    if rows:
        write_list(path, rows)
        return
    for stale_path in (path, path.with_suffix('.frames.json')):
        if stale_path.exists():
            stale_path.unlink()
    logger.info(f'  {path.name}: 无可用验证 scene，跳过')


def read_list(path: Path) -> List[dict]:
    """读取既有scene清单及帧索引，用于无重扫地派生官方子清单。"""
    if not path.is_file():
        raise FileNotFoundError(f'清单不存在: {path}')
    index_path = path.with_suffix('.frames.json')
    if not index_path.is_file():
        raise FileNotFoundError(f'帧索引不存在: {index_path}')
    with open(index_path) as file:
        frame_index = json.load(file)
    rows = []
    with open(path) as file:
        for line in file:
            if not line.strip():
                continue
            rel, n, tier, has_mv, fps = line.rstrip('\n').split('\t')
            if rel not in frame_index:
                raise ValueError(f'{index_path}缺少scene: {rel}')
            rows.append({
                'rel': rel, 'n': int(n), 'tier': tier,
                'has_mv': int(has_mv), 'fps': int(fps),
                'frames': frame_index[rel],
            })
    return rows


def write_official_subsets(out_dir: Path, train_rows: List[dict],
                           val_rows: List[dict], summary_lines: list) -> None:
    """从同一opensource切分派生Vimeo与完整65帧X-TRAIN清单。"""
    for subset, (prefix, required_frames) in (
            OFFICIAL_OPENSOURCE_SUBSETS.items()):
        subset_train = [
            {**row, 'tier': subset} for row in train_rows
            if row['rel'].startswith(prefix) and row['n'] >= required_frames
        ]
        subset_val = [
            {**row, 'tier': subset} for row in val_rows
            if row['rel'].startswith(prefix) and row['n'] >= required_frames
        ]
        write_list(out_dir / f'{subset}_train.txt', subset_train)
        write_val_list(out_dir / f'{subset}_val.txt', subset_val)
        summary_lines.append(
            f'{subset:>12}: train={len(subset_train):>6} scenes/'
            f'{sum(r["n"] for r in subset_train):>9}帧  '
            f'val={len(subset_val):>5} scenes/'
            f'{sum(r["n"] for r in subset_val):>8}帧')


def main() -> None:
    p = argparse.ArgumentParser(description='VFI 数据清单生成 (scene级txt)')
    p.add_argument('--root', type=str, required=True,
                   help='数据根目录 (含各tier目录、val、teacher)')
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
    p.add_argument('--tiers', nargs='+', default=list(TIERS),
                   help='要生成的无MV分类目录; 默认: %(default)s')
    p.add_argument('--short_sequence_tiers', nargs='+',
                   default=list(SHORT_SEQUENCE_TIERS),
                   help='允许三帧triplet的分类; 默认: %(default)s')
    p.add_argument('--val_ratio', type=float, default=0.005,
                   help='每个训练分类按scene抽入验证集的比例 (默认 0.005)')
    p.add_argument('--val_seed', type=int, default=1234,
                   help='训练/验证切分随机种子 (默认 1234)')
    p.add_argument(
        '--official_sublists_only', action='store_true',
        help='不重新扫描/切分数据，仅从既有opensource清单派生vimeo/xtrain清单')
    args = p.parse_args()

    if not 0.0 <= args.val_ratio < 1.0:
        p.error('--val_ratio 必须在 [0, 1) 范围内')

    root = Path(args.root)
    out_dir = Path(args.output) if args.output else root / 'lists'
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.official_sublists_only:
        summary_lines = []
        train_rows = read_list(out_dir / 'opensource_train.txt')
        val_path = out_dir / 'opensource_val.txt'
        val_rows = read_list(val_path) if val_path.is_file() else []
        write_official_subsets(
            out_dir, train_rows, val_rows, summary_lines)
        logger.info('\n════════ 官方子清单统计 ════════\n'
                    + '\n'.join(summary_lines))
        return
    min_frames = 2 * args.max_framestep + 1
    logger.info(f'最小帧数要求: {min_frames} (max_framestep={args.max_framestep})')

    broken: List[str] = []
    summary_lines = []
    sampled_val_rows: List[dict] = []

    # 常规 tier
    for tier in args.tiers:
        tier_min_frames = 3 if tier in args.short_sequence_tiers else min_frames
        rows = scan_tier(root, tier, tier, tier_min_frames,
                         args.allow_gaps, broken)
        train_rows, val_rows = split_rows(
            rows, args.val_ratio,
            args.val_seed + sum(ord(char) for char in tier))
        write_list(out_dir / f'{tier}_train.txt', train_rows)
        write_val_list(out_dir / f'{tier}_val.txt', val_rows)
        sampled_val_rows.extend(val_rows)
        summary_lines.append(
            f'{tier:>12}: train={len(train_rows):>6} scenes/'
            f'{sum(r["n"] for r in train_rows):>9}帧  '
            f'val={len(val_rows):>5} scenes/'
            f'{sum(r["n"] for r in val_rows):>8}帧')

        # opensource同时保存官方VFIMamba curriculum所需的独立子清单。
        # 子清单沿用opensource已经完成的train/val切分，不重复抽样，避免泄漏。
        if tier == 'opensource':
            write_official_subsets(
                out_dir, train_rows, val_rows, summary_lines)

    # teacher
    rows = scan_teacher(root, min_frames, args.allow_gaps,
                        args.pair_ratio, broken, args.teacher_fps)
    train_rows, val_rows = split_rows(
        rows, args.val_ratio, args.val_seed + 1000003,
        group_key=teacher_group_key)
    write_list(out_dir / 'teacher_train.txt', train_rows)
    write_val_list(out_dir / 'teacher_val.txt', val_rows)
    sampled_val_rows.extend(val_rows)
    fps_dist = {}
    for r in train_rows:
        fps_dist[r['fps']] = fps_dist.get(r['fps'], 0) + 1
    summary_lines.append(
        f'{"teacher":>12}: train={len(train_rows):>6} scenes/'
        f'{sum(r["n"] for r in train_rows):>9}帧  '
        f'val={len(val_rows):>5} scenes/'
        f'{sum(r["n"] for r in val_rows):>8}帧  '
        f'train_fps={dict(sorted(fps_dist.items()))}')

    # 合并各分类抽样验证集与 root/val 中已有的预划分验证集。
    predefined_val_rows = scan_tier(
        root, VAL_DIR, 'val', min_frames, args.allow_gaps, broken)
    all_val_rows = predefined_val_rows + sampled_val_rows
    write_list(out_dir / 'val.txt', all_val_rows)
    summary_lines.append(
        f'{"all val":>12}: sampled={len(sampled_val_rows):>5} + '
        f'predefined={len(predefined_val_rows):>5} = '
        f'{len(all_val_rows):>6} scenes/'
        f'{sum(r["n"] for r in all_val_rows):>9}帧')

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
