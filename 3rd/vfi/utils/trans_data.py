'''
EXR → PNG 训练集转换脚本
==========================
从 root 下扫描所有名字包含 "source" 的目录, 每个 source 目录下包含多个 scene,
每个 scene 有 Left / Right 两个视角目录, 其中的 image 子目录存放 4K exr 帧。
将 exr 载入并转存为 png, 按 scene+视角 展平为训练集目录结构。

输入结构:
    root/
    └── .../xxx_source/            ← 目录名包含 "source" 即可, 任意深度
        ├── sceneA/
        │   ├── Left/
        │   │   └── image/
        │   │       ├── 0001.exr
        │   │       └── ...
        │   └── Right/
        │       └── image/*.exr
        └── sceneB/ ...

输出结构 (与训练 pipeline 的 scenes/<video>/scene_0000 布局对齐):
    output/
    ├── sceneA_Left/
    │   ├── 000000.png     ← 帧号按顺序重编
    │   └── ...
    ├── sceneA_Right/
    └── sceneB_Left/ ...

exr(线性HDR) → png 需要传递函数处理, 提供:
    --transfer  srgb (默认) / gamma22 / linear
    --exposure  曝光倍数 (转换前先乘, 默认 1.0)
    --png16     输出 16bit png (默认 8bit)
    --max_w/--max_h  可选等比缩小 (默认 0 = 保留 4K 原尺寸)

用法:
    python trans_data.py --root /Users/qhong/Documents/nas2/BE3D/6_processing --output /Volumes/hongqing/vfi --workers 8

依赖: pip install opencv-python-headless numpy tqdm
      (有 file_utils 时优先用 mvread 读 exr, 无则回退 cv2)
'''
import os
import sys
import re
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── exr 读取: 优先 file_utils.read, 回退 cv2 ─────────────────────────────────
cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cur_path + '/../algo')

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2  # noqa: E402

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/..')
    from file_utils import read as fu_read      # noqa: E402

    def read_exr(path: Path) -> Optional[np.ndarray]:
        """返回 float32 RGB, 线性域。"""
        res = fu_read(str(path), type='hdr')     # mvread, [...,:3], RGB
        return None if res is None else res.astype(np.float32)

    logger.info('使用 file_utils.read 读取 exr')
except ImportError:
    def read_exr(path: Path) -> Optional[np.ndarray]:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        img = img.astype(np.float32)
        if img.ndim == 2:
            img = img[..., None].repeat(3, axis=2)
        return np.ascontiguousarray(img[..., :3][..., ::-1])   # BGR→RGB

    logger.warning('未找到 file_utils, 使用 cv2 回退读取 exr')


# ─────────────────────────────────────────────────────────────────────────────
# 转换
# ─────────────────────────────────────────────────────────────────────────────

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """标准 sRGB OETF (分段), 输入输出均为 [0,1] float32。"""
    a = 0.055
    lo = x * 12.92
    hi = (1 + a) * np.power(np.clip(x, 1e-8, None), 1 / 2.4) - a
    return np.where(x <= 0.0031308, lo, hi)


def apply_transfer(img: np.ndarray, transfer: str, exposure: float) -> np.ndarray:
    """线性 HDR → 显示域 [0,1]。"""
    img = img * exposure
    img = np.clip(img, 0.0, 1.0)
    if transfer == 'srgb':
        img = linear_to_srgb(img)
    elif transfer == 'gamma22':
        img = np.power(img, 1.0 / 2.2)
    # 'linear': 直接截断, 不做传递函数
    return img


def to_png_array(img01: np.ndarray, png16: bool) -> np.ndarray:
    if png16:
        return (img01 * 65535.0 + 0.5).astype(np.uint16)
    return (img01 * 255.0 + 0.5).astype(np.uint8)


def extract_number(p: Path) -> int:
    nums = re.findall(r'(\d+)', p.stem)
    return int(nums[-1]) if nums else -1


# ─────────────────────────────────────────────────────────────────────────────
# 扫描
# ─────────────────────────────────────────────────────────────────────────────

def find_source_dirs(root: Path) -> List[Path]:
    """任意深度下, 目录名包含 'source' (不区分大小写) 的目录。
    命中即剪枝: 不再深入其内部, 天然只保留最外层, 无需事后去重。"""
    out = []
    for dirpath, dirnames, _ in os.walk(root):
        matched = [d for d in dirnames if 'source' in d.lower()]
        for d in matched:
            print(f'find : {Path(dirpath) / d}')
            out.append(Path(dirpath) / d)
        # 原地剔除已命中的目录 → os.walk 不再进入它们内部
        dirnames[:] = [d for d in dirnames if 'source' not in d.lower()]
    return sorted(out)


def collect_tasks(root: Path) -> List[dict]:
    """
    返回任务列表, 每项:
      {'scene': sceneA, 'side': Left, 'exrs': [...], 'out_name': sceneA_Left}
    out_name 冲突时 (不同 source 下同名 scene) 加上 source 目录名前缀消歧。
    """
    tasks = []
    name_count: dict = {}

    source_dirs = find_source_dirs(root)
    if not source_dirs:
        logger.error(f'在 {root} 下未找到名字包含 "source" 的目录')
        return tasks
    logger.info(f'发现 {len(source_dirs)} 个 source 目录')

    for src in source_dirs:
        for scene in sorted(p for p in src.iterdir() if p.is_dir()):
            for side in ('Left', 'Right'):
                img_dir = scene / side / 'image'
                if not img_dir.is_dir():
                    continue
                exrs = sorted((p for p in img_dir.iterdir()
                               if p.suffix.lower() == '.exr'
                               and not p.name.startswith('.')),
                              key=extract_number)
                if not exrs:
                    continue
                out_name = f'{scene.name}_{side}'
                name_count[out_name] = name_count.get(out_name, 0) + 1
                tasks.append({'scene': scene.name, 'side': side,
                              'source': src, 'exrs': exrs,
                              'out_name': out_name})

    # 同名消歧: 前缀加 source 目录名
    dup_names = {k for k, v in name_count.items() if v > 1}
    for t in tasks:
        if t['out_name'] in dup_names:
            src_tag = re.sub(r'[^\w]', '_', t['source'].name)
            t['out_name'] = f"{src_tag}_{t['out_name']}"

    logger.info(f'共 {len(tasks)} 个序列 (scene×视角), '
                f'{sum(len(t["exrs"]) for t in tasks)} 帧')
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# 处理
# ─────────────────────────────────────────────────────────────────────────────

def convert_one(exr_path: Path, out_path: Path, args) -> bool:
    img = read_exr(exr_path)
    if img is None:
        logger.warning(f'读取失败: {exr_path}')
        return False

    if args.max_w > 0 or args.max_h > 0:
        h, w = img.shape[:2]
        max_w = args.max_w if args.max_w > 0 else w
        max_h = args.max_h if args.max_h > 0 else h
        scale = min(max_w / w, max_h / h)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

    img = apply_transfer(img, args.transfer, args.exposure)
    arr = to_png_array(img, args.png16)
    ok = cv2.imwrite(str(out_path), arr[..., ::-1],       # RGB→BGR
                     [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression])
    if not ok:
        logger.warning(f'写入失败: {out_path}')
    return ok


def process_sequence(task: dict, out_root: Path, args) -> int:
    out_dir = out_root / task['out_name']
    done_marker = out_dir / '.done'
    if done_marker.exists() and not args.overwrite:
        return -1                                          # 断点续跑: 跳过

    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(exr, out_dir / f'{i:06d}.png')                # 帧号顺序重编
            for i, exr in enumerate(task['exrs'])]

    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert_one, e, o, args) for e, o in jobs]
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc=f"  {task['out_name']}", leave=False, unit='f'):
            if f.result():
                n_ok += 1

    if n_ok == len(jobs):
        done_marker.touch()
    else:
        logger.warning(f"{task['out_name']}: {len(jobs) - n_ok} 帧失败, 未标记完成")
    return n_ok


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='扫描 source 目录, exr → png 训练集转换')
    p.add_argument('--root', type=str, required=True,
                   help='扫描根目录 (递归查找名字包含 "source" 的目录)')
    p.add_argument('--output', type=str, required=True,
                   help='训练集输出根目录 (每个 scene×视角 一个子目录)')
    p.add_argument('--transfer', type=str, default='srgb',
                   choices=['srgb', 'gamma22', 'linear'],
                   help='线性 exr → 显示域的传递函数 (默认 srgb)')
    p.add_argument('--exposure', type=float, default=1.0,
                   help='转换前的曝光倍数 (默认 1.0)')
    p.add_argument('--png16', action='store_true',
                   help='输出 16bit png (默认 8bit)')
    p.add_argument('--png_compression', type=int, default=3,
                   help='png 压缩等级 0~9 (默认 3, 越大越小越慢)')
    p.add_argument('--max_w', type=int, default=0,
                   help='等比缩小的最大宽度, 0=保留原尺寸 (默认 0)')
    p.add_argument('--max_h', type=int, default=0,
                   help='等比缩小的最大高度, 0=保留原尺寸 (默认 0)')
    p.add_argument('--workers', type=int, default=8,
                   help='并行转换线程数 (默认 8)')
    p.add_argument('--overwrite', action='store_true',
                   help='忽略 .done 标记, 强制重转 (默认跳过已完成序列)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = collect_tasks(Path(args.root))
    if not tasks:
        return

    total, skipped = 0, 0
    for task in tqdm(tasks, desc='序列进度', unit='seq'):
        n = process_sequence(task, out_root, args)
        if n < 0:
            skipped += 1
        else:
            total += n

    logger.info(f'完成: 转换 {total} 帧, 跳过已完成序列 {skipped} 个\n输出: {out_root}')


if __name__ == '__main__':
    main()