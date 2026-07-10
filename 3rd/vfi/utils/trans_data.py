'''
EXR/TIF → PNG/JPG 训练集转换脚本
==================================
从 root 下扫描所有名字包含 "source" 的目录, 每个 source 目录下包含多个 scene,
每个 scene 有 Left / Right 两个视角目录, 其中的 image 子目录存放 4K exr/tif 帧。
将帧载入并转存为 png 或 jpg, 按 scene+视角 展平为训练集目录结构。

输入结构:
    root/
    └── .../xxx_source/            ← 目录名包含 "source" 即可, 任意深度
        ├── sceneA/
        │   ├── Left/
        │   │   └── image/
        │   │       ├── 0001.exr   (或 .tif)
        │   │       └── ...
        │   └── Right/
        │       └── image/*.exr
        └── sceneB/ ...

输出结构 (与训练 pipeline 的 scenes/<video>/scene_0000 布局对齐):
    output/
    ├── sceneA_Left/
    │   ├── 000000.png (或 .jpg)   ← 帧号按顺序重编
    │   └── ...
    ├── sceneA_Right/
    └── sceneB_Left/ ...

输入格式与传递函数:
    .exr        视为线性HDR, 应用 --transfer (srgb/gamma22/linear)
    .tif/.tiff  视为已在显示域 (uint16/65535 或 uint8/255 归一化后直接量化),
                不再套传递函数; 若你的 tif 实际是线性数据, 加 --tif_as_linear
                让其走与 exr 相同的 transfer 链

输出格式 (存储体积):
    --format png (默认)  无损; --png16 输出16bit; --png_compression 调压缩等级
    --format jpg         最小存储方案: 体积约为 png 的 1/5~1/10,
                         --jpg_quality 默认 95 (对VFI训练影响可忽略)

用法:
    python trans_data.py --root /Users/qhong/Documents/nas2/BE3D/6_processing \
        --output /Volumes/hongqing/vfi --workers 8 --format jpg

依赖: pip install opencv-python-headless numpy tqdm
      (有 file_utils 时优先用 mvread 读 exr, 无则回退 cv2; tif 始终走 cv2)
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

IMG_EXTS = {'.exr', '.tif', '.tiff'}

# ── exr 读取: 优先 file_utils.read, 回退 cv2 ─────────────────────────────────
cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cur_path + '/../algo')

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2  # noqa: E402

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
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


def read_tif(path: Path) -> Optional[np.ndarray]:
    """tif 读取 (cv2), 按位深归一化到 [0,1] float32 RGB。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)             # float tif 原样
    if img.ndim == 2:
        img = img[..., None].repeat(3, axis=2)
    return np.ascontiguousarray(img[..., :3][..., ::-1])       # BGR→RGB


def read_frame(path: Path):
    """返回 (float32 RGB 图像, 是否为线性域exr)。"""
    if path.suffix.lower() == '.exr':
        return read_exr(path), True
    return read_tif(path), False


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


# ── DCI XYZ 支持 ─────────────────────────────────────────────────────────────
# 线性 XYZ → 线性 Rec709 (D65) 标准矩阵
_XYZ_TO_REC709 = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108,  0.0415560],
    [0.0556434, -0.2040259,  1.0572252],
], dtype=np.float32)


def xyz_to_linear_rec709(img: np.ndarray, decode_gamma: float) -> np.ndarray:
    """DCI X'Y'Z' → 线性 Rec709。
    decode_gamma > 0: 先做 ^gamma 反编码 (DCDM 标准为 2.6, 整数tif容器的典型情况);
    decode_gamma = 0: 输入已是线性 XYZ (如 exr 容器), 跳过反编码只做色域矩阵。
    色域外产生的负值截为 0。"""
    if decode_gamma > 0:
        img = np.power(np.clip(img, 0.0, 1.0), decode_gamma)
    shp = img.shape
    out = img.reshape(-1, 3) @ _XYZ_TO_REC709.T
    return np.clip(out.reshape(shp), 0.0, None)


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


def _list_frames(d: Path) -> List[Path]:
    return sorted((p for p in d.iterdir()
                   if p.suffix.lower() in IMG_EXTS
                   and not p.name.startswith('.')),
                  key=extract_number)


def _path_tag(root: Path, d: Path) -> str:
    """由 root→d 的相对路径生成序列名 (路径唯一 → 名字唯一)。
    例: scene_name/2018X1125/awg3_loc3/left/source → scene_name_2018X1125_awg3_loc3_left"""
    rel = d.relative_to(root)
    parts = [re.sub(r'[^\w]', '_', p) for p in rel.parts
             if 'source' not in p.lower()]        # 去掉 source 这级, 无信息量
    return '_'.join(parts) if parts else re.sub(r'[^\w]', '_', d.name)


def collect_tasks(root: Path) -> List[dict]:
    """
    返回任务列表, 每项: {'frames': [...], 'out_name': ...}
    支持两种布局:
      A) source 目录本身直接包含帧文件 (新):
         .../scene_name/2018X1125/awg3_loc3/left/source/*.tif
         → out_name 由 root 到 source 的相对路径派生 (天然唯一)
      B) source 目录下是 scene/Left|Right/image/ (旧):
         .../xxx_source/sceneA/Left/image/*.exr
         → out_name = scene_side, 同名时加 source 目录名前缀消歧
    """
    tasks = []
    name_count: dict = {}

    source_dirs = find_source_dirs(root)
    if not source_dirs:
        logger.error(f'在 {root} 下未找到名字包含 "source" 的目录')
        return tasks
    logger.info(f'发现 {len(source_dirs)} 个 source 目录')

    for src in source_dirs:
        # ── 布局A: source 目录直接含帧 ────────────────────────────────────
        direct_frames = _list_frames(src)
        if direct_frames:
            tasks.append({'frames': direct_frames,
                          'out_name': _path_tag(root, src),
                          'source': src, 'layout': 'A'})
            continue

        # ── 布局B: source/scene/Left|Right/image/ (视角目录大小写不敏感) ──
        for scene in sorted(p for p in src.iterdir() if p.is_dir()):
            for side_dir in sorted(p for p in scene.iterdir() if p.is_dir()):
                if side_dir.name.lower() not in ('left', 'right'):
                    continue
                img_dir = side_dir / 'image'
                search_dir = img_dir if img_dir.is_dir() else side_dir
                frames = _list_frames(search_dir)
                if not frames:
                    continue
                out_name = f'{scene.name}_{side_dir.name}'
                name_count[out_name] = name_count.get(out_name, 0) + 1
                tasks.append({'frames': frames, 'out_name': out_name,
                              'source': src, 'layout': 'B'})

    # 布局B 同名消歧: 前缀加 source 目录名 (布局A 路径派生名天然唯一, 不参与)
    dup_names = {k for k, v in name_count.items() if v > 1}
    for t in tasks:
        if t['layout'] == 'B' and t['out_name'] in dup_names:
            src_tag = re.sub(r'[^\w]', '_', t['source'].name)
            t['out_name'] = f"{src_tag}_{t['out_name']}"

    logger.info(f'共 {len(tasks)} 个序列, '
                f'{sum(len(t["frames"]) for t in tasks)} 帧')
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# 处理
# ─────────────────────────────────────────────────────────────────────────────

def convert_one(src_path: Path, out_path: Path, args) -> bool:
    img, is_linear = read_frame(src_path)
    if img is None:
        logger.warning(f'读取失败: {src_path}')
        return False

    if args.max_w > 0 or args.max_h > 0:
        h, w = img.shape[:2]
        max_w = args.max_w if args.max_w > 0 else w
        max_h = args.max_h if args.max_h > 0 else h
        scale = min(max_w / w, max_h / h)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

    # XYZ 源: 先解码到线性 Rec709, 之后一律按线性源走 transfer 链
    if args.xyz:
        # tif整数容器默认带 gamma2.6 编码; exr 容器视为已线性 (跳过反gamma)
        decode_gamma = 0.0 if is_linear else args.xyz_gamma
        img = xyz_to_linear_rec709(img, decode_gamma)
        is_linear = True

    # exr(线性) 走 transfer 链; tif 默认已在显示域, 仅截断, 除非 --tif_as_linear
    if is_linear or args.tif_as_linear:
        img = apply_transfer(img, args.transfer, args.exposure)
    else:
        img = np.clip(img, 0.0, 1.0)

    # 量化 + 编码
    if args.format == 'jpg':
        arr = (img * 255.0 + 0.5).astype(np.uint8)
        params = [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality]
    else:
        if args.png16:
            arr = (img * 65535.0 + 0.5).astype(np.uint16)
        else:
            arr = (img * 255.0 + 0.5).astype(np.uint8)
        params = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]

    ok = cv2.imwrite(str(out_path), arr[..., ::-1], params)   # RGB→BGR
    if not ok:
        logger.warning(f'写入失败: {out_path}')
    return ok


def process_sequence(task: dict, out_root: Path, args) -> int:
    out_dir = out_root / task['out_name']
    done_marker = out_dir / '.done'
    if done_marker.exists() and not args.overwrite:
        return -1                                          # 断点续跑: 跳过

    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.format                                      # png / jpg
    jobs = [(src, out_dir / f'{i:06d}.{ext}')              # 帧号顺序重编
            for i, src in enumerate(task['frames'])]

    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert_one, s, o, args) for s, o in jobs]
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
    p = argparse.ArgumentParser(description='扫描 source 目录, exr/tif → png/jpg 训练集转换')
    p.add_argument('--root', type=str, required=True,
                   help='扫描根目录 (递归查找名字包含 "source" 的目录)')
    p.add_argument('--output', type=str, required=True,
                   help='训练集输出根目录 (每个 scene×视角 一个子目录)')
    p.add_argument('--format', type=str, default='png', choices=['png', 'jpg'],
                   help='输出格式; jpg 为最小存储方案, 体积约 png 的 1/5~1/10 (默认 png)')
    p.add_argument('--jpg_quality', type=int, default=95,
                   help='jpg 质量 1~100 (默认 95, 对VFI训练影响可忽略)')
    p.add_argument('--transfer', type=str, default='srgb',
                   choices=['srgb', 'gamma22', 'linear'],
                   help='线性源 → 显示域的传递函数 (默认 srgb, 仅作用于 exr 或 --tif_as_linear)')
    p.add_argument('--exposure', type=float, default=1.0,
                   help='转换前的曝光倍数 (默认 1.0, 仅作用于线性源)')
    p.add_argument('--tif_as_linear', action='store_true',
                   help='将 tif 视为线性数据, 走与 exr 相同的 transfer 链 '
                        '(默认 tif 视为已在显示域, 仅截断量化)')
    p.add_argument('--xyz', action='store_true',
                   help='输入为 DCI XYZ 色彩空间 (电影DCP/DCDM源): 反gamma解码 + '
                        'XYZ→Rec709 色域矩阵后再走 transfer 链')
    p.add_argument('--xyz_gamma', type=float, default=2.6,
                   help='XYZ 反编码 gamma (DCDM标准 2.6); 仅作用于整数tif容器, '
                        'exr 容器视为已线性自动跳过 (默认 2.6)')
    p.add_argument('--png16', action='store_true',
                   help='输出 16bit png (默认 8bit; jpg 格式下无效)')
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