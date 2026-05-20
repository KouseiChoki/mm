"""
VFI Training Data Preparation Script (两阶段低内存版)
======================================================

阶段一  decode  逐帧解码 + resize → 写盘到 raw_frames/<video>/
阶段二  classify  读小缩略图 → 场景切换检测 → 用 rename/symlink 按场景整理

输出结构:
    output_dir/
    ├── raw_frames/          ← 阶段一产物（可删）
    │   └── video_name/
    │       ├── 000000.png
    │       └── ...
    └── scenes/              ← 阶段二产物（训练用）
        └── video_name/
            ├── scene_0000/
            │   ├── 000000.png → (symlink or copy)
            │   └── ...
            └── scene_0001/

依赖:
    pip install opencv-python-headless tqdm
    brew install ffmpeg          # 处理 10bit / y4m

用法:
    # 完整两阶段
    python prepare_vfi_data.py --input_dir /raw --output_dir /out --width 448 --height 256

    # 只跑阶段一（解码）
    python prepare_vfi_data.py ... --stage decode

    # 只跑阶段二（分类，阶段一已完成）
    python prepare_vfi_data.py ... --stage classify
"""
# python prepare_vfi_train.py --input_dir /Volumes/optflow_ssd/Youtube/y4m --output_dir /Volumes/optflow_ssd/vfi_train_data_y4m
import re
import cv2
import json
import shutil
import argparse
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".mp4", ".y4m", ".avi", ".mov", ".mkv", ".webm"}


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

def safe_name(stem: str) -> str:
    return re.sub(r"[^\w]", "_", stem)


def find_videos(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
        and p.is_file()
        and not p.name.startswith(".")
        and not p.name.startswith("._")
    )


# ─────────────────────────────────────────────────────────────────────────────
# 阶段一：解码 + resize → 逐帧写盘
# ─────────────────────────────────────────────────────────────────────────────

def _probe_video(video_path: Path) -> Tuple[int, int, float]:
    """
    返回 (orig_width, orig_height, fps)。
    失败时 width/height 为 0。
    """
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(video_path)],
            stderr=subprocess.DEVNULL,
        )
        info = json.loads(out)
        for s in info.get("streams", []):
            w = s.get("width", 0)
            h = s.get("height", 0)
            r = s.get("r_frame_rate", "")
            fps = 24.0
            if "/" in r:
                a, b = r.split("/")
                fps = float(a) / float(b) if float(b) else 24.0
            if w and h:
                return w, h, fps
    except Exception:
        pass
    return 0, 0, 24.0


def _opencv_can_decode(video_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False
    ret, _ = cap.read()
    cap.release()
    return ret


def decode_video_ffmpeg_pipe(
    video_path: Path,
    out_dir: Path,
    width: int,
    height: int,
    fmt: str,
    jpg_quality: int,
) -> int:
    """ffmpeg pipe 逐帧解码写盘，内存中只保留 1 帧。返回写入帧数。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = f".{fmt}"
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality] if fmt == "jpg" else []

    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(video_path),
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", f"scale={width}:{height}:flags=lanczos",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = width * height * 3
    idx = 0
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        cv2.imwrite(str(out_dir / f"{idx:06d}{ext}"), frame, encode_params)
        idx += 1
    proc.stdout.close()
    proc.wait()
    return idx


def decode_video_opencv_pipe(
    video_path: Path,
    out_dir: Path,
    width: int,
    height: int,
    fmt: str,
    jpg_quality: int,
) -> int:
    """OpenCV 逐帧读取写盘，内存中只保留 1 帧。返回写入帧数。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = f".{fmt}"
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality] if fmt == "jpg" else []

    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(out_dir / f"{idx:06d}{ext}"), resized, encode_params)
        idx += 1
    cap.release()
    return idx


def stage_decode(
    input_dir: Path,
    raw_dir: Path,
    width: int,
    height: int,
    fmt: str,
    jpg_quality: int,
) -> None:
    """
    阶段一：遍历所有视频，逐帧解码+resize 写入 raw_dir/<video_name>/<frame>.png
    内存峰值 ≈ 1 帧大小。
    """
    videos = find_videos(input_dir)
    if not videos:
        logger.error(f"未找到视频: {input_dir}")
        return

    logger.info(f"[阶段一] {len(videos)} 个视频 → {raw_dir}  ({width}×{height})")

    for video_path in tqdm(videos, desc="解码", unit="video"):
        name    = safe_name(video_path.stem)
        out_dir = raw_dir / name

        # 跳过已完成的（断点续跑）
        done_marker = out_dir / ".done"
        if done_marker.exists():
            logger.info(f"  跳过（已完成）: {video_path.name}")
            continue

        logger.info(f"  解码: {video_path.name}")

        use_ffmpeg = not _opencv_can_decode(video_path)
        if use_ffmpeg:
            logger.info(f"    → ffmpeg pipe（10bit/特殊格式）")

        try:
            if use_ffmpeg:
                n = decode_video_ffmpeg_pipe(video_path, out_dir, width, height, fmt, jpg_quality)
            else:
                n = decode_video_opencv_pipe(video_path, out_dir, width, height, fmt, jpg_quality)
        except Exception as e:
            logger.error(f"  失败: {e}")
            continue

        done_marker.touch()
        logger.info(f"  ✓ {n} 帧 → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 阶段二：读小图做场景检测 → 按场景整理
# ─────────────────────────────────────────────────────────────────────────────

def load_thumbnail(path: Path, thumb_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """读图并缩到小尺寸用于场景检测，节省内存和时间。"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.resize(img, thumb_size, interpolation=cv2.INTER_AREA)


def detect_scene_boundaries(
    frame_paths: List[Path],
    thumb_size: Tuple[int, int],
    threshold: float,
) -> List[int]:
    """
    逐帧读缩略图计算帧差，返回场景边界帧号列表（含首 0 和尾 len）。
    内存中最多同时持有 2 帧缩略图。
    """
    boundaries = [0]
    prev_gray: Optional[np.ndarray] = None

    for i, path in enumerate(tqdm(frame_paths, desc="    场景检测", leave=False, unit="f")):
        thumb = load_thumbnail(path, thumb_size)
        if thumb is None:
            continue
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if prev_gray is not None:
            diff = np.mean(np.abs(gray - prev_gray))
            if diff > threshold:
                boundaries.append(i)

        prev_gray = gray

    boundaries.append(len(frame_paths))
    return boundaries


def organize_scene(
    src_paths: List[Path],
    scene_dir: Path,
    use_symlink: bool,
) -> int:
    """将一组帧文件移动/软链接到 scene_dir，帧号重新从 0 编号。"""
    scene_dir.mkdir(parents=True, exist_ok=True)
    ext = src_paths[0].suffix if src_paths else ".png"
    for local_idx, src in enumerate(src_paths):
        dst = scene_dir / f"{local_idx:06d}{ext}"
        if use_symlink:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
        else:
            shutil.move(str(src), str(dst))
    return len(src_paths)


def stage_classify(
    raw_dir: Path,
    scene_dir_root: Path,
    thumb_size: Tuple[int, int],
    threshold: float,
    min_scene_frames: int,
    use_symlink: bool,
) -> None:
    """
    阶段二：遍历 raw_dir 下每个视频目录，
    读小图做场景检测，将帧按场景整理到 scene_dir_root。
    """
    video_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if not video_dirs:
        logger.error(f"raw_dir 为空: {raw_dir}")
        return

    logger.info(
        f"[阶段二] {len(video_dirs)} 个视频目录 → {scene_dir_root}\n"
        f"  缩略图尺寸={thumb_size}  阈值={threshold}  "
        f"最少帧={min_scene_frames}  symlink={use_symlink}"
    )

    total_scenes = 0
    total_frames = 0

    for vdir in tqdm(video_dirs, desc="分类", unit="video"):
        frame_paths = sorted(
            p for p in vdir.iterdir()
            if p.suffix.lower() in {".png", ".jpg"} and not p.name.startswith(".")
        )
        if len(frame_paths) < 3:
            logger.warning(f"  {vdir.name}: 帧数不足，跳过")
            continue

        logger.info(f"  {vdir.name}: {len(frame_paths)} 帧")

        boundaries = detect_scene_boundaries(frame_paths, thumb_size, threshold)
        n_raw_scenes = len(boundaries) - 1
        scene_out    = scene_dir_root / vdir.name
        scene_counter = 0

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            chunk = frame_paths[start:end]
            if len(chunk) < min_scene_frames:
                logger.debug(f"    场景 [{start},{end}) 仅 {len(chunk)} 帧，丢弃")
                continue
            dst = scene_out / f"scene_{scene_counter:04d}"
            organize_scene(chunk, dst, use_symlink)
            total_frames += len(chunk)
            scene_counter += 1

        logger.info(f"  ✓ 保留 {scene_counter}/{n_raw_scenes} 个场景")
        total_scenes += scene_counter

    logger.info(
        f"\n[阶段二完成] 共 {total_scenes} 个场景，{total_frames} 帧\n"
        f"输出: {scene_dir_root}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    output_dir   = Path(args.output_dir)
    raw_dir      = output_dir / "raw_frames"
    scene_dir    = output_dir / "scenes"
    thumb_size   = (args.thumb_width, args.thumb_height)

    if args.stage in ("decode", "both"):
        stage_decode(
            input_dir   = Path(args.input_dir),
            raw_dir     = raw_dir,
            width       = args.width,
            height      = args.height,
            fmt         = args.format,
            jpg_quality = args.jpg_quality,
        )

    if args.stage in ("classify", "both"):
        stage_classify(
            raw_dir          = raw_dir,
            scene_dir_root   = scene_dir,
            thumb_size       = thumb_size,
            threshold        = args.threshold,
            min_scene_frames = args.min_scene_frames,
            use_symlink      = args.symlink,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VFI 两阶段数据预处理：①解码写盘  ②小图场景检测分类"
    )
    p.add_argument("--input_dir",        type=str, required=True,
                   help="原始视频根目录")
    p.add_argument("--output_dir",       type=str, required=True,
                   help="输出根目录（含 raw_frames/ 和 scenes/ 两个子目录）")
    p.add_argument("--stage",            type=str, default="both",
                   choices=["decode", "classify", "both"],
                   help="执行阶段 (默认 both)")

    g1 = p.add_argument_group("阶段一：解码参数")
    g1.add_argument("--width",           type=int, default=1920,  help="目标宽度")
    g1.add_argument("--height",          type=int, default=1080,  help="目标高度")
    g1.add_argument("--format",          type=str, default="png", choices=["png", "jpg"],
                    help="帧保存格式")
    g1.add_argument("--jpg_quality",     type=int, default=100,   help="jpg 质量 1~100")

    g2 = p.add_argument_group("阶段二：场景检测参数")
    g2.add_argument("--thumb_width",     type=int,   default=64,
                    help="场景检测用缩略图宽度（越小越快，默认 64）")
    g2.add_argument("--thumb_height",    type=int,   default=36,
                    help="场景检测用缩略图高度（默认 36）")
    g2.add_argument("--threshold",       type=float, default=8.0,
                    help="帧差阈值（小图上的均值差，默认 8.0）")
    g2.add_argument("--min_scene_frames",type=int,   default=5,
                    help="场景最少帧数，低于此值丢弃（默认 5）")
    g2.add_argument("--symlink",         action="store_true",
                    help="用软链接代替移动文件（保留 raw_frames 原件）")

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
