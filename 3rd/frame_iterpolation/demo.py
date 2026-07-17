import os
import sys
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.nn import functional as F
from collections import defaultdict
import glob
import re
import warnings

from file_utils import read, write

warnings.filterwarnings("ignore")

# ── 参数 ─────────────────────────────────────────────────────────────
# python demo.py --img /home/zhenying/qhong/data/ssd/AIFRC --output /home/zhenying/qhong/data/ssd/AIFRC_VFI/RIFE
parser = argparse.ArgumentParser(description='RIFE 图片文件夹插帧（递归）')
parser.add_argument('--img',      dest='img',      type=str, required=True,  help='输入根目录')
parser.add_argument('--output',   dest='output',   type=str, required=True,  help='输出根目录（保留层级）')
parser.add_argument('--modelDir', dest='modelDir', type=str,
                    default='/home/zhenying/qhong/repo/mm/3rd/frame_iterpolation/train_log',
                    help='模型目录')
parser.add_argument('--fp16',  dest='fp16',  action='store_true', help='fp16 推理')
parser.add_argument('--UHD',   dest='UHD',   action='store_true', help='4K 模式')
parser.add_argument('--scale', dest='scale', type=float, default=1.0)
parser.add_argument('--exp',   dest='exp',   type=int,   default=1)
args = parser.parse_args()

if args.UHD and args.scale == 1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]

# ── 设备 ─────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = True
    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

# ── 模型加载 ──────────────────────────────────────────────────────────
try:
    try:
        try:
            from model.RIFE_HDv2 import Model
            model = Model(); model.load_model(args.modelDir, -1)
            print("Loaded v2.x HD model.")
        except:
            from train_log.RIFE_HDv3 import Model
            model = Model(); model.load_model(args.modelDir, -1)
            print("Loaded v3.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model(); model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")
except:
    from model.RIFE import Model
    model = Model(); model.load_model(args.modelDir, -1)
    print("Loaded ArXiv-RIFE model")

model.eval()
model.device = device

# ── resize 上限 ───────────────────────────────────────────────────────
MAX_W, MAX_H = 1920, 1080

def cap_size(w: int, h: int):
    scale = min(MAX_W / w, MAX_H / h)
    if scale < 1.0:
        w, h = int(w * scale), int(h * scale)
    return w, h

def resize_if_needed(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    cw, ch = cap_size(w, h)
    if (cw, ch) != (w, h):
        img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)
    return img

def to_float(img: np.ndarray) -> np.ndarray:
    """统一转 float32 [0,1]。"""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)

# ── 工具函数 ──────────────────────────────────────────────────────────
def extract_number(x):
    nums = re.findall(r'(\d+)', os.path.basename(x))
    return int(nums[-1]) if nums else -1

def collect_pairs(root: str):
    """递归收集所有子目录中的相邻图片对，返回 list of (path0, path1)。"""
    exts = ["*.png", "*.exr","*.tif"]
    seq_dict = defaultdict(list)
    for ext in exts:
        for f in glob.glob(os.path.join(root, "**", ext), recursive=True):
            seq_dict[os.path.dirname(f)].append(f)

    pairs = []
    for folder, files in seq_dict.items():
        files = sorted(files, key=extract_number)
        for i in range(len(files) - 1):
            pairs.append((files[i], files[i + 1]))
    return pairs

def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """float32 HxWx3 [0,1] → tensor (1,3,H,W) on device。"""
    return torch.from_numpy(
        img.transpose(2, 0, 1)
    ).to(device, non_blocking=True).unsqueeze(0).float()

def make_inference(I0, I1, n):
    middle = model.inference(I0, I1, args.scale)
    if n == 1:
        return [middle]
    first_half  = make_inference(I0, middle, n=n // 2)
    second_half = make_inference(middle, I1, n=n // 2)
    if n % 2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]

def tensor_to_numpy(t: torch.Tensor, h: int, w: int) -> np.ndarray:
    """tensor (1,C,H,W) [0,1] → float32 HxWx3 [0,1]，裁回原始 h,w。"""
    arr = t[0].cpu().numpy().transpose(1, 2, 0)[:h, :w]
    return arr.astype(np.float32)

def save_frame(arr: np.ndarray, path: str, is_exr: bool):
    """
    arr: float32 HxWx3 [0,1]
    - exr → write 直接保存（HDR 精度）
    - png → 转 uint8 再 write
    """
    if is_exr:
        write(path, arr)
    else:
        write(path, (arr * 255).clip(0, 255).astype(np.uint8))

# ── 主流程 ────────────────────────────────────────────────────────────
in_root  = args.img
out_root = args.output

pairs = collect_pairs(in_root)
if not pairs:
    print("Error: 未找到任何图片对"); exit(1)
print(f"共找到 {len(pairs)} 对图片，插帧倍数 2^{args.exp}={2**args.exp}x")

for pair in tqdm(pairs, desc="插帧进度"):
    path0, path1 = pair
    ext    = os.path.splitext(path0)[1]          # '.png' or '.exr'
    is_exr = ext.lower() == '.exr' or ext.lower() == '.tif'

    # 读取 & 预处理
    I0_raw = resize_if_needed(to_float(read(path0, type='image')))
    I1_raw = resize_if_needed(to_float(read(path1, type='image')))

    h, w = I0_raw.shape[:2]

    # padding（32 对齐）
    tmp = max(32, int(32 / args.scale))
    ph  = ((h - 1) // tmp + 1) * tmp
    pw  = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    def pad(t):
        return F.pad(t, padding).half() if args.fp16 else F.pad(t, padding)

    I0 = pad(img_to_tensor(I0_raw))
    I1 = pad(img_to_tensor(I1_raw))

    # 插帧
    output = make_inference(I0, I1, 2 ** args.exp - 1) if args.exp else []

    # 输出路径（保留层级）
    rel_folder  = os.path.relpath(os.path.dirname(path0), in_root)
    save_folder = os.path.join(out_root, rel_folder)
    os.makedirs(save_folder, exist_ok=True)

    idx0    = extract_number(path0) * 2
    idx1    = extract_number(path1) * 2
    mid_idx = (idx0 + idx1) // 2                # sf=2 时只有一个中间帧

    # 写插值帧（sf=2 → 1 帧；sf=4 → 3 帧，按顺序编号）
    n_mid = len(output)
    for k, mid_t in enumerate(output):
        if n_mid == 1:
            cur_idx = mid_idx
        else:
            # 均匀分布在 idx0~idx1 之间
            cur_idx = idx0 + (idx1 - idx0) * (k + 1) // (n_mid + 1)
        mid_arr = tensor_to_numpy(mid_t, h, w)
        save_frame(mid_arr, os.path.join(save_folder, f"{cur_idx:06d}{ext}"), is_exr)

    # 写原始帧（不重复写）
    for f, new_idx in [(path0, idx0), (path1, idx1)]:
        out_path = os.path.join(save_folder, f"{new_idx:06d}{ext}")
        if not os.path.exists(out_path):
            arr = resize_if_needed(to_float(read(f, type='image')))
            save_frame(arr, out_path, is_exr)

print(f"\n完成！结果保存于：{out_root}")