import cv2
import sys
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
import glob
import re
import torch.nn.functional as F
import requests
sys.path.append('.')
import config as cfg
from Trainer import Model
from file_utils import read, write,mkdir


# ── Padder ───────────────────────────────────────────────────────────
class InputPadder:
    """Pads images such that dimensions are divisible by divisor"""
    def __init__(self, dims, divisor=16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2,
                     pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3],
             self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


# ── VideoWriter ───────────────────────────────────────────────────────
class SequenceVideoWriter:
    def __init__(self, fps, ext):
        self.fps = fps
        self.ext = ext
        self.fourcc = (cv2.VideoWriter_fourcc(*'mp4v') if ext == 'mp4'
                       else cv2.VideoWriter_fourcc(*'XVID'))
        self._frames = defaultdict(list)

    def add_frame(self, save_folder: str, frame_idx: int, img_f32: np.ndarray):
        bgr = to_uint8(img_f32)[..., ::-1].copy()
        self._frames[save_folder].append((frame_idx, bgr))

    def flush(self):
        for folder, frame_list in self._frames.items():
            frame_list.sort(key=lambda x: x[0])
            h, w = frame_list[0][1].shape[:2]
            video_path = os.path.join(folder, f'output.{self.ext}')
            writer = cv2.VideoWriter(video_path, self.fourcc, self.fps, (w, h))
            if not writer.isOpened():
                print(f'[VideoWriter] 无法创建视频: {video_path}')
                continue
            for _, bgr in frame_list:
                writer.write(bgr)
            writer.release()
            print(f'[VideoWriter] 已保存: {video_path}  ({len(frame_list)} 帧)')
        self._frames.clear()


# ── 工具函数 ──────────────────────────────────────────────────────────
def to_float(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def to_uint8(img_f32):
    return (img_f32 * 255).clip(0, 255).astype(np.uint8)


def extract_number(x):
    nums = re.findall(r'(\d+)', os.path.basename(x))
    return int(nums[-1]) if nums else -1


def resize_if_needed(img, max_w=7280, max_h=4320):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    if scale < 1.0:
        w, h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img

def download_file(url, destination):
    """下载文件，显示进度条"""
    response = requests.get(url, stream=True,timeout=30)
    if response.status_code != 200:
        return False
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    mkdir(os.path.dirname(destination))
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("error: connection failed, please retry later。")
    else:
        print("finished")
    return True

def check_and_download_pth_file(file_path, download_url):
    flag = False
    """检查.pth文件是否存在，如果不存在，则从URL下载"""
    if not os.path.exists(file_path):
        print(f"file {file_path} not exist, downloading...")
        flag = download_file(download_url, file_path)
    return flag


def define_model(args):
    ckpt_path = os.path.join(args.fp,args.algo)
    if '.pth' not in ckpt_path:
        ckpt_path+='.pth'
    if not os.path.isfile(ckpt_path):
        mkdir(args.fp)
        download_url = ckpt_path
        md = args.server
        md += '/vfi'
        md += '/' + args.algo + '.pth'
        print(download_url,md)
        flag = check_and_download_pth_file(download_url,md)
        if not flag:
            raise NotImplementedError(f'[MM ERROR][model]model file not exists:{args.algo},please use mmalgo to check')
        # raise NotImplementedError(f'[MM ERROR][model]model file loss!{model_name}')
    return ckpt_path


# ── 模型初始化 ────────────────────────────────────────────────────────
def build_model(args):
    if args.model == 'VFIMamba':
        cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F=32, depth=[2, 2, 2, 3, 3]
        )
    elif args.model == 'VFIMamba_KouSei':
        cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMambaKouSei'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F=24, depth=[2, 2, 2, 3, 3]
        )
    cfg.MODEL_CONFIG['MODEL_ARCH'][1]['version'] = 1 if args.old_version else 2

    model = Model(-1)
    args.fp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../checkpoints"))
    ckpt = define_model(args) 
    model.load_model(ckpt)
    model.eval()
    model.device()
    return model


# ── pair 收集 ─────────────────────────────────────────────────────────
def collect_pairs(root, out_root):
    exts = ["*.png", "*.exr", "*.tif"]
    seq_dict = defaultdict(list)
    for ext in exts:
        for f in glob.glob(os.path.join(root, "**", ext), recursive=True):
            seq_dict[os.path.dirname(f)].append(f)

    pairs = []
    last_index_of = {}
    for folder, files in seq_dict.items():
        files = sorted(files, key=extract_number)
        rel_folder = os.path.relpath(folder, root)
        save_folder = os.path.join(out_root, rel_folder)
        for i in range(len(files) - 1):
            pairs.append((files[i], files[i + 1], save_folder))
            last_index_of[save_folder] = len(pairs) - 1
    return pairs, last_index_of


# ── dump_data 输出 ────────────────────────────────────────────────────
def dump_debug_data(save_folder, mid_idx, ext,
                    warp0, warp1, flow, mask, merged, res, padder):
    subdirs = ['warp0', 'warp1', 'mv0', 'mv1', 'mask', 'merged', 'res']
    for d in subdirs:
        os.makedirs(os.path.join(save_folder, d), exist_ok=True)

    def to_np(t):
        return padder.unpad(t)[0].detach().cpu().numpy().transpose(1, 2, 0)

    flow_np  = flow[0].detach().cpu().numpy().transpose(1, 2, 0)
    mv0, mv1 = flow_np[..., :2], flow_np[..., 2:]

    write(os.path.join(save_folder, 'warp0',  f"{mid_idx:06d}{ext}"), to_np(warp0))
    write(os.path.join(save_folder, 'warp1',  f"{mid_idx:06d}{ext}"), to_np(warp1[0]))
    write(os.path.join(save_folder, 'mv0',    f"{mid_idx:06d}.exr"),  mv0)
    write(os.path.join(save_folder, 'mv1',    f"{mid_idx:06d}.exr"),  mv1)
    write(os.path.join(save_folder, 'mask',   f"{mid_idx:06d}.exr"),  to_np(mask))
    write(os.path.join(save_folder, 'merged', f"{mid_idx:06d}{ext}"), to_np(merged))
    write(os.path.join(save_folder, 'res',    f"{mid_idx:06d}.exr"),  to_np(res))


# ── 主函数 ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',       default='VFIMamba_KouSei', type=str,
                        choices=['VFIMamba_S', 'VFIMamba', 'VFIMamba_KouSei'])
    parser.add_argument('--algo',        default='VFIKousei_TEST', type=str)
    parser.add_argument('--root', '--path', required=True, type=str)
    parser.add_argument('--output',      required=True, type=str)
    parser.add_argument('--scale',       default=0, type=float)
    parser.add_argument('--timestep',       default=0.5, type=float)
    parser.add_argument('--output_mode', default='image', type=str,
                        choices=['image', 'video', 'both'])
    parser.add_argument('--fps',         default=24, type=float)
    parser.add_argument('--max_w',         default=1920, type=int)
    parser.add_argument('--max_h',         default=1080, type=int)
    parser.add_argument('--video_ext',   default='mp4', type=str,
                        choices=['mp4', 'avi'])
    parser.add_argument('--dump_data',   action='store_true')
    parser.add_argument('--old_version', action='store_true')
    parser.add_argument('--server',      default='http://10.35.180.69:80', type=str)
    args = parser.parse_args()

    assert args.model in ['VFIMamba_S', 'VFIMamba', 'VFIMamba_KouSei'], 'Model not exists!'

    TTA = False

    # 模型
    model = build_model(args)

    # 输出控制
    need_image   = args.output_mode in ('image', 'both')
    need_video   = args.output_mode in ('video', 'both')
    video_writer = SequenceVideoWriter(args.fps, args.video_ext) if need_video else None

    # 收集序列
    pairs, last_index_of = collect_pairs(args.root, args.output)
    print(f'=========================Start Generating ({len(pairs)} pairs)=========================')

    for index in tqdm(range(len(pairs))):
        src0, src1, save_folder = pairs[index]
        ext     = os.path.splitext(src0)[1]
        is_exr  = ext.lower() in ('.exr', '.tif')
        is_last = (index == last_index_of[save_folder])
        os.makedirs(save_folder, exist_ok=True)

        # 读取 & 预处理
        I0 = to_float(resize_if_needed(read(src0, type='image'),max_w=args.max_w,max_h=args.max_h))
        I2 = to_float(resize_if_needed(read(src1, type='image'),max_w=args.max_w,max_h=args.max_h))

        I0_ = torch.tensor(I0.transpose(2, 0, 1)).to(model._dev).unsqueeze(0)
        I2_ = torch.tensor(I2.transpose(2, 0, 1)).to(model._dev).unsqueeze(0)

        padder  = InputPadder(I0_.shape, divisor=16)
        I0_, I2_ = padder.pad(I0_, I2_)

        # 推理
        with torch.no_grad():
            mid, flow, mask, merged, res, warp0, warp1 = model.inference(
                I0_, I2_, True, TTA=TTA, fast_TTA=TTA, scale=args.scale,timestep=args.timestep
            )

        mid_np = padder.unpad(mid)[0].detach().cpu().numpy().transpose(1, 2, 0)

        # 命名
        idx0    = extract_number(src0) * 2
        idx1    = extract_number(src1) * 2
        mid_idx = (idx0 + idx1) // 2

        # 写帧
        frames_to_write = [(idx0, I0), (mid_idx, mid_np)]
        if is_last:
            frames_to_write.append((idx1, I2))

        for frame_idx, img_f32 in frames_to_write:
            if need_image:
                out_path = os.path.join(save_folder, f"{frame_idx:06d}{ext}")
                write(out_path, img_f32 if is_exr else to_uint8(img_f32))
            if need_video:
                video_writer.add_frame(save_folder, frame_idx, img_f32)

        # debug dump
        if args.dump_data:
            dump_debug_data(save_folder, mid_idx, ext,
                            warp0, warp1, flow, mask, merged, res, padder)

    if need_video:
        video_writer.flush()

    print('=========================Done=========================')


if __name__ == '__main__':
    main()