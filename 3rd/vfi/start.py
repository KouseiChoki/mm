import cv2
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
import yaml
import config as cfg
from Trainer import Model
from file_utils import read, write,mkdir
from model.warplayer import clear_warp_cache

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
        self._writer = None
        self._folder = None
        self._frame_count = 0
        self._size = None

    def add_frame(self, save_folder: str, frame_idx: int, img_f32: np.ndarray):
        bgr = to_uint8(img_f32)[..., ::-1].copy()
        h, w = bgr.shape[:2]
        if self._writer is None or self._folder != save_folder:
            self.close()
            video_path = os.path.join(save_folder, f'output.{self.ext}')
            self._writer = cv2.VideoWriter(
                video_path, self.fourcc, self.fps, (w, h))
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                raise RuntimeError(f'[VideoWriter] 无法创建视频: {video_path}')
            self._folder = save_folder
            self._frame_count = 0
            self._size = (w, h)
        elif self._size != (w, h):
            raise ValueError(
                f'[VideoWriter] 同一序列帧尺寸不一致: '
                f'expected={self._size}, got={(w, h)}, frame={frame_idx}')
        self._writer.write(bgr)
        self._frame_count += 1

    def close(self):
        if self._writer is None:
            return
        video_path = os.path.join(self._folder, f'output.{self.ext}')
        self._writer.release()
        print(f'[VideoWriter] 已保存: {video_path}  '
              f'({self._frame_count} 帧)')
        self._writer = None
        self._folder = None
        self._frame_count = 0
        self._size = None

    def finish_sequence(self, save_folder):
        if self._writer is not None and self._folder != save_folder:
            raise RuntimeError(
                f'[VideoWriter] 序列关闭顺序错误: '
                f'active={self._folder}, requested={save_folder}')
        self.close()

    def flush(self):
        self.close()


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
_TRAIN_ONLY_KEYS = (
    'loss_type', 'flow_loss_weight', 'flow_stage_gamma',
    'flow_motion_threshold', 'flow_motion_balance', 'flow_motion_gain',
    'flow_motion_scale', 'flow_motion_weight_cap', 'flow_charbonnier_eps',
    'flow_loss_warmup_steps', 'merge_loss_gamma', 'merge_loss_weights',
    'normalize_pixel_loss', 'residual_loss_weight',
    'lc_charbonnier_eps', 'lc_census_weight', 'lc_lap_weight',
    'lc_warp_weight', 'pervfi_mask_loss_weight',
    'multi_hypothesis_oracle_weight', 'multi_hypothesis_oracle_eps',
    'edge_loss_weight', 'edge_warp_loss_weight',
    'edge_motion_gain', 'edge_motion_scale', 'edge_weight_cap',
    'edge_charbonnier_eps',
    'hard_roi_ratio', 'hard_roi_min_residual', 'hard_roi_dilation',
    'hard_roi_final_weight', 'hard_roi_warp_weight',
    'hard_roi_edge_weight', 'hard_roi_charbonnier_eps',
)                                                        # 训练侧消费, 不进结构
_NAMED_KEYS = ('F', 'depth', 'M', 'version')


def arch_from_model_section(m):
    """yaml的model段 → (backbonecfg, multiscalecfg)。与train.py透传逻辑同一份约定:
    F/depth/M/version 具名, 训练键排除, 其余结构字段全部透传覆盖。"""
    extra = {k: v for k, v in m.items()
             if k not in _NAMED_KEYS + _TRAIN_ONLY_KEYS}
    return cfg.init_model_config(
        F=m.get('F', 32),
        depth=m.get('depth', [2, 2, 2, 3, 3]),
        M=m.get('M', False),
        version=m.get('version', 2),
        **extra)


def resolve_model_yaml(ckpt_path, args):
    """按优先级定位模型结构yaml, 找不到返回None:
      1. --model_yaml 显式指定 (不存在则报错)
      2. checkpoint 同名yaml     (../../checkpoints/{algo}.yaml, 与.pth同stem)
      3. checkpoint 同目录 model.yaml (训练侧 ckpt/{exp}/ 归档布局)
    """
    explicit = getattr(args, 'model_yaml', None)
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f'--model_yaml 指定的文件不存在: {explicit}')
        return explicit
    stem_yaml = os.path.splitext(ckpt_path)[0] + '.yaml'
    if not os.path.isfile(stem_yaml):
        # 软尝试从checkpoint服务器同步；最终仍缺失时由build_model统一报错。
        try:
            check_and_download_pth_file(stem_yaml,
                                        f'{args.server}/vfi/{args.algo}.yaml')
        except Exception:
            pass
    if os.path.isfile(stem_yaml):
        return stem_yaml
    dir_yaml = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), 'model.yaml')
    return dir_yaml if os.path.isfile(dir_yaml) else None


def build_model(args):
    args.fp = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../checkpoints"))
    ckpt = define_model(args)                 # 先定位checkpoint, 再找配套yaml

    yaml_path = resolve_model_yaml(ckpt, args)
    if yaml_path is None:
        raise FileNotFoundError(
            '未找到模型结构yaml；release必须同时提供模型与yaml。'
            f'请使用 --model_yaml，或放置 {os.path.splitext(ckpt)[0]}.yaml / '
            f'{os.path.join(os.path.dirname(os.path.abspath(ckpt)), "model.yaml")}')

    # 约定路径: 结构完全由训练归档的yaml决定, 与训练逐字段对齐
    with open(yaml_path) as f:
        conf = yaml.safe_load(f)
    m = conf.get('model', conf)               # 兼容整份train_config或裸model段
    cfg.MODEL_CONFIG['LOGNAME'] = conf.get('exp_name', args.algo)
    cfg.MODEL_CONFIG['MODEL_ARCH'] = arch_from_model_section(m)
    print(f'[build_model] 结构来自yaml: {yaml_path}  '
          f'(version={m.get("version", 2)}, '
          f'blend={m.get("blend_mode", "soft")})')

    model = Model()
    model.load_model(ckpt)
    # Inference never resumes optimizer/scaler state. Keeping the complete
    # training checkpoint here can occupy several extra GiB of unified memory.
    model.loaded_checkpoint = None
    model.optimG = None
    model.eval()
    model.device()
    return model


# ── pair 收集 ─────────────────────────────────────────────────────────
def collect_pairs(root, out_root):
    exts = ["*.png", "*.exr", "*.tif","*.jpg"]
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
        if len(t.shape)==4:
            tmp = padder.unpad(t)[0]
        else:
            tmp = padder.unpad(t)
        return tmp.detach().cpu().numpy().transpose(1, 2, 0)

    # 网络内部flow使用像素位移；dump按训练数据约定保存归一化位移:
    # x/W, y/H。必须先去掉pad，再按原始图像尺寸归一化。
    flow_np = to_np(flow).astype(np.float32, copy=False)
    h, w = flow_np.shape[:2]
    flow_scale = np.array([w, h, w, h], dtype=np.float32)
    flow_np = flow_np[..., :4] / flow_scale
    mv0, mv1 = flow_np[..., 2:4], flow_np[..., :2]

    # mask_list保存的是sigmoid后的[0,1]融合权重。file_utils对单通道EXR
    # 默认使用Z通道，常规RGB查看器会因此显示成全黑；复制为RGB三通道，
    # 既保留原始浮点数值，也能直接预览。
    mask_np = to_np(mask).astype(np.float32, copy=False)
    if mask_np.ndim != 3 or mask_np.shape[-1] != 1:
        raise ValueError(
            f'[dump] mask应为HxWx1，实际为{mask_np.shape}')
    mask_rgb = np.repeat(mask_np, 3, axis=-1)

    write(os.path.join(save_folder, 'warp0',  f"{mid_idx:06d}{ext}"), to_np(warp0), dtype='image')
    write(os.path.join(save_folder, 'warp1',  f"{mid_idx:06d}{ext}"), to_np(warp1), dtype='image')
    write(os.path.join(save_folder, 'mv0',    f"{mid_idx:06d}.exr"),  mv0,
          dtype='flow')
    write(os.path.join(save_folder, 'mv1',    f"{mid_idx:06d}.exr"),  mv1,
          dtype='flow')
    write(os.path.join(save_folder, 'mask',   f"{mid_idx:06d}.exr"),  mask_rgb,
          dtype='image')
    write(os.path.join(save_folder, 'merged', f"{mid_idx:06d}{ext}"), to_np(merged), dtype='image')
    write(os.path.join(save_folder, 'res',    f"{mid_idx:06d}.exr"),  to_np(res))


# ── 主函数 ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo',        default='VFIKousei_TEST', type=str)
    parser.add_argument('--root', '--path', required=True, type=str)
    parser.add_argument('--output',      required=True, type=str)
    parser.add_argument('--scale',       default=0, type=float)
    parser.add_argument('--timestep',       default=0.5, type=float)
    parser.add_argument('--output_mode', default='image', type=str,
                        choices=['image', 'video', 'both'])
    parser.add_argument('--fps',         default=24, type=float)
    parser.add_argument('--max_w',         default=4096, type=int)
    parser.add_argument('--max_h',         default=4096, type=int)
    parser.add_argument('--video_ext',   default='mp4', type=str,
                        choices=['mp4', 'avi'])
    parser.add_argument('--dump_data',   action='store_true')
    tta_group = parser.add_mutually_exclusive_group()
    tta_group.add_argument(
        '--tta', dest='tta', action='store_true',
        help='启用标准TTA（两次串行推理；默认启用）')
    tta_group.add_argument(
        '--no-tta', '--no_tta', dest='tta', action='store_false',
        help='关闭TTA，只执行一次推理')
    parser.set_defaults(tta=True)
    parser.add_argument(
        '--fast_tta', '--fast-tta', action='store_true',
        help='启用batch合并的快速TTA；设置后优先于标准TTA')
    parser.add_argument('--model_yaml',  default=None, type=str,
                        help='模型结构yaml；缺省时自动查找 {algo}.yaml / model.yaml')
    parser.add_argument(
        '--refine-res-scale', type=float, default=None,
        help='仅本次推理覆盖refiner残差强度，例如0.05；不修改归档yaml')
    parser.add_argument('--server',      default='http://10.35.180.69:80', type=str)
    args = parser.parse_args()

    # 模型
    model = build_model(args)
    if args.refine_res_scale is not None:
        if not 0.0 <= args.refine_res_scale <= 1.0:
            parser.error('--refine-res-scale 必须在 [0, 1] 范围内')
        model.net.refine_res_scale = args.refine_res_scale
    print(f'[refiner] scale={model.net.refine_res_scale:g}')
    tta_mode = 'fast' if args.fast_tta else ('standard' if args.tta else 'off')
    print(f'[inference] TTA={tta_mode}')

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
        with torch.inference_mode():
            outputs = model.inference(
                I0_, I2_, True, TTA=args.tta, fast_TTA=args.fast_tta,
                scale=args.scale, timestep=args.timestep,
                return_debug=args.dump_data)
        if args.dump_data:
            mid, flow, mask, merged, res, warp0, warp1 = outputs
        else:
            mid = outputs
            flow = mask = merged = res = warp0 = warp1 = None
        del outputs

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
                out_path = os.path.join(save_folder,'vfi', f"{frame_idx:06d}{ext}")
                write(out_path, img_f32 if is_exr else to_uint8(img_f32))
            if need_video:
                video_writer.add_frame(save_folder, frame_idx, img_f32)

        # debug dump
        if args.dump_data:
            dump_debug_data(save_folder, mid_idx, ext,
                            warp0, warp1, flow, mask, merged, res, padder)

        if need_video and is_last:
            video_writer.finish_sequence(save_folder)

        # Loop variables would otherwise keep the previous MPS tensors alive
        # while the next inference is evaluated, increasing the steady peak.
        del I0_, I2_, mid, flow, mask, merged, res, warp0, warp1
        del I0, I2, mid_np, frames_to_write, img_f32, padder

        if is_last:
            clear_warp_cache()
            if model._dev.type == 'mps':
                torch.mps.synchronize()
                torch.mps.empty_cache()

    if need_video:
        video_writer.flush()

    print('=========================Done=========================')


if __name__ == '__main__':
    main()
