import os
import sys
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from myutil import immc, jhelp_folder, jhelp_file, write, read
import argparse
from tqdm import tqdm


# ── 工具函数 ──────────────────────────────────────────────────────────
def find_valid_folders(root_dir):
    valid = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        has_image = any(d in ('image', 'video') for d in dirnames)
        has_mv1   = any('mv1' in d for d in dirnames)
        if has_image and has_mv1:
            valid.append(dirpath)
    return valid


def get_mv_name(folder, keyword):
    for d in jhelp_folder(folder):
        if keyword in d and keyword != d:
            return d
    return keyword


def load_mask(path):
    """
    读取 mask 图像，返回 [H, W] float32，值域 0~1。
    1 = 有效区域（前景或需要参与计算的区域）
    0 = 无效区域（背景 border、遮挡区等）
    支持单通道和多通道图像，多通道取均值后二值化。
    """
    mask = read(path, type='image').astype(np.float32)
    if mask.ndim == 3:
        mask = mask.mean(axis=-1)
    # 归一化到 0~1
    if mask.max() > 1.0:
        mask = mask / 255.0
    # 二值化：> 0.5 为有效区域
    return (mask > 0.5).astype(np.float32)


def calc_psnr(img_a, img_b):
    mse = np.mean((img_a.astype(np.float32) - img_b.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(255.0 ** 2 / mse))


def calc_psnr_masked(img_a, img_b, mask_valid):
    """只在 mask_valid=1 的像素上计算 PSNR"""
    if mask_valid.sum() == 0:
        return None
    mask3 = mask_valid[..., np.newaxis]
    diff  = (img_a.astype(np.float32) - img_b.astype(np.float32)) ** 2
    mse   = (diff * mask3).sum() / (mask_valid.sum() * img_a.shape[-1])
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(255.0 ** 2 / mse))


def save_metrics(save_path, metrics: dict):
    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, 'metrics.json')
    metrics['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'  [metrics] 已写入: {out_file}')


# ── 核心处理 ──────────────────────────────────────────────────────────
def run_immc(folder_image, folder_mv, folder_mask,
             save_path, args, direction):
    """
    执行 immc，同时统计：
      - 全图 PSNR（原始指标）
      - mask 区域 PSNR（用外部 Mask 文件夹的 mask）
      - mask 覆盖占比

    folder_mask: Mask 文件夹的文件列表，与 folder_image 一一对应。
                 若为 None 则只计算全图 PSNR。
    """
    os.makedirs(save_path, exist_ok=True)
    dtype = 'hdr' if args.hdr else 'image'
    n     = len(folder_image)

    psnr_full_list   = []
    psnr_masked_list = []
    mask_ratio_list  = []
    psnr_per_frame   = {}

    for i in tqdm(range(n), desc=os.path.basename(save_path)):
        frame_name = os.path.basename(folder_image[i])

        mv = read(folder_mv[i], 'flo').astype(np.float32)
        h, w = mv.shape[:2]
        mv[..., 0] *= w
        mv[..., 1] *= h

        if direction == 'from0':
            src_idx = i + 1
            ref_idx = i - 1 if i > 0 else i + 1
        else:
            src_idx = i - 1
            ref_idx = i + 1 if i < n - 1 else i - 1

        valid = (direction == 'from0' and i < n - 1) or \
                (direction == 'from1' and i > 0)

        if valid:
            src_img = read(folder_image[src_idx], type=dtype)
            tgt_img = read(folder_image[i],       type=dtype)
            out     = immc(src_img, mv)

            # ── 全图 PSNR ─────────────────────────────────────────────
            psnr_full = calc_psnr(tgt_img, out)
            psnr_full_list.append(psnr_full)

            # ── 外部 mask 区域 PSNR ───────────────────────────────────
            psnr_masked = None
            mask_ratio  = None

            if folder_mask is not None and i < len(folder_mask):
                mask       = load_mask(folder_mask[i])
                mask_ratio = float(mask.mean())

                # mask 尺寸和图像不一致时 resize
                if mask.shape[:2] != (h, w):
                    import cv2
                    mask = cv2.resize(mask, (w, h),
                                      interpolation=cv2.INTER_NEAREST)

                psnr_masked = calc_psnr_masked(tgt_img, out, mask)
                if psnr_masked is not None:
                    psnr_masked_list.append(psnr_masked)
                mask_ratio_list.append(mask_ratio)

            psnr_per_frame[frame_name] = {
                'psnr_full':   round(psnr_full,   4),
                'psnr_masked': round(psnr_masked, 4) if psnr_masked is not None else None,
                'mask_ratio':  round(mask_ratio,  4) if mask_ratio  is not None else None,
            }

        else:
            ref_img = read(folder_image[ref_idx], type=dtype)
            out     = np.zeros_like(ref_img)
            psnr_per_frame[frame_name] = {
                'psnr_full':   None,
                'psnr_masked': None,
                'mask_ratio':  None,
                'note':        'boundary frame',
            }

        write(os.path.join(save_path, frame_name), out)

    metrics = {
        'direction': direction,
        'n_frames':  n,
        'n_valid':   len(psnr_full_list),
        # 原始指标（全图）
        'psnr_full_avg': round(float(np.mean(psnr_full_list)),  4) if psnr_full_list else None,
        'psnr_full_min': round(float(np.min(psnr_full_list)),   4) if psnr_full_list else None,
        'psnr_full_max': round(float(np.max(psnr_full_list)),   4) if psnr_full_list else None,
        # mask 区域指标
        'psnr_masked_avg': round(float(np.mean(psnr_masked_list)), 4) if psnr_masked_list else None,
        'psnr_masked_min': round(float(np.min(psnr_masked_list)),  4) if psnr_masked_list else None,
        'psnr_masked_max': round(float(np.max(psnr_masked_list)),  4) if psnr_masked_list else None,
        # mask 覆盖占比
        'mask_ratio_avg': round(float(np.mean(mask_ratio_list)), 4) if mask_ratio_list else None,
        'mask_ratio_max': round(float(np.max(mask_ratio_list)),  4) if mask_ratio_list else None,
        # 逐帧明细
        'psnr_per_frame': psnr_per_frame,
    }
    return metrics


def check_fb_consistency(folder_mv0, folder_mv1, folder_mask):
    """
    前向-后向一致性检验，同时报告：
      - 全图 EPE（原始指标）
      - mask 区域 EPE（用外部 Mask）
      - mask 覆盖占比
    """
    n     = min(len(folder_mv0), len(folder_mv1))
    errors = {}

    epe_full_list   = []
    epe_masked_list = []
    mask_ratio_list = []

    for i in tqdm(range(n - 1), desc='fb_consistency'):
        mv0_raw = read(folder_mv0[i],   'flo').astype(np.float32)
        mv1_raw = read(folder_mv1[i+1], 'flo').astype(np.float32)

        if mv0_raw.shape != mv1_raw.shape:
            print(f'  [WARN] shape mismatch at frame {i}, skipped')
            continue

        h, w = mv0_raw.shape[:2]
        mv0  = mv0_raw.copy(); mv0[..., 0] *= w; mv0[..., 1] *= h
        mv1  = mv1_raw.copy(); mv1[..., 0] *= w; mv1[..., 1] *= h

        diff    = mv0 + mv1
        epe_map = np.sqrt((diff ** 2).sum(-1))   # [H, W]

        epe_full   = float(epe_map.mean())
        epe_masked = None
        mask_ratio = None

        if folder_mask is not None and i < len(folder_mask):
            mask = load_mask(folder_mask[i])
            if mask.shape[:2] != (h, w):
                import cv2
                mask = cv2.resize(mask, (w, h),
                                  interpolation=cv2.INTER_NEAREST)
            mask_bool  = mask.astype(bool)
            mask_ratio = float(mask.mean())
            epe_masked = float(epe_map[mask_bool].mean()) if mask_bool.any() else None
            if epe_masked is not None:
                epe_masked_list.append(epe_masked)
            mask_ratio_list.append(mask_ratio)

        epe_full_list.append(epe_full)

        errors[f'frame_{i:06d}'] = {
            'fb_epe_full':   round(epe_full,   6),
            'fb_epe_masked': round(epe_masked, 6) if epe_masked is not None else None,
            'mask_ratio':    round(mask_ratio, 4) if mask_ratio is not None else None,
        }

    metrics = {
        'n_pairs': n - 1,
        'n_valid': len(epe_full_list),
        # 原始指标（全图）
        'fb_epe_full_avg': round(float(np.mean(epe_full_list)),  6) if epe_full_list else None,
        'fb_epe_full_min': round(float(np.min(epe_full_list)),   6) if epe_full_list else None,
        'fb_epe_full_max': round(float(np.max(epe_full_list)),   6) if epe_full_list else None,
        # mask 区域指标
        'fb_epe_masked_avg': round(float(np.mean(epe_masked_list)), 6) if epe_masked_list else None,
        'fb_epe_masked_min': round(float(np.min(epe_masked_list)),  6) if epe_masked_list else None,
        'fb_epe_masked_max': round(float(np.max(epe_masked_list)),  6) if epe_masked_list else None,
        # mask 覆盖占比
        'mask_ratio_avg': round(float(np.mean(mask_ratio_list)), 4) if mask_ratio_list else None,
        'mask_ratio_max': round(float(np.max(mask_ratio_list)),  4) if mask_ratio_list else None,
        # 逐帧明细
        'fb_epe_per_pair': errors,
    }
    return metrics


def process_folder(folder, args):
    image_name = 'video' if args.hdr else 'image'

    # 找 image 文件夹
    img_dir = os.path.join(folder, image_name)
    if not os.path.isdir(img_dir):
        img_dir = folder
    folder_image = jhelp_file(img_dir)
    if not folder_image:
        print(f'[SKIP] 无图像文件: {folder}')
        return

    # 找 mv1（必须有）
    mv1_name   = get_mv_name(folder, 'mv1')
    mv1_dir    = os.path.join(folder, mv1_name)
    folder_mv1 = jhelp_file(mv1_dir) if os.path.isdir(mv1_dir) else None
    if not folder_mv1:
        print(f'[SKIP] 找不到 mv1: {folder}')
        return

    # 找 mv0（可选）
    mv0_name   = get_mv_name(folder, 'mv0')
    mv0_dir    = os.path.join(folder, mv0_name)
    folder_mv0 = jhelp_file(mv0_dir) if os.path.isdir(mv0_dir) else None

    # 找 Mask（可选）
    mask_dir    = os.path.join(folder, 'Mask')
    folder_mask = jhelp_file(mask_dir) if os.path.isdir(mask_dir) else None
    if folder_mask:
        print(f'  [Mask] 已找到 {len(folder_mask)} 个 mask 文件')
        assert len(folder_mask) == len(folder_image), \
            f'Mask({len(folder_mask)}) 和 image({len(folder_image)}) 数量不一致: {folder}'
    else:
        print('  [Mask] 未找到 Mask 文件夹，仅计算全图指标')

    assert len(folder_image) == len(folder_mv1), \
        f'image({len(folder_image)}) 和 mv1({len(folder_mv1)}) 数量不一致: {folder}'
    if folder_mv0:
        assert len(folder_image) == len(folder_mv0), \
            f'image({len(folder_image)}) 和 mv0({len(folder_mv0)}) 数量不一致: {folder}'

    save_base   = os.path.join(folder, 'immc')
    all_metrics = {
        'folder':      folder,
        'has_mask':    folder_mask is not None,
    }

    # ── from1 ────────────────────────────────────────────────────────
    print('  运行 from1...')
    m1 = run_immc(folder_image, folder_mv1, folder_mask,
                  os.path.join(save_base, 'from1'), args, direction='from1')
    per_frame_m1 = m1.pop('psnr_per_frame')
    all_metrics['from1'] = m1
    all_metrics['from1_psnr_per_frame'] = per_frame_m1
    print(f'  from1  psnr_full={m1["psnr_full_avg"]}  '
          f'psnr_masked={m1["psnr_masked_avg"]}  '
          f'mask_ratio={m1["mask_ratio_avg"]}')

    # ── from0（若有 mv0）─────────────────────────────────────────────
    if folder_mv0:
        print('  运行 from0...')
        m0 = run_immc(folder_image, folder_mv0, folder_mask,
                      os.path.join(save_base, 'from0'), args, direction='from0')
        per_frame_m0 = m0.pop('psnr_per_frame')
        all_metrics['from0'] = m0
        all_metrics['from0_psnr_per_frame'] = per_frame_m0
        print(f'  from0  psnr_full={m0["psnr_full_avg"]}  '
              f'psnr_masked={m0["psnr_masked_avg"]}  '
              f'mask_ratio={m0["mask_ratio_avg"]}')

        # ── fb_consistency ────────────────────────────────────────────
        print('  计算 fb_consistency...')
        fb = check_fb_consistency(folder_mv0, folder_mv1, folder_mask)
        per_pair = fb.pop('fb_epe_per_pair')
        all_metrics['fb_consistency'] = fb
        all_metrics['fb_epe_per_pair'] = per_pair
        print(f'  fb_epe_full={fb["fb_epe_full_avg"]}  '
              f'fb_epe_masked={fb["fb_epe_masked_avg"]}  '
              f'mask_ratio={fb["mask_ratio_avg"]}')
    else:
        print('  [INFO] 无 mv0，跳过 from0 和 fb_consistency')

    save_metrics(save_base, all_metrics)


# ── 主函数 ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--root', required=True, help='根目录')
    parser.add_argument('--hdr',   action='store_true', help='使用 HDR 图像')
    parser.add_argument('--debug', action='store_true', help='Debug 模式')
    args = parser.parse_args()

    folders = find_valid_folders(args.path)
    if not folders:
        print(f'[ERROR] 未找到包含 image + mv1 的子文件夹: {args.path}')
        return

    print(f'找到 {len(folders)} 个有效文件夹')
    for folder in folders:
        print(f'\n处理: {folder}')
        try:
            process_folder(folder, args)
        except Exception as e:
            print(f'[ERROR] {folder}: {e}')
            if args.debug:
                raise


if __name__ == '__main__':
    main()