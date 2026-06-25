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


def load_mv_px(path, h, w):
    """读取光流并反归一化到像素单位"""
    mv = read(path, 'flo').astype(np.float32)
    mv[..., 0] *= w
    mv[..., 1] *= h
    return mv


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


def compute_occlusion_mask(mv_fwd, mv_bwd, threshold=1.0):
    """
    前向-后向一致性生成遮挡 mask。
    mv_fwd + mv_bwd ≈ 0 为非遮挡区。
    返回 occ_mask [H,W]（1=遮挡）和 epe_map [H,W]。
    """
    diff    = mv_fwd + mv_bwd
    epe_map = np.sqrt((diff ** 2).sum(-1))
    occ_mask = (epe_map > threshold).astype(np.float32)
    return occ_mask, epe_map


def save_metrics(save_path, metrics: dict):
    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, 'metrics.json')
    metrics['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'  [metrics] 已写入: {out_file}')


# ── 核心处理 ──────────────────────────────────────────────────────────
def run_immc(folder_image, folder_mv, folder_mv_bwd,
             save_path, args, direction, occ_threshold=1.0):
    """
    执行 immc 并同时统计：
      - 全图 PSNR（原始指标）
      - 非遮挡区 PSNR（加 mask 后指标）
      - 每帧遮挡占比

    folder_mv_bwd: 反方向光流文件列表，用于生成遮挡 mask。
                   from1 方向传 folder_mv0，from0 方向传 folder_mv1。
                   若为 None 则退化为全图 PSNR。
    """
    os.makedirs(save_path, exist_ok=True)
    dtype = 'hdr' if args.hdr else 'image'
    n     = len(folder_image)

    psnr_full_list    = []
    psnr_visible_list = []
    occ_ratio_list    = []
    psnr_per_frame    = {}

    for i in tqdm(range(n), desc=os.path.basename(save_path)):
        frame_name = os.path.basename(folder_image[i])

        # 读光流并反归一化
        mv_raw  = read(folder_mv[i], 'flo').astype(np.float32)
        h, w    = mv_raw.shape[:2]
        mv      = mv_raw.copy()
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

            # ── 全图 PSNR（原始指标）────────────────────────────────
            psnr_full = calc_psnr(tgt_img, out)
            psnr_full_list.append(psnr_full)

            # ── 遮挡 mask + 非遮挡区 PSNR ───────────────────────────
            psnr_visible = None
            occ_ratio    = None

            if folder_mv_bwd is not None and i < len(folder_mv_bwd):
                mv_bwd_raw = read(folder_mv_bwd[i], 'flo').astype(np.float32)
                mv_bwd     = mv_bwd_raw.copy()
                mv_bwd[..., 0] *= w
                mv_bwd[..., 1] *= h

                occ_mask, _ = compute_occlusion_mask(mv, mv_bwd, occ_threshold)
                visible_mask = 1.0 - occ_mask
                occ_ratio    = float(occ_mask.mean())

                psnr_visible = calc_psnr_masked(tgt_img, out, visible_mask)
                if psnr_visible is not None:
                    psnr_visible_list.append(psnr_visible)
                occ_ratio_list.append(occ_ratio)

            psnr_per_frame[frame_name] = {
                'psnr_full':    round(psnr_full,    4),
                'psnr_visible': round(psnr_visible, 4) if psnr_visible is not None else None,
                'occ_ratio':    round(occ_ratio,    4) if occ_ratio    is not None else None,
            }

        else:
            # 边界帧：输出全零，不参与统计
            ref_img = read(folder_image[ref_idx], type=dtype)
            out     = np.zeros_like(ref_img)
            psnr_per_frame[frame_name] = {
                'psnr_full':    None,
                'psnr_visible': None,
                'occ_ratio':    None,
                'note':         'boundary frame',
            }

        write(os.path.join(save_path, frame_name), out)

    metrics = {
        'direction':  direction,
        'n_frames':   n,
        'n_valid':    len(psnr_full_list),
        # 原始指标（全图）
        'psnr_full_avg': round(float(np.mean(psnr_full_list)),  4) if psnr_full_list else None,
        'psnr_full_min': round(float(np.min(psnr_full_list)),   4) if psnr_full_list else None,
        'psnr_full_max': round(float(np.max(psnr_full_list)),   4) if psnr_full_list else None,
        # 加 mask 后指标（非遮挡区）
        'psnr_visible_avg': round(float(np.mean(psnr_visible_list)), 4) if psnr_visible_list else None,
        'psnr_visible_min': round(float(np.min(psnr_visible_list)),  4) if psnr_visible_list else None,
        'psnr_visible_max': round(float(np.max(psnr_visible_list)),  4) if psnr_visible_list else None,
        # 遮挡占比
        'occ_ratio_avg': round(float(np.mean(occ_ratio_list)), 4) if occ_ratio_list else None,
        'occ_ratio_max': round(float(np.max(occ_ratio_list)),  4) if occ_ratio_list else None,
        # 逐帧明细
        'psnr_per_frame': psnr_per_frame,
    }
    return metrics


def check_fb_consistency(folder_mv0, folder_mv1, threshold=1.0):
    """
    前向-后向一致性检验，同时报告：
      - 全图 EPE（原始指标）
      - 非遮挡区 EPE（加 mask 后指标）
      - 遮挡占比
    """
    n      = min(len(folder_mv0), len(folder_mv1))
    errors = {}

    epe_full_list    = []
    epe_visible_list = []
    occ_ratio_list   = []

    for i in tqdm(range(n - 1), desc='fb_consistency'):
        mv0_raw = read(folder_mv0[i],   'flo').astype(np.float32)
        mv1_raw = read(folder_mv1[i+1], 'flo').astype(np.float32)

        if mv0_raw.shape != mv1_raw.shape:
            print(f'  [WARN] shape mismatch at frame {i}, skipped')
            continue

        h, w = mv0_raw.shape[:2]
        mv0  = mv0_raw.copy(); mv0[..., 0] *= w; mv0[..., 1] *= h
        mv1  = mv1_raw.copy(); mv1[..., 0] *= w; mv1[..., 1] *= h

        diff        = mv0 + mv1
        epe_map     = np.sqrt((diff ** 2).sum(-1))          # [H, W]
        occ_mask    = (epe_map > threshold).astype(bool)
        visible_mask = ~occ_mask

        epe_full     = float(epe_map.mean())
        epe_visible  = float(epe_map[visible_mask].mean()) if visible_mask.any() else None
        occ_ratio    = float(occ_mask.mean())

        epe_full_list.append(epe_full)
        if epe_visible is not None:
            epe_visible_list.append(epe_visible)
        occ_ratio_list.append(occ_ratio)

        errors[f'frame_{i:06d}'] = {
            'fb_epe_full':    round(epe_full,    6),
            'fb_epe_visible': round(epe_visible, 6) if epe_visible is not None else None,
            'occ_ratio':      round(occ_ratio,   4),
        }

    metrics = {
        'n_pairs':  n - 1,
        'n_valid':  len(epe_full_list),
        # 原始指标（全图）
        'fb_epe_full_avg': round(float(np.mean(epe_full_list)),  6) if epe_full_list else None,
        'fb_epe_full_min': round(float(np.min(epe_full_list)),   6) if epe_full_list else None,
        'fb_epe_full_max': round(float(np.max(epe_full_list)),   6) if epe_full_list else None,
        # 加 mask 后指标（非遮挡区）
        'fb_epe_visible_avg': round(float(np.mean(epe_visible_list)), 6) if epe_visible_list else None,
        'fb_epe_visible_min': round(float(np.min(epe_visible_list)),  6) if epe_visible_list else None,
        'fb_epe_visible_max': round(float(np.max(epe_visible_list)),  6) if epe_visible_list else None,
        # 遮挡占比
        'occ_ratio_avg': round(float(np.mean(occ_ratio_list)), 4) if occ_ratio_list else None,
        'occ_ratio_max': round(float(np.max(occ_ratio_list)),  4) if occ_ratio_list else None,
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

    assert len(folder_image) == len(folder_mv1), \
        f'image({len(folder_image)}) 和 mv1({len(folder_mv1)}) 数量不一致: {folder}'
    if folder_mv0:
        assert len(folder_image) == len(folder_mv0), \
            f'image({len(folder_image)}) 和 mv0({len(folder_mv0)}) 数量不一致: {folder}'

    save_base   = os.path.join(folder, 'immc')
    all_metrics = {'folder': folder, 'occ_threshold': args.occ_threshold}

    # ── from1：mv1 warp i-1→i，用 mv0 生成遮挡 mask ─────────────────
    print('  运行 from1...')
    m1 = run_immc(
        folder_image, folder_mv1,
        folder_mv_bwd=folder_mv0,          # from1 的反向是 mv0
        save_path=os.path.join(save_base, 'from1'),
        args=args, direction='from1',
        occ_threshold=args.occ_threshold,
    )
    per_frame_m1 = m1.pop('psnr_per_frame')
    all_metrics['from1'] = m1
    all_metrics['from1_psnr_per_frame'] = per_frame_m1
    print(f'  from1  psnr_full={m1["psnr_full_avg"]}  '
          f'psnr_visible={m1["psnr_visible_avg"]}  '
          f'occ_ratio={m1["occ_ratio_avg"]}')

    # ── from0：mv0 warp i+1→i，用 mv1 生成遮挡 mask（若有 mv0）──────
    if folder_mv0:
        print('  运行 from0...')
        m0 = run_immc(
            folder_image, folder_mv0,
            folder_mv_bwd=folder_mv1,      # from0 的反向是 mv1
            save_path=os.path.join(save_base, 'from0'),
            args=args, direction='from0',
            occ_threshold=args.occ_threshold,
        )
        per_frame_m0 = m0.pop('psnr_per_frame')
        all_metrics['from0'] = m0
        all_metrics['from0_psnr_per_frame'] = per_frame_m0
        print(f'  from0  psnr_full={m0["psnr_full_avg"]}  '
              f'psnr_visible={m0["psnr_visible_avg"]}  '
              f'occ_ratio={m0["occ_ratio_avg"]}')

        # ── 前向-后向一致性 ──────────────────────────────────────────
        print('  计算 fb_consistency...')
        fb = check_fb_consistency(
            folder_mv0, folder_mv1,
            threshold=args.occ_threshold,
        )
        per_pair = fb.pop('fb_epe_per_pair')
        all_metrics['fb_consistency'] = fb
        all_metrics['fb_epe_per_pair'] = per_pair
        print(f'  fb_epe_full={fb["fb_epe_full_avg"]}  '
              f'fb_epe_visible={fb["fb_epe_visible_avg"]}  '
              f'occ_ratio={fb["occ_ratio_avg"]}')
    else:
        print('  [INFO] 无 mv0，跳过 from0 和 fb_consistency')

    save_metrics(save_base, all_metrics)


# ── 主函数 ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--root', required=True, help='根目录')
    parser.add_argument('--hdr',   action='store_true', help='使用 HDR 图像')
    parser.add_argument('--debug', action='store_true', help='Debug 模式')
    parser.add_argument('--occ_threshold', default=1.0, type=float,
                        help='遮挡判定阈值（像素），默认 1.0')
    args = parser.parse_args()

    folders = find_valid_folders(args.path)
    if not folders:
        print(f'[ERROR] 未找到包含 image + mv1 的子文件夹: {args.path}')
        return

    print(f'找到 {len(folders)} 个有效文件夹  occ_threshold={args.occ_threshold}')
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