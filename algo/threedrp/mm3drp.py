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
    mask = read(path, type='image').astype(np.float32)
    if mask.ndim == 3:
        mask = mask.mean(axis=-1)
    if mask.max() > 1.0:
        mask = mask / 255.0
    return (mask > 0.5).astype(np.float32)


def calc_psnr(img_a, img_b):
    mse = np.mean((img_a.astype(np.float32) - img_b.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(255.0 ** 2 / mse))


def calc_psnr_masked(img_a, img_b, mask_valid):
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
    os.makedirs(save_path, exist_ok=True)
    dtype = 'hdr' if args.hdr else 'image'
    n     = len(folder_image)
    frames = {}

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

            psnr_full   = calc_psnr(tgt_img, out)
            psnr_masked = None
            mask_ratio  = None

            if folder_mask is not None and i < len(folder_mask):
                mask = load_mask(folder_mask[i])
                if mask.shape[:2] != (h, w):
                    import cv2
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_ratio  = float(mask.mean())
                psnr_masked = calc_psnr_masked(tgt_img, out, mask)

            frames[frame_name] = {
                'frame':             frame_name,
                'index':             i,
                'boundary':          False,
                'psnr_full':         round(psnr_full,   4),
                'psnr_masked':       round(psnr_masked, 4) if psnr_masked is not None else None,
                'mask_ratio':        round(mask_ratio,  4) if mask_ratio  is not None else None,
                'fb_epe_full':       None,
                'fb_epe_masked':     None,
                'abnormal':          False,
                'abnormal_reasons':  [],
            }
        else:
            ref_img = read(folder_image[ref_idx], type=dtype)
            out     = np.zeros_like(ref_img)
            frames[frame_name] = {
                'frame':             frame_name,
                'index':             i,
                'boundary':          True,
                'psnr_full':         None,
                'psnr_masked':       None,
                'mask_ratio':        None,
                'fb_epe_full':       None,
                'fb_epe_masked':     None,
                'abnormal':          False,
                'abnormal_reasons':  ['boundary frame'],
            }

        write(os.path.join(save_path, frame_name), out)

    return frames


def calc_fb_per_frame(folder_mv0, folder_mv1, folder_mask, folder_image):
    n      = min(len(folder_mv0), len(folder_mv1), len(folder_image))
    result = {}

    for i in tqdm(range(n), desc='fb_epe'):
        frame_name = os.path.basename(folder_image[i])

        mv0_raw = read(folder_mv0[i], 'flo').astype(np.float32)
        mv1_raw = read(folder_mv1[i], 'flo').astype(np.float32)

        if mv0_raw.shape != mv1_raw.shape:
            print(f'  [WARN] fb shape mismatch at frame {i}, skipped')
            result[frame_name] = {'fb_epe_full': None, 'fb_epe_masked': None}
            continue

        h, w = mv0_raw.shape[:2]
        mv0  = mv0_raw.copy(); mv0[..., 0] *= w; mv0[..., 1] *= h
        mv1  = mv1_raw.copy(); mv1[..., 0] *= w; mv1[..., 1] *= h

        diff    = mv0 + mv1
        epe_map = np.sqrt((diff ** 2).sum(-1))

        epe_full   = float(epe_map.mean())
        epe_masked = None

        if folder_mask is not None and i < len(folder_mask):
            mask = load_mask(folder_mask[i])
            if mask.shape[:2] != (h, w):
                import cv2
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bool  = mask.astype(bool)
            epe_masked = float(epe_map[mask_bool].mean()) if mask_bool.any() else None

        result[frame_name] = {
            'fb_epe_full':   round(epe_full,   6),
            'fb_epe_masked': round(epe_masked, 6) if epe_masked is not None else None,
        }

    return result


def merge_and_judge(frames: dict, fb_map: dict, args) -> tuple:
    """
    1. 合并 fb_epe 进 frames
    2. 用场景内所有有效帧的均值和标准差计算动态阈值
       psnr_masked  异常条件：< mean - k_sigma * std
       fb_epe_masked 异常条件：> mean + k_sigma * std
    3. 逐帧打标，汇总 summary
    """
    # ── Step 1: 合并 fb_epe ──────────────────────────────────────────
    for fname, rec in frames.items():
        if not rec['boundary'] and fname in fb_map:
            rec['fb_epe_full']   = fb_map[fname]['fb_epe_full']
            rec['fb_epe_masked'] = fb_map[fname]['fb_epe_masked']

    # ── Step 2: 收集有效帧的指标，计算场景统计量 ─────────────────────
    psnr_masked_vals  = []
    epe_masked_vals   = []
    psnr_full_vals    = []
    epe_full_vals     = []

    for rec in frames.values():
        if rec['boundary']:
            continue
        if rec['psnr_full']    is not None: psnr_full_vals.append(rec['psnr_full'])
        if rec['psnr_masked']  is not None: psnr_masked_vals.append(rec['psnr_masked'])
        if rec['fb_epe_full']  is not None: epe_full_vals.append(rec['fb_epe_full'])
        if rec['fb_epe_masked'] is not None: epe_masked_vals.append(rec['fb_epe_masked'])

    def scene_stats(vals):
        if not vals:
            return None, None, None, None
        a = np.array(vals, dtype=np.float64)
        return float(a.mean()), float(a.std()), float(a.min()), float(a.max())

    pm_mean, pm_std, pm_min, pm_max = scene_stats(psnr_masked_vals)
    em_mean, em_std, em_min, em_max = scene_stats(epe_masked_vals)

    # 动态阈值：均值 ± k_sigma 倍标准差
    k = args.k_sigma
    psnr_thr = (pm_mean - k * pm_std) if (pm_mean is not None and pm_std is not None) else None
    epe_thr  = (em_mean + k * em_std) if (em_mean is not None and em_std is not None) else None

    dynamic_thresholds = {
        'k_sigma':              k,
        'psnr_masked_mean':     round(pm_mean, 4) if pm_mean is not None else None,
        'psnr_masked_std':      round(pm_std,  4) if pm_std  is not None else None,
        'psnr_masked_thr':      round(psnr_thr,4) if psnr_thr is not None else None,
        'fb_epe_masked_mean':   round(em_mean, 4) if em_mean is not None else None,
        'fb_epe_masked_std':    round(em_std,  4) if em_std  is not None else None,
        'fb_epe_masked_thr':    round(epe_thr, 4) if epe_thr is not None else None,
    }

    # ── Step 3: 逐帧异常判断 ─────────────────────────────────────────
    abnormal_frames = []

    for rec in frames.values():
        if rec['boundary']:
            continue

        reasons = []

        if psnr_thr is not None and rec['psnr_masked'] is not None:
            if rec['psnr_masked'] < psnr_thr:
                reasons.append(
                    f'psnr_masked={rec["psnr_masked"]:.2f} < '
                    f'scene_mean-{k}σ ({psnr_thr:.2f})'
                )

        if epe_thr is not None and rec['fb_epe_masked'] is not None:
            if rec['fb_epe_masked'] > epe_thr:
                reasons.append(
                    f'fb_epe_masked={rec["fb_epe_masked"]:.2f} > '
                    f'scene_mean+{k}σ ({epe_thr:.2f})'
                )

        rec['abnormal']         = len(reasons) > 0
        rec['abnormal_reasons'] = reasons

        if rec['abnormal']:
            abnormal_frames.append({
                'frame':             rec['frame'],
                'index':             rec['index'],
                'psnr_masked':       rec['psnr_masked'],
                'fb_epe_masked':     rec['fb_epe_masked'],
                'abnormal_reasons':  reasons,
            })

    # ── Step 4: 汇总 summary ─────────────────────────────────────────
    def safe_stats(vals):
        if not vals:
            return {'avg': None, 'min': None, 'max': None, 'std': None}
        a = np.array(vals)
        return {
            'avg': round(float(a.mean()), 4),
            'std': round(float(a.std()),  4),
            'min': round(float(a.min()),  4),
            'max': round(float(a.max()),  4),
        }

    summary = {
        'n_frames':          len(frames),
        'n_valid':           len(psnr_full_vals),
        'n_abnormal':        len(abnormal_frames),
        'psnr_full':         safe_stats(psnr_full_vals),
        'psnr_masked':       safe_stats(psnr_masked_vals),
        'fb_epe_full':       safe_stats(epe_full_vals),
        'fb_epe_masked':     safe_stats(epe_masked_vals),
        'dynamic_thresholds': dynamic_thresholds,
    }

    return summary, abnormal_frames


def process_folder(folder, args):
    image_name = 'video' if args.hdr else 'image'

    img_dir = os.path.join(folder, image_name)
    if not os.path.isdir(img_dir):
        img_dir = folder
    folder_image = jhelp_file(img_dir)
    if not folder_image:
        print(f'[SKIP] 无图像文件: {folder}')
        return

    mv1_name   = get_mv_name(folder, 'mv1')
    mv1_dir    = os.path.join(folder, mv1_name)
    folder_mv1 = jhelp_file(mv1_dir) if os.path.isdir(mv1_dir) else None
    if not folder_mv1:
        print(f'[SKIP] 找不到 mv1: {folder}')
        return

    mv0_name   = get_mv_name(folder, 'mv0')
    mv0_dir    = os.path.join(folder, mv0_name)
    folder_mv0 = jhelp_file(mv0_dir) if os.path.isdir(mv0_dir) else None

    mask_dir    = os.path.join(folder, 'Mask')
    folder_mask = jhelp_file(mask_dir) if os.path.isdir(mask_dir) else None
    if folder_mask:
        print(f'  [Mask] 已找到 {len(folder_mask)} 个 mask 文件')
        assert len(folder_mask) == len(folder_image), \
            f'Mask({len(folder_mask)}) 和 image({len(folder_image)}) 数量不一致'
    else:
        print('  [Mask] 未找到 Mask 文件夹，仅计算全图指标')

    assert len(folder_image) == len(folder_mv1), \
        f'image({len(folder_image)}) 和 mv1({len(folder_mv1)}) 数量不一致'
    if folder_mv0:
        assert len(folder_image) == len(folder_mv0), \
            f'image({len(folder_image)}) 和 mv0({len(folder_mv0)}) 数量不一致'

    save_base = os.path.join(folder, 'immc')

    print('  运行 from1...')
    frames_m1 = run_immc(folder_image, folder_mv1, folder_mask,
                         os.path.join(save_base, 'from1'), args, direction='from1')

    frames_m0 = None
    fb_map    = {}

    if folder_mv0:
        print('  运行 from0...')
        frames_m0 = run_immc(folder_image, folder_mv0, folder_mask,
                             os.path.join(save_base, 'from0'), args, direction='from0')
        print('  计算逐帧 fb_epe...')
        fb_map = calc_fb_per_frame(folder_mv0, folder_mv1, folder_mask, folder_image)
    else:
        print('  [INFO] 无 mv0，跳过 from0 和 fb_epe')

    summary_m1, abnormal_m1 = merge_and_judge(frames_m1, fb_map, args)
    thr1 = summary_m1['dynamic_thresholds']
    print(f'  from1  psnr_masked={summary_m1["psnr_masked"]["avg"]}  '
          f'(thr<{thr1["psnr_masked_thr"]})  '
          f'fb_epe_masked={summary_m1["fb_epe_masked"]["avg"]}  '
          f'(thr>{thr1["fb_epe_masked_thr"]})  '
          f'异常帧={summary_m1["n_abnormal"]}')

    output = {
        'folder':   folder,
        'has_mask': folder_mask is not None,
        'from1': {
            'summary':         summary_m1,
            'abnormal_frames': abnormal_m1,
            'frames':          list(frames_m1.values()),
        },
    }

    if frames_m0 is not None:
        summary_m0, abnormal_m0 = merge_and_judge(frames_m0, fb_map, args)
        thr0 = summary_m0['dynamic_thresholds']
        print(f'  from0  psnr_masked={summary_m0["psnr_masked"]["avg"]}  '
              f'(thr<{thr0["psnr_masked_thr"]})  '
              f'fb_epe_masked={summary_m0["fb_epe_masked"]["avg"]}  '
              f'(thr>{thr0["fb_epe_masked_thr"]})  '
              f'异常帧={summary_m0["n_abnormal"]}')
        output['from0'] = {
            'summary':         summary_m0,
            'abnormal_frames': abnormal_m0,
            'frames':          list(frames_m0.values()),
        }

    save_metrics(save_base, output)

    all_abnormal = abnormal_m1 + (abnormal_m0 if frames_m0 else [])
    if all_abnormal:
        print(f'\n  ⚠ 共发现 {len(all_abnormal)} 个异常帧：')
        for rec in all_abnormal[:10]:
            print(f'    [{rec["index"]:04d}] {rec["frame"]}  {rec["abnormal_reasons"]}')
        if len(all_abnormal) > 10:
            print(f'    ... 更多见 metrics.json')


# ── 主函数 ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--root', required=True, help='根目录')
    parser.add_argument('--hdr',   action='store_true', help='使用 HDR 图像')
    parser.add_argument('--debug', action='store_true', help='Debug 模式')
    parser.add_argument('--k_sigma', default=2.0, type=float,
                        help='异常判定的标准差倍数 k，默认 2.0\n'
                             '  psnr_masked  < mean - k*std → 异常\n'
                             '  fb_epe_masked > mean + k*std → 异常')
    args = parser.parse_args()

    folders = find_valid_folders(args.path)
    if not folders:
        print(f'[ERROR] 未找到包含 image + mv1 的子文件夹: {args.path}')
        return

    print(f'找到 {len(folders)} 个有效文件夹  k_sigma={args.k_sigma}')
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