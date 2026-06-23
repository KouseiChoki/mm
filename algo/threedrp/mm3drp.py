import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from myutil import immc, jhelp_folder, jhelp_file, write, read
import argparse
from tqdm import tqdm


# ── 工具函数 ──────────────────────────────────────────────────────────
def find_valid_folders(root_dir):
    """扫描所有包含 image 和 mv1 子文件夹的目录"""
    valid = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        has_image = any(d in ('image', 'video') for d in dirnames)
        has_mv1   = any('mv1' in d for d in dirnames)
        if has_image and has_mv1:
            valid.append(dirpath)
    return valid


def get_mv_name(folder, keyword):
    """在 folder 的子文件夹中找到包含 keyword 的目录名"""
    for d in jhelp_folder(folder):
        if keyword in d and keyword != d:
            return d
    return keyword


# ── 核心处理 ──────────────────────────────────────────────────────────
def run_immc(folder_image, folder_mv, save_path, args, direction):
    """
    对一个方向（mv0 或 mv1）做 immc，结果存入 save_path。
    direction: 'from0'（用 mv0 把 i+1 帧 warp 到 i）
               'from1'（用 mv1 把 i-1 帧 warp 到 i）
    """
    os.makedirs(save_path, exist_ok=True)
    dtype = 'hdr' if args.hdr else 'image'
    n = len(folder_image)

    for i in tqdm(range(n), desc=f'{os.path.basename(save_path)}'):
        mv   = read(folder_mv[i], 'flo')
        h, w = mv.shape[:2]
        mv[..., 0] *= w
        mv[..., 1] *= h

        if direction == 'from0':
            # mv0：把第 i+1 帧 warp 到第 i 帧位置
            src_idx = i + 1
            ref_idx = i - 1 if i > 0 else i + 1
        else:
            # mv1：把第 i-1 帧 warp 到第 i 帧位置
            src_idx = i - 1
            ref_idx = i + 1 if i < n - 1 else i - 1

        valid = (direction == 'from0' and i < n - 1) or \
                (direction == 'from1' and i > 0)

        if valid:
            src_img = read(folder_image[src_idx], type=dtype)
            out     = immc(src_img, mv)
        else:
            # 边界帧：用对侧参考帧填零占位
            ref_img = read(folder_image[ref_idx], type=dtype)
            out     = np.zeros_like(ref_img)

        sp = os.path.join(save_path, os.path.basename(folder_image[i]))
        write(sp, out)


def process_folder(folder, args):
    dtype       = 'hdr' if args.hdr else 'image'
    image_name  = 'video' if args.hdr else 'image'

    # 找 image 文件夹
    img_dir = os.path.join(folder, image_name)
    if not os.path.isdir(img_dir):
        img_dir = folder
    folder_image = jhelp_file(img_dir)
    if not folder_image:
        print(f'[SKIP] 无图像文件: {folder}')
        return

    # 找 mv1（必须有）
    mv1_name    = get_mv_name(folder, 'mv1')
    mv1_dir     = os.path.join(folder, mv1_name)
    folder_mv1  = jhelp_file(mv1_dir) if os.path.isdir(mv1_dir) else None
    if not folder_mv1:
        print(f'[SKIP] 找不到 mv1: {folder}')
        return

    # 找 mv0（可选）
    mv0_name    = get_mv_name(folder, 'mv0')
    mv0_dir     = os.path.join(folder, mv0_name)
    folder_mv0  = jhelp_file(mv0_dir) if os.path.isdir(mv0_dir) else None

    assert len(folder_image) == len(folder_mv1), \
        f'image({len(folder_image)}) 和 mv1({len(folder_mv1)}) 数量不一致: {folder}'
    if folder_mv0:
        assert len(folder_image) == len(folder_mv0), \
            f'image({len(folder_image)}) 和 mv0({len(folder_mv0)}) 数量不一致: {folder}'

    save_base = os.path.join(folder, 'immc')

    # mv1 → from1
    run_immc(folder_image, folder_mv1,
             os.path.join(save_base, 'from1'), args, direction='from1')

    # mv0 → from0（若有）
    if folder_mv0:
        run_immc(folder_image, folder_mv0,
                 os.path.join(save_base, 'from0'), args, direction='from0')
    else:
        print(f'[INFO] 无 mv0，跳过 from0: {folder}')


# ── 主函数 ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path','--root' , required=True,      help='根目录')
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