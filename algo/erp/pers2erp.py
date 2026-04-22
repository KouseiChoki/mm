import os
import re
from pathlib import Path
import MP2E
from mm.algo.erp.ray_mapper.file_utils import read, write,jhelp_file
from tqdm import tqdm
# ---------- 配置 ----------
PARAM_MAP = {
    "RU":  [90, 30, 45],
    "RM":  [90, 45, 0],
    "RD":  [90, 30, -45],
    "FU":  [90, 0, 45],
    "FM":  [90, 0, 0],
    "FD":  [90, 0, -45],
    "LU":  [90, -30, 45],
    "LM":  [90, -45, 0],
    "LD":  [90, -30, -45],
}

ERP_HEIGHT = 4096
ERP_WIDTH = 4096

INPUT_ROOT = "/Users/qhong/Desktop/0422/24fps"
OUTPUT_ROOT = "/Users/qhong/Desktop/0422/erps"

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
MV_EXTS = ('.exr', '.flo')


def extract_last_number(filename: str) -> str:
    """
    提取文件名中最后一个连续数字串（作为序号）。
    例如 '1.FinalImage.0000.png' -> '0000'
    """
    numbers = re.findall(r'\d+', filename)
    return numbers[-1] if numbers else None


def find_file_by_number(directory: Path, number: str, exts: tuple) -> Path:
    """
    在 directory 中查找文件名包含给定 number 的文件。
    若找到唯一匹配则返回路径，否则返回 None。
    """
    candidates = []
    for ext in exts:
        candidates.extend(directory.glob(f"*{ext}"))
    matches = [f for f in candidates if number in f.name]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"  警告: 在 {directory} 中找到多个包含序号 {number} 的文件，跳过。")
    return None


def find_corresponding_files(root,passright=1):
    """
    递归扫描根目录，找到所有同时包含 image/、mv0/、mv1/ 子目录的文件夹，
    并通过序号匹配返回 (img_path, mv0_path, mv1_path, base_name) 列表。
    """
    triplets = []
    root_path = Path(root)

    for img_dir in root_path.rglob("image"):
        if not img_dir.is_dir():
            continue
        parent = img_dir.parent
        mv0_dir = parent / "mv0"
        mv1_dir = parent / "mv1"
        if not (mv0_dir.is_dir() and mv1_dir.is_dir()):
            continue

        # 遍历该 image 目录下的所有图片
        img_file = jhelp_file(img_dir)
        mv0_file = jhelp_file(mv0_dir)
        mv1_file = jhelp_file(mv1_dir)
        for index in range(len(img_file)-passright):
            triplets.append((
                        str(img_file[index]),
                        str(mv0_file[index]),
                        str(mv1_file[index]),
                        Path(img_file[index]).stem   # 保留原始图片主名作为输出基础名
                    ))
    return triplets

def mv_denormalize(mv,normalize=False):
    h,w,_ = mv.shape
    if normalize:
        mv[...,0] /= w
        mv[...,1] /= h
    else:
        mv[...,0] *= w
        mv[...,1] *= h
    return mv


def process_all():
    print("正在扫描输入目录...")
    triplets = find_corresponding_files(INPUT_ROOT)
    if not triplets:
        print("未找到任何可处理的文件组合，退出。")
        return
    print(f"找到 {len(triplets)} 组待处理文件。")

    for index in tqdm(range(len(triplets))):
        img_path, mv0_path, mv1_path, base_name = triplets[index]
        mv0_data = mv_denormalize(read(mv0_path)[..., :2])
        mv1_data = mv_denormalize(read(mv1_path)[..., :2])
        #denormalized
        

        for param_name, (fov, theta, phi) in PARAM_MAP.items():
            equ = MP2E.MPerspective(
                [[img_path, mv0_data, mv1_data]],
                [[fov, theta, phi]]
            )
            img_erp, mask_erp, mv0_erp, mv1_erp = equ.run(ERP_HEIGHT, ERP_WIDTH)

            out_subdir = Path(OUTPUT_ROOT) / param_name
            out_img_dir = out_subdir / "img"
            out_mask_dir = out_subdir / "mask"
            out_mv0_dir = out_subdir / "mv0"
            out_mv1_dir = out_subdir / "mv1"

            write(str(out_img_dir / f"{base_name}.png"), img_erp[..., ::-1])
            write(str(out_mask_dir / f"{base_name}.png"), mask_erp)
            write(str(out_mv0_dir / f"{base_name}.exr"), mv_denormalize(mv0_erp,normalize=True))
            write(str(out_mv1_dir / f"{base_name}.exr"), mv_denormalize(mv1_erp,normalize=True))

    print("\n所有任务完成！")


if __name__ == "__main__":
    process_all()