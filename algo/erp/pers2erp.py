import os
from pathlib import Path
import MP2E
from file_utils import read, write

# ---------- 参数配置 ----------
PARAM_MAP = {
    "FM":  [90, 0, 0],
    "FR":  [90, 45, 0],
    "FL":  [90, -45, 0],
    "FU":  [90, 0, 45],
    "FD":  [90, 0, -45],
    "BR":  [90, 135, 0],
    "BL":  [90, -135, 0],
    "BU":  [90, 0, 135],
    "BD":  [90, 0, -135],
}

ERP_HEIGHT = 2048
ERP_WIDTH = 2048

INPUT_ROOT = "/Users/qhong/Desktop/0420"
OUTPUT_ROOT = "/Users/qhong/Desktop/0420/p2e/flat"

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')


def find_corresponding_files(root):
    """
    递归扫描根目录，找到所有同时包含 image/、mv0/、mv1/ 子目录的文件夹，
    返回 (img_path, mv0_path, mv1_path, base_name) 列表。
    """
    triplets = []
    root_path = Path(root)

    # 遍历所有名为 "image" 的目录（任意深度）
    for img_dir in root_path.rglob("image"):
        if not img_dir.is_dir():
            continue
        parent = img_dir.parent
        mv0_dir = parent / "mv0"
        mv1_dir = parent / "mv1"
        if not (mv0_dir.is_dir() and mv1_dir.is_dir()):
            continue

        # 处理该 image 目录下的所有图片
        for ext in IMAGE_EXTS:
            for img_file in img_dir.glob(f"*{ext}"):
                stem = img_file.stem
                mv0_candidates = list(mv0_dir.glob(f"{stem}.*"))
                mv1_candidates = list(mv1_dir.glob(f"{stem}.*"))
                if mv0_candidates and mv1_candidates:
                    triplets.append((
                        str(img_file),
                        str(mv0_candidates[0]),
                        str(mv1_candidates[0]),
                        stem
                    ))
                else:
                    print(f"警告: {img_file} 缺少对应的 mv0/mv1 文件，跳过")
    return triplets


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def process_all():
    print("正在扫描输入目录...")
    triplets = find_corresponding_files(INPUT_ROOT)
    if not triplets:
        print("未找到任何可处理的文件组合，退出。")
        return
    print(f"找到 {len(triplets)} 组待处理文件。")

    for img_path, mv0_path, mv1_path, base_name in triplets:
        print(f"\n处理: {base_name} ({img_path})")
        mv0_data = read(mv0_path)[..., :2]
        mv1_data = read(mv1_path)[..., :2]

        for param_name, (fov, theta, phi) in PARAM_MAP.items():
            print(f"  参数组 {param_name} (FOV={fov}, THETA={theta}, PHI={phi})")
            equ = MP2E.MPerspective(
                [[img_path, mv0_data, mv1_data]],
                [[fov, theta, phi]]
            )
            img_erp, mask_erp, mv0_erp, mv1_erp = equ.GetEquirec(ERP_HEIGHT, ERP_WIDTH)

            # 保存结果
            out_subdir = Path(OUTPUT_ROOT) / param_name
            out_img_dir = out_subdir / "img"
            out_mask_dir = out_subdir / "mask"
            out_mv0_dir = out_subdir / "mv0"
            out_mv1_dir = out_subdir / "mv1"
            ensure_dir(out_img_dir)
            ensure_dir(out_mask_dir)
            ensure_dir(out_mv0_dir)
            ensure_dir(out_mv1_dir)

            write(str(out_img_dir / f"{base_name}.png"), img_erp[..., ::-1])
            write(str(out_mask_dir / f"{base_name}.png"), mask_erp)
            write(str(out_mv0_dir / f"{base_name}.exr"), mv0_erp)
            write(str(out_mv1_dir / f"{base_name}.exr"), mv1_erp)

            print(f"    已保存至 {out_subdir}")

    print("\n所有任务完成！")


if __name__ == "__main__":
    process_all()