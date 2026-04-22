import os,sys
import subprocess
import tempfile
from pathlib import Path
import shutil
from file_utils import read,write,jhelp_file
from tqdm import tqdm
# ---------- 配置 ----------
# C++ 可执行文件路径
EXECUTABLE = os.path.dirname(os.path.abspath(__file__))+"/PWTCRayMapper" 

# 基础模板文件路径
BASE_PRM = os.path.dirname(os.path.abspath(__file__))+"/flat2erp_rgbmv_I2N.prm"

# 输出目录（存放临时 prm 和日志）
WORK_DIR = os.path.dirname(os.path.abspath(__file__))+"/tmp"
os.makedirs(WORK_DIR, exist_ok=True)

import re
from pathlib import Path

def extract_pattern_and_range(img_dir,rpass=1):
    """
    从 image 目录中的文件提取文件名模式、起始帧和总帧数。
    假设文件命名如 "prefix.0000.png" 或 "1.FinalImage.0000.png"
    返回 (strInputFilePattern, startFrame, totalFrames)
    """
    img_path = Path(img_dir)
    # 获取所有图片文件
    image_files = sorted([f for f in img_path.glob("*") if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    if not image_files:
        raise ValueError(f"No image files in {img_dir}")
    
    # 尝试从第一个文件名中提取前缀和序号格式
    first_name = image_files[0].stem  # 不带扩展名
    # 查找末尾的数字部分
    match = re.search(r'^(.*?)(\d+)$', first_name)
    if not match:
        raise ValueError(f"Cannot parse number from filename: {first_name}")
    prefix = match.group(1)
    num_str = match.group(2)
    num_digits = len(num_str)
    
    # 构建 printf 格式模式
    pattern = f"{prefix}%0{num_digits}d.png"
    
    # 获取所有序号
    numbers = []
    for f in image_files:
        m = re.search(r'(\d+)$', f.stem)
        if m:
            numbers.append(int(m.group(1)))
    numbers.sort()
    start_frame = numbers[0]
    total_frames = len(numbers)
    
    return pattern, start_frame, total_frames-rpass

def extract_mv_pattern(mv_dir, mv_type='mv0'):
    """
    从 mv 目录提取文件名模式，假设 mv 文件命名如 "mv_00000000.exr"
    返回 strInputMv0FilePattern 或类似
    """
    mv_path = Path(mv_dir)
    exr_files = sorted(mv_path.glob("*.exr"))
    if not exr_files:
        # 如果没有 .exr 可能 .flo
        exr_files = sorted(mv_path.glob("*.flo"))
    if not exr_files:
        raise ValueError(f"No motion vector files in {mv_dir}")
    
    first_name = exr_files[0].stem
    # 同样提取前缀和数字部分
    match = re.search(r'^(.*?)(\d+)$', first_name)
    if not match:
        raise ValueError(f"Cannot parse number from mv filename: {first_name}")
    prefix = match.group(1)
    num_digits = len(match.group(2))
    ext = exr_files[0].suffix
    pattern = f"{prefix}%0{num_digits}d{ext}"
    return pattern

def generate_prm(template_path, output_path, overrides):
    """
    读取模板文件，替换指定键的值，生成新的 prm 文件。
    """
    with open(template_path, 'r') as f:
        content = f.read()

    lines = content.splitlines()
    new_lines = []
    for line in lines:
        # 跳过注释和空行
        stripped = line.strip()
        if not stripped or stripped.startswith('//') or stripped.startswith('#'):
            new_lines.append(line)
            continue

        if '=' in stripped:
            key, val = stripped.split('=', 1)
            key = key.strip()
            if key in overrides:
                new_val = overrides[key]
                new_lines.append(f"{key}={new_val}")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    with open(output_path, 'w') as f:
        f.write('\n'.join(new_lines))


def run_task(prm_path, task_name):
    """
    调用 C++ 程序处理单个 prm 文件。
    """
    cmd = [EXECUTABLE, prm_path]
    # print(f"[{task_name}] 开始运行: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"[{task_name}] 成功完成")
        if result.stdout:
            print(f"[{task_name}] stdout:\n{result.stdout}")
        if result.stderr:
            print(f"[{task_name}] stderr:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"[{task_name}] 运行失败，返回码 {e.returncode}")
        print(f"[{task_name}] stdout:\n{e.stdout}")
        print(f"[{task_name}] stderr:\n{e.stderr}")
        raise


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
        triplets.append((img_dir,mv0_dir,mv1_dir))
    return triplets

def main():

    assert len(sys.argv)>=2 ,'usage: python batch_ray_mapper.py root (output)'
    root = sys.argv[1]
    output = None if len(sys.argv)==2 else sys.argv[2]
    print("正在扫描输入目录...")
    triplets = find_corresponding_files(root)
    if not triplets:
        print("未找到任何可处理的文件组合，退出。")
        return
    print(f"找到 {len(triplets)} 组待处理文件。")

    for idx in tqdm(range(len(triplets))):
        img_dir,mv0_dir,mv1_dir = triplets[idx]
        
        # 提取图片模式与帧范围
        img_pattern, start_frame, total_frames = extract_pattern_and_range(img_dir)
        mv0_pattern = extract_mv_pattern(mv0_dir)
        mv1_pattern = extract_mv_pattern(mv1_dir)
        
        # 输出路径可以基于原目录生成，例如在父目录下创建 flat 子目录
        parent_dir = Path(img_dir).parent
        output_dir = parent_dir / "flat" if not output else Path(os.path.join(output,os.path.basename(parent_dir)))
        output_dir.mkdir(exist_ok=True)
        
        overrides = {
            "strInputPath": str(img_dir),
            "strInputFilePattern": img_pattern,
            "strInputMv0Path": str(mv0_dir),
            "strInputMv0FilePattern": mv0_pattern,
            "strInputMv1Path": str(mv1_dir),
            "strInputMv1FilePattern": mv1_pattern,
            "strOutputPath": str(output_dir),
            "strOutputFilePattern": f"{Path(img_pattern).stem}_Erp.png",  # 根据需求调整
            "strOutputMv0FilePattern": "mv_%08d.exr",
            "strOutputMv1FilePattern": "mv_%08d.exr",
            "strOutputMaskFilePattern": "mask_%08d.exr",
            "startFrame": str(start_frame),
            "totalFrames": str(total_frames),
            "outputNoExrBpc":str(start_frame),
            "strFlatViewsFile":str(os.path.dirname(os.path.abspath(__file__))+'/FlatViews.txt')
        }

        task_name = f"task_{idx:03d}"
        prm_path = os.path.join(WORK_DIR, f"{task_name}.prm")

        # 生成临时 prm
        generate_prm(BASE_PRM, prm_path, overrides)

        # 执行
        run_task(prm_path, task_name)

        # （可选）保留 prm 文件以便调试，或删除
        # os.remove(prm_path)

    print("所有批量任务完成！")


if __name__ == "__main__":
    main()