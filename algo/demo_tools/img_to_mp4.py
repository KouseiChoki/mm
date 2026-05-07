'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2026-05-06 14:14:13
Description: 
         ▄              ▄
        ▌▒█           ▄▀▒▌     
        ▌▒▒▀▄       ▄▀▒▒▒▐
       ▐▄▀▒▒▀▀▀▀▄▄▄▀▒▒▒▒▒▐     ,-----------------.
     ▄▄▀▒▒▒▒▒▒▒▒▒▒▒█▒▒▄█▒▐     (Wow,kousei's code)
   ▄▀▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▀██▀▒▐     `-,---------------' 
  ▐▒▒▒▄▄▄▒▒▒▒▒▒▒▒▒▒▒▒▒▀▄▒▒▌  _.-'   ,----------.
  ▌▒▒▐▄█▀▒▒▒▒▄▀█▄▒▒▒▒▒▒▒█▒▐         (surabashii)
 ▐▒▒▒▒▒▒▒▒▒▒▒▀██▀▒▒▒▒▒▒▒▒▀▄▌        `-,--------' 
 ▌▒▀▄██▄▒▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒▌      _.-'
 ▌▀▐▄█▄█▌▄▒▀▒▒▒▒▒▒░░░░░░▒▒▒▐ _.-'
▐▒▀▐▀▐▀▒▒▄▄▒▄▒▒▒▒▒░░░░░░▒▒▒▒▌
▐▒▒▒▀▀▄▄▒▒▒▄▒▒▒▒▒▒░░░░░░▒▒▒▐
 ▌▒▒▒▒▒▒▀▀▀▒▒▒▒▒▒▒▒░░░░▒▒▒▒▌
 ▐▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▐
  ▀▄▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▄▒▒▒▒▌
    ▀▄▒▒▒▒▒▒▒▒▒▒▄▄▄▀▒▒▒▒▄▀
      ▀▄▄▄▄▄▄▀▀▀▒▒▒▒▒▄▄▀
         ▒▒▒▒▒▒▒▒▒▒▀▀
When I wrote this, only God and I understood what I was doing
Now, God only knows
'''
import os,sys
import cv2
import argparse
from natsort import natsorted
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/..')
from file_utils import read

def images_to_video(input_dir, output_video, fps=24, resize=None,passone=1):
    """
    将文件夹中的图片合成为 MP4 视频。

    参数:
        input_dir: 包含图片的文件夹路径
        output_video: 输出视频文件路径（如 'output.mp4'）
        fps: 视频帧率
        resize: (宽度, 高度) 元组，可选。若不提供，则使用第一张图片的尺寸
    """
    # 支持的图片扩展名
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif','.exr')

    # 获取文件夹下所有图片文件并自然排序
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(img_extensions)]
    if not img_files:
        print("错误：未找到任何图片文件。")
        return False

    img_files = natsorted(img_files)[passone:]  # 自然排序，如 frame1, frame2, ..., frame10

    # 读取第一张图片以获取尺寸
    first_img_path = os.path.join(input_dir, img_files[0])
    first_img = read(first_img_path,type='image')
    if first_img is None:
        print(f"错误：无法读取图片 {first_img_path}")
        return False

    h, w = first_img.shape[:2]
    if resize:
        w, h = resize
        print(f"将调整所有图片尺寸为: {w} x {h}")

    # 定义视频编码器和创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 编码
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    print(f"开始处理 {len(img_files)} 张图片...")
    for idx, filename in enumerate(img_files, 1):
        img_path = os.path.join(input_dir, filename)
        img = read(img_path,type='image')
        if img is None:
            print(f"警告：跳过无法读取的图片 {filename}")
            continue

        # 调整尺寸
        if resize:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        else:
            # 如果图片尺寸与第一张不同，也强制调整到第一张的尺寸
            if (img.shape[1], img.shape[0]) != (w, h):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        out.write(img[...,::-1])
        if idx % 10 == 0 or idx == len(img_files):
            print(f"已处理 {idx}/{len(img_files)} 帧")

    out.release()
    print(f"视频已保存至: {output_video}")
    return True
# python /Users/qhong/Documents/1117test/MM/mm_NEW/mm/algo/demo_tools/img_to_mp4.py --input_dir /Users/qhong/Desktop/0429/vfi/0402/VFIMamba -o /Users/qhong/Desktop/0429/vfi/0402/VFIMamba.mp4
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将文件夹中的图片合成为 MP4 视频")
    parser.add_argument("--input_dir", help="包含图片的文件夹路径")
    parser.add_argument("-o", "--output", default="/Users/qhong/Desktop/0506/tmp.mp4", help="输出视频文件路径 (默认: output.mp4)")
    parser.add_argument("--fps", type=int, default=24, help="视频帧率 (默认: 24)")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        help="调整所有图片尺寸到指定宽高，例如 --resize 1920 1080")
    parser.add_argument("--passone", type=int, default=1,help="跳过第一张图片")

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"错误：目录 '{args.input_dir}' 不存在。")
    else:
        output = args.output
        # output = os.path.join(args.output,os.path.basename(args.input_dir)+'.mp4')
        images_to_video(args.input_dir, output, args.fps, args.resize,args.passone)