'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2025-01-17 12:10:58
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
import numpy as np 
import cv2
import os
from tqdm import tqdm
# 读取 MP4 视频并将帧转换为 NumPy 数组
def video_to_numpy(video_path):
    # 创建视频捕捉对象
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # 视频读取结束
        
        # 将视频帧转换为 NumPy 数组并存储到列表中
        frames.append(frame)

    # 释放视频捕获对象
    cap.release()

    # 将所有帧转换为一个 NumPy 数组（N帧, 高度, 宽度, 通道数）
    return np.array(frames)

def concat_image(data,x=True,desc=None,target_h=None,target_w=None,resize_rate=None):
    num_of_image,h,w,_ = data.shape
    if desc is not None:
        for i in range(num_of_image):
            cv2.putText(data[i], desc[i], org=(0,h), fontFace= cv2.FONT_HERSHEY_SIMPLEX, 
                      fontScale=4, color=(0, 0, 255), thickness=2)
    if not x:
        tmp = np.concatenate(data,axis=0)
    else:
        tmp = np.concatenate(data,axis=1)

    if target_h is not None and target_w is not None:
        tmp = cv2.resize(tmp,(target_w,target_h))
    elif resize_rate is not None:
        tmp = cv2.resize(tmp,None,fx=1/resize_rate,fy=1/resize_rate)
    return tmp

def write_video(sp,data,fps=24):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式
     # video = cv2.VideoWriter('results/result.avi',fourcc, cap_fps, size, isColor=0)
    video = None
    for index in tqdm(range(len(data)),desc=f'writing video{os.path.basename(sp)}'):
        imgs = data[index]
        s = imgs[0]
        # img2 = cv2.resize(img1,None,fx=2,fy=2)
        if video ==None:
            size = (s.shape[1],s.shape[0])
            video = cv2.VideoWriter(sp, fourcc, fps, size)#设置保存视频的名称和路径，默认在根目录下
            # imageio.mimwrite(os.path.join(target, 'result.mp4'), comp_frames, fps=30, quality=8)
        video.write(s)
    video.release()