'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-09-24 12:18:22
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

