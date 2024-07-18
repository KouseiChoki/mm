'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-06-18 15:10:12
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
import cv2
import os,sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/../..')
from file_utils import write,jhelp_file,gofind
VIDEONAME = '.mov'
def video_process_core(video,sp):
    # 打开视频文件
    cap = cv2.VideoCapture(video)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 假设深度信息和图像信息在不同的通道上
        # 比如：RGB 通道存储颜色图像，第4个通道存储深度信息
        if frame.shape[2] == 4:  # 检查是否有4个通道
            color_image = frame[:, :, :3]  # 提取前3个通道作为颜色图像
            depth_image = frame[:, :, 3]   # 提取第4个通道作为深度图像
            # 保存或处理图像和深度信息
            write('{}/image/image_{:0>6}.jpg'.format(sp,frame_count), color_image)
            write('{}/world_depth/depth_{:0>6}.exr'.format(sp,frame_count), depth_image)
        frame_count += 1
    cap.release()

if __name__ == '__main__':
    assert len(sys.argv)==2 ,'usage: python exr_get_mv.py root(or video)'
    datas = [sys.argv[1]] if not os.path.isdir(sys.argv[1]) else gofind(jhelp_file(sys.argv[1]),VIDEONAME)
    for data in datas:
        sp = data.replace(VIDEONAME,'')
        video_process_core(data,sp)
        