'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-09-24 11:33:25
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
from video_util import video_to_numpy,concat_image
import sys,os
import numpy as np 
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/..')
from file_utils import mvwrite,read,jhelp_file,gofind,prune
from tqdm import tqdm
if __name__ == '__main__':
    assert len(sys.argv) ==2,'usage: python X root'
    root = sys.argv[1]
    outputname = 'demo.mp4'
    mp4s = prune(gofind(jhelp_file(root),keyword='.mp4'),outputname)
    nums_of_range = 0
    datas = []
    desc = []
    target_h=None
    target_w=None
    resize_rate = 1.7
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    cap_fps = 24
    size = (target_w,target_h)
    video = None
    for mp4 in mp4s:
        print(f'loading {os.path.basename(mp4)}...')
        tmp = video_to_numpy(mp4)
        if nums_of_range<tmp.shape[0]:
            nums_of_range = tmp.shape[0]
        datas.append(tmp)
        desc.append(os.path.basename(mp4).split('.')[0])
    for i in tqdm(range(nums_of_range),desc='processing'):
        frame = []
        for data in datas:
            if i>data.shape[0]-1:
                cy = data[-1]
            else:
                cy = data[i]
            frame.append(cy)
        output_frame = concat_image(np.stack(frame),x=False,desc=desc,target_h=target_h,target_w=target_w,resize_rate=resize_rate)
        if video ==None:
            size = (output_frame.shape[1],output_frame.shape[0])
            video = cv2.VideoWriter(os.path.join(root,outputname), fourcc, cap_fps, size)
        video.write(output_frame)
    video.release()
    print('finished..')


    