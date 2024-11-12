'''
Author: Qing Hong
Date: 2023-01-11 13:13:30
LastEditors: Qing Hong
LastEditTime: 2024-11-06 10:15:02
Description: file content
'''
import cv2
import os
from tqdm import tqdm
import numpy as np
import sys
def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]


fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式
# cap_fps是帧率，根据自己需求设置帧率
cap_fps = 24

# imgs1 = jhelp('/Users/qhong/Downloads/0615_inpainting/inpainting_result/image_512_864')
# imgs2 = jhelp('/Users/qhong/Downloads/0615_inpainting/inpainting_result/inpaint_512_864')
# imgs3 = jhelp('/Users/qhong/Downloads/0615_inpainting/inpainting_result/inpaint_MM_512_864')

if __name__ == '__main__':
    assert len(sys.argv) ==2,'usage: python X root'
    root = sys.argv[1]

    imgs1 = jhelp(root)
    target = os.path.dirname(root)
    name = os.path.basename(root)+'.mp4'
    # size要和图片的size一样，但是通过img.shape得到图像的参数是（height，width，channel），
    # 可以实现在图片文件夹下查看图片属性，获得图片的分辨率
    # 设置输出视频的参数，如果是灰度图，可以加上 isColor = 0 这个参数
    # video = cv2.VideoWriter('results/result.avi',fourcc, cap_fps, size, isColor=0)
    video = None
    
    for index in tqdm(range(len(imgs1))):
        s = cv2.imread(imgs1[index])
        # img2 = cv2.resize(img1,None,fx=2,fy=2)
        # img2 = cv2.resize(img2,None,fx=2,fy=2)
        # img3 = cv2.resize(img3,None,fx=2,fy=2)
        # if img1.shape != img2.shape:
            # img1 = cv2.resize(img1,(img2.shape[1],img2.shape[0]))
        # s = np.concatenate((img1,img2,img3),axis=1)

        if video ==None:
            size = (s.shape[1],s.shape[0])
            video = cv2.VideoWriter(os.path.join(target,name), fourcc, cap_fps, size)#设置保存视频的名称和路径，默认在根目录下
            # imageio.mimwrite(os.path.join(target, 'result.mp4'), comp_frames, fps=30, quality=8)
        video.write(s)
    video.release()
            