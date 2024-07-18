'''
Author: Qing Hong
Date: 2022-09-26 15:01:24
LastEditors: QingHong
LastEditTime: 2023-11-20 17:32:16
Description: file content
'''
import os,sys
import imageio
from tqdm import tqdm
import numpy as np 
import cv2
import re
import shutil
def mkdir(path):
        if  not os.path.exists(path):
            os.makedirs(path,exist_ok=True)

def jhelp(c):
    return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]

def hdr_to_rgb(hdr_image):
    # 对HDR图像进行色调映射
    tonemap = cv2.createTonemapReinhard(1.0, 0, 0, 0)
    ldr_image = tonemap.process(hdr_image.copy())
    
    # 将[0, 1]范围的图像转换为[0, 255]
    ldr_image_8bit = np.clip(ldr_image * 255, 0, 255).astype('uint8')
    
    # 保存转换后的图像
    return ldr_image_8bit

if __name__ == '__main__':
    # a = [os.path.join(root,i) for i in sorted(list(filter(lambda x:x[0]!='.',os.listdir(root))))]
    assert len(sys.argv)==3 ,'usage: python exr_2_png.py root save_path'
    root = sys.argv[1]
    save_path = sys.argv[2]
    mkdir(save_path)
    imgs = jhelp(root)
    for index in tqdm(range(len(imgs))):
        img =imgs[index]
        hdr_image = imageio.imread(img)[...,:3]
        cimage = hdr_to_rgb(hdr_image)
        sp = os.path.join(save_path,os.path.basename(img)).replace('.exr','.png')
        cv2.imwrite(sp,cimage[...,::-1])

    
    