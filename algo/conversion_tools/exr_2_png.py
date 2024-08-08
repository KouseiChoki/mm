'''
Author: Qing Hong
Date: 2022-09-26 15:01:24
LastEditors: Qing Hong
LastEditTime: 2024-08-07 10:42:35
Description: file content
'''
import os,sys
from tqdm import tqdm
import numpy as np 
import cv2
import re
import shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/..')
from file_utils import read
def mkdir(path):
        if  not os.path.exists(path):
            os.makedirs(path,exist_ok=True)

def jhelp(c):
    return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]

def hdr_to_rgb(hdr_image):
    # 对HDR图像进行色调映射
    # tonemap = cv2.createTonemapReinhard(1.0, 0, 0, 0)
    # ldr_image = tonemap.process(np.ascontiguousarray(hdr_image.copy()[...,:3]))
    # 将[0, 1]范围的图像转换为[0, 255]
    ldr_image_8bit = np.clip(hdr_image * 255, 0, 255).astype('uint8')
    ldr_image = adjust_gamma(ldr_image_8bit)
    # 保存转换后的图像
    return ldr_image[...,:3]


def adjust_gamma(image, gamma=2.4):
    # 建立一个映射表
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    
    # 应用gamma校正使用查找表
    return cv2.LUT(image, table)

if __name__ == '__main__':
    # a = [os.path.join(root,i) for i in sorted(list(filter(lambda x:x[0]!='.',os.listdir(root))))]
    assert len(sys.argv)==3 ,'usage: python exr_2_png.py root save_path'
    root = sys.argv[1]
    save_path = sys.argv[2]
    mkdir(save_path)
    imgs = jhelp(root)
    for index in tqdm(range(len(imgs))):
        img =imgs[index]
        hdr_image = read(img)[...,:3]
        cimage = hdr_to_rgb(hdr_image)
        sp = os.path.join(save_path,os.path.basename(img)).replace('.exr','.png')
        cv2.imwrite(sp,cimage[...,::-1])

    
    