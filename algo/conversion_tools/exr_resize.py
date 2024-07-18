'''
Author: Qing Hong
Date: 2022-09-26 15:01:24
LastEditors: QingHong
LastEditTime: 2023-03-08 13:02:18
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


if __name__ == '__main__':
    # a = [os.path.join(root,i) for i in sorted(list(filter(lambda x:x[0]!='.',os.listdir(root))))]
    # assert len(sys.argv)==3 ,'usage: python exr_resize_mergedepth.py root save_path'
    # root = sys.argv[1]
    # save_path = sys.argv[2]
    root = '/Users/qhong/Downloads/GMA'
    save_path = '/Users/qhong/Downloads/GMA_4k'
    sequences = jhelp(root)
    for index in tqdm(range(len(sequences))):
        seq = sequences[index]
        mkdir(os.path.join(save_path,os.path.basename(seq)))
        imgs = jhelp(seq)
        for id,img in enumerate(imgs):
            flo = imageio.imread(img).astype('float32')
            flo = cv2.resize(flo,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
            sp = os.path.join(save_path,os.path.basename(seq),os.path.basename(img))
            imageio.imwrite(sp,flo,flags=imageio.plugins.freeimage.IO_FLAGS.EXR_ZIP|imageio.plugins.freeimage.IO_FLAGS.EXR_FLOAT)
            #copy first frame
            frame = re.findall(r'\d+',sp)[-1]
            if id == 0 and '_mv1' in seq:
                shutil.copy(sp,sp.replace(frame,'{:0>8}').format(int(frame)-1))        
            #copy last frame
            if id == len(imgs)-1 and '_mv0' in seq:
                shutil.copy(sp,sp.replace(frame,'{:0>8}').format((int(frame)+1)))

    
    