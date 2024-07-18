'''
Author: Qing Hong
Date: 2022-09-26 15:01:24
LastEditors: QingHong
LastEditTime: 2023-05-30 10:39:09
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

def core(path):
    flos = jhelp(path)
    for i in tqdm(range(len(flos))):
        flo = imageio.imread(flos[i])
        imageio.imwrite(flos[i],flo,flags=imageio.plugins.freeimage.IO_FLAGS.EXR_ZIP|imageio.plugins.freeimage.IO_FLAGS.EXR_FLOAT)


if __name__ == '__main__':
    # a = [os.path.join(root,i) for i in sorted(list(filter(lambda x:x[0]!='.',os.listdir(root))))]
    assert len(sys.argv)==3 ,'usage: python mode root,mode=1:single,mode=2:multi'
    root = sys.argv[2]
    mode = sys.argv[1]
    if mode ==1: #single
        print('single file mode')
        files = jhelp(root)
        for file in files:
            if os.path.isdir(file):
                core(file)
    else:
        print('multi file mode')
        scenes = jhelp(root)
        for scene in scenes:
            if os.path.isdir(scene):
                files = jhelp(scene)
                for file in files:
                    if os.path.isdir(file):
                        core(file)

    
    