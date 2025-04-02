'''
Author: Qing Hong
Date: 2022-09-26 15:01:24
LastEditors: Qing Hong
LastEditTime: 2025-03-12 10:12:12
Description: file content
'''
import os,sys
sys.path.append('../..')
from tqdm import tqdm
import cv2
import numpy as np
from file_utils import read,write
def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))


if __name__ == '__main__':
    assert len(sys.argv)==2,'usage python XXXX.py root '
    root = sys.argv[1]
    imgs = jhelp(root)
    save_path = os.path.join(os.path.dirname(root),'shadow_mask')
    mkdir(save_path)
    for i in tqdm(range(len(imgs))):
        image = read(imgs[i],type='image')
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        tmp = (hsv[:, :, 2] < 80) & (hsv[:, :, 2] > 50)
        shadow_mask = np.zeros_like(image)
        shadow_mask[tmp] = 255
        # shadow_mask = cv2.cvtColor(shadow_mask.astype(np.uint8), cv2.COLOR_HSV2BGR)
        write(os.path.join(save_path,os.path.basename(imgs[i])).replace('tif','png'),shadow_mask)
        

    
    