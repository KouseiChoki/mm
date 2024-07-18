


'''
Author: Qing Hong
Date: 2023-08-29 14:07:02
LastEditors: QingHong
LastEditTime: 2023-08-29 14:43:08
Description: file content
'''
'''
Author: Qing Hong
Date: 2023-08-03 14:32:52
LastEditors: QingHong
LastEditTime: 2023-08-29 14:23:25
Description: file content
'''
import sys,os
import numpy as np
import cv2
from tqdm import tqdm
def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))

def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

def motion_blur(img,size=15):
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    return cv2.filter2D(img, -1, kernel_motion_blur)

if __name__ == '__main__':
    assert len(sys.argv)==4 ,'usage: python xxx.py mv_path save_root size'
    root = sys.argv[1]
    save_path = sys.argv[2] 
    mkdir(save_path)
    imgs = jhelp_file(root)
    size = int(sys.argv[3])
    
    # 应用运动模糊内核
    for i in tqdm(range(len(imgs))):
        img = cv2.imread(imgs[i])
        img = motion_blur(img,size)
        cv2.imwrite(os.path.join(save_path,os.path.basename(imgs[i])),img)