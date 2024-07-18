'''
Author: Qing Hong
Date: 2022-09-26 15:01:24
LastEditors: Qing Hong
LastEditTime: 2024-05-28 15:32:39
Description: file content
'''
import os,sys
sys.path.append('..')
from tqdm import tqdm
import cv2
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
    assert len(sys.argv)>3,'usage python resize.py root output_root 2k imagetype(image or mask)'
    root = sys.argv[1]
    save_path = sys.argv[2]
    size = sys.argv[3]
    imgtype = sys.argv[4]
    imgs = jhelp(root)
    mkdir(save_path)
    for i in tqdm(range(len(imgs))):
        image = read(imgs[i],type=imgtype.lower())
        if size.lower() == '1k':
            image = cv2.resize(image,(960,540))
        elif size.lower() == '4k':
            image = cv2.resize(image,(3840,2160))
        elif size.lower() == 'trans':
            # image = cv2.resize(image,(3840,2160))
            pass
        else:
            image = cv2.resize(image,(864,512))
        write(os.path.join(save_path,os.path.basename(imgs[i])).replace('tif','png'),image)
        

    
    