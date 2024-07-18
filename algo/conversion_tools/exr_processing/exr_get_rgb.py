'''
Author: Qing Hong
Date: 2023-06-29 11:27:43
LastEditors: QingHong
LastEditTime: 2024-01-12 16:42:13
Description: file content
'''
import numpy as np
import OpenEXR, Imath, array
import os,sys
import cv2
from tqdm import tqdm
pt = Imath.PixelType(Imath.PixelType.FLOAT)
'''
description: 读取exr文件
param {*} filePath 文件地址
return {*} 返回rgb图像，mv1，mask
'''
def get_channel_data(img_exr,keyword,type='f'):
    data = np.array(array.array(type, img_exr.channels(keyword,pt)))
    return data


def exr_imread(filePath):
    img_exr = OpenEXR.InputFile(filePath)
    dw = img_exr.header()['dataWindow']
    size = (dw.max.y - dw.min.y + 1,dw.max.x - dw.min.x + 1)
    # 获取rgb
    r_str, g_str, b_str = img_exr.channels('RGB',pt)
    red = np.array(array.array('f', r_str))
    green = np.array(array.array('f', g_str))
    blue = np.array(array.array('f', b_str))
    red = red.reshape(size)
    green = green.reshape(size)
    blue = blue.reshape(size)
    image = np.stack([red,green,blue],axis=2).clip(0,1)*255
    return image.astype('uint8')

def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]



def write_flo_file(flow, filename): # flow: H x W x 2
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    if flow.ndim == 4: # has batch
        flow = flow[0]

    outpath = os.path.dirname(filename)
    if outpath != '' and not os.path.isdir(outpath):
        os.makedirs(outpath)

    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    height, width = flow.shape[:2]
    magic.tofile(f)
    np.int32(width).tofile(f)
    np.int32(height).tofile(f)
    data = np.float32(flow).flatten()
    data.tofile(f)
    f.close()

if __name__=='__main__':
    assert len(sys.argv)==3,'usage python exr_get_rgb.py root output_root'
    source_root = sys.argv[1]
    save_path = sys.argv[2]
    resize_rate = 1
    half = False
    sources = jhelp(source_root)
    image_save_path = os.path.join(save_path,'image')

    mkdir(image_save_path)

    for i in tqdm(range(len(sources))):
        source = sources[i]
        image = exr_imread(source)
        if half:
            image = cv2.resize(image,None,fx=resize_rate,fy=resize_rate)
        name = os.path.basename(source).replace('.exr','')
        cv2.imwrite(os.path.join(image_save_path,name)+'.png',image[...,::-1])

    
