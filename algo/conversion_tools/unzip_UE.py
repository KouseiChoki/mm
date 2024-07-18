
'''
Author: Qing Hong
Date: 2023-08-24 11:25:08
LastEditors: QingHong
LastEditTime: 2023-10-16 15:45:41
Description: file content
'''
import os,sys
import numpy as np
import Imath,OpenEXR,array
pt = Imath.PixelType(Imath.PixelType.FLOAT)
def exr_imread(filePath):
    # pt = Imath.PixelType(Imath.PixelType.HALF)
    img_exr = OpenEXR.InputFile(filePath)
    dw = img_exr.header()['dataWindow']
    size = (dw.max.y - dw.min.y + 1,dw.max.x - dw.min.x + 1)
    r_str, g_str, b_str ,a_str= img_exr.channels('RGBA',pt)
    red = np.array(array.array('f', r_str))
    green = np.array(array.array('f', g_str))
    blue = np.array(array.array('f', b_str))
    alpha = np.array(array.array('f', a_str))
    red = red.reshape(size)
    green = green.reshape(size)
    blue = blue.reshape(size)
    alpha = alpha.reshape(size)
    image = np.stack([red,green,blue,alpha],axis=2)
    return image.astype('float32')

def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))
if __name__=='__main__':
    assert len(sys.argv)==2,'usage python unzip_UE.py root'
    source_root = sys.argv[1]
    source_root = '/Volumes/hongqing/915final'
    scenes = jhelp_folder(source_root)
    for scene in scenes:
        for fps in jhelp_folder(scene):
            if 'image' in jhelp_folder(fps):
                continue
            exrs = jhelp_file(fps)
             
    