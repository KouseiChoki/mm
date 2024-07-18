'''
Author: Qing Hong
Date: 2023-08-29 14:07:02
LastEditors: QingHong
LastEditTime: 2023-08-29 14:31:59
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
import OpenEXR
import numpy as np
import Imath,array
import imageio
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

pt = Imath.PixelType(Imath.PixelType.FLOAT)

def exr_imread(filePath):
    img_exr = OpenEXR.InputFile(filePath)
    dw = img_exr.header()['dataWindow']
    size = (dw.max.y - dw.min.y + 1,dw.max.x - dw.min.x + 1)
    def getchan(name):
        result = img_exr.channel(name,pt)
        return np.array(array.array('f', result)).reshape(size)
    mv0x =getchan('MV0.x')
    mv0y =getchan('MV0.y')
    mv1x =getchan('MV1.x')
    mv1y =getchan('MV1.y')
    mv0 = np.concatenate((mv0x[...,None],mv0y[...,None]),axis=-1)
    mv1 = np.concatenate((mv1x[...,None],mv1y[...,None]),axis=-1)
    return mv0,mv1
   
def write(path,flow):
    if flow.shape[2] == 2:
        flow = np.insert(flow,2,0,axis=2)
    if flow.shape[2] == 3:
        flow = np.insert(flow,2,0,axis=2)
    imageio.imwrite(path,flow[...,:4],flags=imageio.plugins.freeimage.IO_FLAGS.EXR_ZIP|imageio.plugins.freeimage.IO_FLAGS.EXR_FLOAT)
if __name__ == '__main__':
    assert len(sys.argv)==3 or len(sys.argv)==2 ,'usage: python MM_decoder mv_path save_root'
    root = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv)==3 else os.path.abspath(os.path.join(root,'..'))
    sp0 = os.path.join(save_path,'mv0')
    sp1 = os.path.join(save_path,'mv1')
    mkdir(sp0),mkdir(sp1)
    imgs = jhelp_file(root)
    for i in tqdm(range(len(imgs))):
        img = imgs[i]
        mv0,mv1 = exr_imread(img)
        write(os.path.join(sp0,os.path.basename(img)),mv0)
        write(os.path.join(sp1,os.path.basename(img)),mv1)