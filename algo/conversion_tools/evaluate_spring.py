'''
Author: Qing Hong
Date: 2023-09-04 17:06:57
LastEditors: QingHong
LastEditTime: 2023-09-18 13:11:46
Description: file content
'''

import os.path as osp
import os
import Imath,OpenEXR
import h5py
from tqdm import tqdm
import numpy as np
import array
import re
from tqdm.contrib.concurrent import process_map
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))
def writeFlo5File(filename,flow):
    if not os.path.exists(os.path.dirname(filename)):  # 判断目录是否存在
        os.makedirs(os.path.dirname(filename),exist_ok=True) 
    with h5py.File(filename, "w") as f:
        f.create_dataset("flow", data=flow, compression="gzip", compression_opts=5)
def exr_imread(filePath,pt=FLOAT):
    img_exr = OpenEXR.InputFile(filePath)
    dw = img_exr.header()['dataWindow']
    size = (dw.max.y - dw.min.y + 1,dw.max.x - dw.min.x + 1)
    if 'A' in img_exr.header()['channels']:
        r_str, g_str, b_str ,a_str= img_exr.channels('RGBA',pt)
        red = np.array(array.array('f', r_str))
        green = np.array(array.array('f', g_str))
        blue = np.array(array.array('f', b_str))
        alpha = np.array(array.array('f', a_str))
    else:
        r_str, g_str, b_str = img_exr.channels('RGB',pt)
        red = np.array(array.array('f', r_str))
        green = np.array(array.array('f', g_str))
        blue = np.array(array.array('f', b_str))
        alpha = np.zeros_like(red)
    red = red.reshape(size)
    green = green.reshape(size)
    blue = blue.reshape(size)
    alpha = alpha.reshape(size)
    image = np.stack([red,green,blue,alpha],axis=2)
    return image.astype('float32')

def help(args):
    i,mvs,sp,task,ttask = args
    mv = mvs[i]
    name = os.path.basename(mv)
    spname = ('{}_{:0>4}.flo5').format(ttask,int(re.findall('\d+',name)[0]))
    ssp = osp.join(sp,spname)
    if os.path.isfile(ssp):
        return
    flow = exr_imread(mv)[...,:2]
    h,w,_ = flow.shape
    flow[...,0] *= w
    flow[...,1] *= h
    if 'mv0' in task:
        if i ==len(mvs)-1:
            return
        flow[...,1] *= -1
    else:
        if i ==0:
            return
        flow[...,0] *= -1
    writeFlo5File(ssp,flow)

if __name__ == '__main__':
    root = '/Users/qhong/Downloads/spring_1007'
    save_root = '/Volumes/optflow_ssd/1'
    mkdir(save_root)
    algo = 'kousei-v1'
    scenes = jhelp_folder(root)
    for scene in scenes:
        for task,ttask in [[f'{algo}_left_mv0','flow_FW_left'],[f'{algo}_left_mv1','flow_BW_left'],[f'{algo}_right_mv0','flow_FW_right'],[f'{algo}_right_mv1','flow_BW_right']]:
            old_name = osp.join(scene,task)
            sp = osp.join(save_root,os.path.basename(scene),ttask)
            mvs = jhelp_file(old_name)
            data = [(i,mvs,sp,task,ttask) for i in range(len(mvs))]
            process_map(help, data, max_workers=64)
            