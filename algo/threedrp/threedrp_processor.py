'''
Author: Qing Hong
Date: 2023-08-24 15:55:04
LastEditors: QingHong
LastEditTime: 2024-04-17 15:51:17
Description: file content
'''
import numpy as np
import os,sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/..')
from file_utils import read,write
import ctypes
import numpy.ctypeslib as npct
import platform
import re
from tqdm import tqdm
from PIL import Image
def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))
def initnpct():
    if platform.system().lower() == 'darwin':
        if platform.machine().lower() =='arm64':
            libfile = '3drp_arm64.so'
        else:
            libfile = '3drp.so'
    elif platform.system().lower() == 'linux':
            libfile = '3drp_linux.so'
    else:
        raise NotImplementedError('[MM ERROR][system]valid system')
    array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags="CONTIGUOUS")
    libcd = npct.load_library(os.path.join(os.path.abspath(os.path.abspath(__file__)+'/..'),libfile),'.')
    libcd.fn.restype = None
    libcd.fn.argtypes = [array_1d_float,array_1d_float, ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_float]
    return libcd

def threedrp_core(folder_image,folder_mv0,folder_mv1,save_path,args,depth_=None,force=False):
    dtype = 'hdr' if args.hdr else 'image'
    sp_mv0 = os.path.join(save_path,'from0')
    sp_mv1 = os.path.join(save_path,'from1')
    lib= initnpct()
    for i in tqdm(range(len(folder_image))):
        depth = None if depth_ is None else depth_[i]
        if folder_mv0 is not None:
            threedrp_algo(folder_mv0[i],folder_image[i],sp_mv0,lib,args.nphase,args.edge_thr,depth,force=force)
            
        if folder_mv1 is not None:
            threedrp_algo(folder_mv1[i],folder_image[i],sp_mv1,lib,args.nphase,args.edge_thr,depth,force=force)
    
def threedrp_algo(seq_mv,seq_img,sp,lib,nphase,edge_thr,depth=None,force=False):
    name = os.path.basename(seq_img)
    number = re.findall(r'\d+', name)[-1]
    ismv0 = 'mv0/' in seq_mv
    new_number = str(int(number) + 1).zfill(len(number)) if ismv0 else str(int(number) + -1).zfill(len(number))
    name = name.replace(number, new_number)
    if not force and os.path.isfile(os.path.join(sp,name)):
        return
    
    mv = read(seq_mv,type='flo')
    h, w = mv.shape[:2]
    img = read(seq_img,type='image')
    if img.shape[2]==3:
        ones_array = np.ones((h, w, 1)) * 255
        # 将原始数组和 ones_array 沿着第三个维度（通道维度）合并
        img = np.concatenate((img, ones_array), axis=2)
    if ismv0:
        mv[...,0]*=mv.shape[1]
        mv[...,1]*=mv.shape[0]
    else:
        mv[...,0]*=mv.shape[1]
        mv[...,1]*=mv.shape[0]
    if depth is not None:
        dp = read(depth)[...,0]
        mv[...,-1] = dp
    # print(depth)
    # write('/Users/qhong/Downloads/0206/test.exr',mv)
    sz = img.shape
    img,mv = img.reshape(-1),mv.reshape(-1)
    # ln = 1e-6
    # mm[np.where(mm==0)] += ln
    # nn[np.where(nn==0)] += ln
    cimage = img.astype('float32').copy()
    lib.fn(cimage,mv,h,w,float(nphase),edge_thr)
    cimage = cimage.reshape(sz).astype('uint8')
    write(os.path.join(sp,name),cimage)
    del img,cimage,mv
    