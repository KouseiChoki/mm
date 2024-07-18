'''
Author: Qing Hong
Date: 2023-08-24 15:55:04
LastEditors: QingHong
LastEditTime: 2023-09-21 17:20:19
Description: file content
'''
import numpy as np
import os
from myutil import jhelp_file,jhelp_folder
from file_utils import read,write
from tqdm.contrib.concurrent import process_map
import ctypes
import numpy.ctypeslib as npct
import platform
from tqdm import tqdm
def initnpct():
    if platform.system().lower() == 'darwin':
        if platform.machine().lower() =='arm64':
            libfile = 'virtualdepth_arm64.so'
        else:
            libfile = 'virtualdepth.so'
    elif platform.system().lower() == 'linux':
            libfile = 'virtualdepth_linux.so'
    else:
        raise NotImplementedError('[MM ERROR][system]valid system')
    array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags="CONTIGUOUS")
    libcd = npct.load_library(os.path.join(os.path.abspath(os.path.abspath(__file__)+'/..'),libfile),'.')
    libcd.fn.restype = None
    libcd.fn.argtypes = [array_1d_float,array_1d_float, ctypes.c_int,ctypes.c_int]
    return libcd
def virtual_depth_core(args):
    seqs = jhelp_folder(args.output)
    for seq in seqs:
        if os.path.basename(seq) == args.MM: ##定死mv0mv1跑depth
            continue
        contents = jhelp_folder(seq)
        folder = set()
        for t in contents:
            if os.path.isdir(t) and 'dumpedfile' not in t and ('_mv0' in t or '_mv1' in t):
                tmp = os.path.dirname(t)
                tmp = os.path.join(tmp,os.path.basename(t).replace('mv0','{}').replace('mv1','{}'))
                folder.add(tmp)
        for task in folder:
            seq_mv0s = jhelp_file(task.format('mv0'))
            seq_mv1s = jhelp_file(task.format('mv1'))
            
            if False:
                for i in tqdm(range(len(seq_mv0s)-1),desc='virtualdepth'):
                    data = (i,seq_mv0s,seq_mv1s)
                    virtual_depth_algo(data)
            else:
                data = [(i,seq_mv0s,seq_mv1s) for i in range(len(seq_mv0s)-1)]
                process_map(virtual_depth_algo, data, max_workers=args.virtualdepth_core)

def virtual_depth_algo(args):
    i,seq_mv0s,seq_mv1s = args
    libcd = initnpct()
    mv0_ = seq_mv0s[i]
    mv1_ = seq_mv1s[i+1]
    mv0_s = read(mv0_,type='flo')
    mv1_s = read(mv1_,type='flo')
    film_border=[0,0,0,0]
    if sum(film_border)>0:
            mv0 = mv0_s[film_border[0]:mv0_s.shape[0]-film_border[1],film_border[2]:mv0_s.shape[1]-film_border[3]].copy()
            mv1 = mv1_s[film_border[0]:mv1_s.shape[0]-film_border[1],film_border[2]:mv1_s.shape[1]-film_border[3]].copy()
    else:
        mv0 = mv0_s.copy()
        mv1 = mv1_s.copy()
    mv0[...,0]*=mv0_s.shape[1]
    mv0[...,1]*=mv0_s.shape[0]
    mv1[...,0]*=mv1_s.shape[1]
    mv1[...,1]*=mv1_s.shape[0]
    ##fix 0 bug
    h, w, _ = mv0.shape[:3]
    m = mv0[...,:2].copy()
    n = mv1[...,:2].copy()
    mm,nn = m.reshape(-1),n.reshape(-1)
    # ln = 1e-6
    # mm[np.where(mm==0)] += ln
    # nn[np.where(nn==0)] += ln
    libcd.fn(mm,nn,h,w)
    result_mv0 = mm.reshape(m.shape)[...,0]/65535
    result_mv1 = nn.reshape(n.shape)[...,0]/65535
    if sum(film_border)>0:  
            result_mv0_ = np.zeros((h+film_border[0]+film_border[1],w+film_border[2]+film_border[3])).astype('float32')
            result_mv0_[film_border[0]:result_mv0.shape[0]+film_border[0],film_border[2]:result_mv0.shape[1]+film_border[2]] = result_mv0
            result_mv0 = result_mv0_

            result_mv1_ = np.zeros((h+film_border[0]+film_border[1],w+film_border[2]+film_border[3])).astype('float32')
            result_mv1_[film_border[0]:result_mv1.shape[0]+film_border[0],film_border[2]:result_mv1.shape[1]+film_border[2]] = result_mv1
            result_mv1 = result_mv1_
    # 65535
    mv0_s[...,3] = result_mv0
    mv1_s[...,3] = result_mv1
    write(mv0_,mv0_s)
    write(mv1_,mv1_s)