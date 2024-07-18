'''
Author: Qing Hong
Date: 2023-01-30 12:14:01
LastEditors: QingHong
LastEditTime: 2023-01-30 12:32:05
Description: file content
'''
from myutil import *
import os
import numpy as np
from tqdm import tqdm
root = '/Users/qhong/Downloads/val_flow_00-09_0_npz16/val/flow'
save = '/Users/qhong/Downloads/val_flow_00-09_0_npz16/val/flow_npy'

mkdir(save)
scenes = jhelp(root)
for scene_index in tqdm(range(len(scenes))):
    scene = scenes[scene_index]
    save_second = os.path.join(save,os.path.basename(scene))
    mkdir(save_second)
    
    npzs = jhelp(scene)
    for npz in npzs:
        im = np.load(npz,allow_pickle=True)
        u,v = im['u'],im['v']
        uv = np.concatenate((u[...,None],v[...,None]),axis=2)
        uv = np.nan_to_num(uv)
        write_flo_file(uv.astype('float32'),os.path.join(save_second,os.path.basename(npz).replace('npz','flo')))





