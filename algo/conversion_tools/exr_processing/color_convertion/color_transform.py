'''
Author: Qing Hong
Date: 2024-03-29 14:25:10
LastEditors: Qing Hong
LastEditTime: 2024-05-17 10:56:50
Description: file content
'''

import os,sys
from tqdm import tqdm
import numpy as np 
import cv2
import shutil
import OpenEXR, Imath, array
import os,sys
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/../../..')
from file_utils import mvwrite,read
from tqdm.contrib.concurrent import process_map
import warnings
import re
import argparse
from colorutil import Color_transform
def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))
def init_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',  help="your data path")
    parser.add_argument('--folder_name', help="your folder_name(batch process)")
    parser.add_argument('--src',default='lin_rec709', help="source color space")
    parser.add_argument('--target',default='acescg', help="target color space")
    parser.add_argument('--show', action='store_true', help="show all color space")
    parser.add_argument('--core', type=int, default=4)
    args = parser.parse_args()
    return args

def find_folders_with_subfolder(root_path, keys = [], path_keys = [] ,excs = [] ,path_excs =[]):
    """
    Find all folders in the root_path that contain a subfolder with the name subfolder_name.
    """
    folders_with_subfolder = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Check if the subfolder_name is in the list of directories
        flag = True
        for key in keys:
            if key not in dirnames:
                flag = False
        for path_key in path_keys:
            if path_key not in dirpath:
                flag = False
        for exc in excs:
            if exc in dirnames:
                flag = False
        for exc in path_excs:
            if exc in dirpath:
                flag = False
        if flag:
            folders_with_subfolder.append(dirpath)

    return folders_with_subfolder

warnings.filterwarnings("ignore")
def get_color_space():
    try:
        import PyOpenColorIO as ocio
        ocio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.ocio')
        config = ocio.Config.CreateFromFile(ocio_path)
        res = []
        cs_iter = config.getColorSpaceNames()
        for cs in cs_iter:
            if ' - ' not in cs: #prune role
                res.append(cs)
        return res
    except:
        print('wrong color space')

def color_trans_core(image_,args):
    # image_,args = datas
    dtmp = os.path.dirname(image_)
    tmp = f'{dtmp}_{args.src}_to_{args.target}'
    sp = image_.replace(dtmp,tmp)
    if not os.path.isfile(sp):
        global cscore
        image = read(image_,type='flo')
        alpha = None
        if image.shape[2] == 4:
            alpha = image[...,3:].copy()
            cimage = image[...,:3].copy()
            cscore.apply(cimage)
            image = np.concatenate((cimage,alpha),axis=-1)
        else:
            cscore.apply(image)
        mvwrite(sp,image,precision = 'half')


if __name__ == '__main__':
    args = init_param()
    if args.show:
        css = sorted(get_color_space())
        for cs in css:
            print(cs)
    else:
        if not (args.path is not None and len(args.path)>0):
            print('error root path,usage:\nmmcolortrans --path (--folder_name *if you want batch process programs*) \nor\nmmcolortrans --show')
        else:
            data = []
            fns = []
            if args.folder_name is None:
                fns = [args.path]
            else:
                fns = find_folders_with_subfolder(args.path,path_keys=[args.folder_name])
            
            global cscore
            cscore = Color_transform(args.src,args.target)
            for fn in fns:
                file_datas = jhelp_file(fn)
                # print(file_datas,fn)
                for i in tqdm(range(len(file_datas)),desc='processing:{}'.format(os.path.basename(fn))):
                    # data.append([file_datas[i],args])
                    color_trans_core(file_datas[i],args)
                # process_map(color_trans_core, data, max_workers= args.core,desc='processing:{}'.format(os.path.basename(fn)))
                # color_trans_core(file_datas[i],args)
        # file_datas = jhelp_file(file_name)
        # #prune data
        # file_datas = prune(file_datas,'finalimage')
        # for i in range(len(file_datas)):
        #     data.append([i,file_datas,save_path,file_datas[i],args])
        # process_map(color_trans_core, data, max_workers= args.core,desc='processing:{}'.format(name))
