'''
Author: Qing Hong
Date: 2022-09-26 15:01:24
LastEditors: QingHong
LastEditTime: 2024-02-07 17:33:09
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/..')
from file_utils import mvwrite,read,custom_refine
from tqdm.contrib.concurrent import process_map
import warnings
import re
import argparse

def mkdir(path):
        if  not os.path.exists(path):
            os.makedirs(path,exist_ok=True)
def init_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',  help="your data path", required=True)
    parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
    parser.add_argument('--pass_when_exist', action='store_true', help="pass cal")
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    args = parser.parse_args()
    return args

def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))
def prune(c,keyword,mode = 'basename'):
    if mode =='basename':
        res = list(filter(lambda x:keyword.lower() not in os.path.basename(x).lower(),c)) 
    else:
        res = list(filter(lambda x:keyword.lower() not in x.lower(),c))
    return res 

def gofind(c,keyword,mode = 'basename'):
    if mode =='basename':
        res = list(filter(lambda x:keyword.lower() in os.path.basename(x).lower(),c)) 
    else:
        res = list(filter(lambda x:keyword.lower() in x.lower(),c)) 
    return res 

def custom_refine(flow,zero_to_one=True):
    height,width = flow[...,0].shape
    #average value
    if zero_to_one:
        flow[...,0]/=width
        flow[...,1]/=-height
    else:
        flow[...,0]/=-width
        flow[...,1]/=height
    # if flow.shape[2] >= 3:
    #     flow[...,2] /= 65535 #65535*255
    return flow

def rename(source,target):
    try:
        os.rename(source,target)
        print(f"文件夹名已从 '{source}' 改为 '{target}'")
    except OSError as error:
        print(f"更改文件夹名时发生错误: {error}")

def restore_file_name(root):
    for file in jhelp_folder(root):
        if os.path.basename(file).lower() =='orinal':
            rename(file, os.path.join(os.path.dirname(file),'ori'))
        if os.path.basename(file).lower() =='12':
            rename(file, os.path.join(os.path.dirname(file),'12fps'))
        if os.path.basename(file).lower() =='24':
            rename(file, os.path.join(os.path.dirname(file),'24fps'))
        if os.path.basename(file).lower() =='48':
            rename(file, os.path.join(os.path.dirname(file),'48fps'))
        if os.path.basename(file) =='mask':
            rename(file, os.path.join(os.path.dirname(file),'Mask'))
            
            
def loop_helper(files,key='ori'):
    if len(jhelp_folder(files)) == 0:
        return [files]
    res = []
    for file in jhelp_folder(files):
        if os.path.basename(file) == key:
            return [file]
        if 'fps' in os.path.basename(file).lower() or os.path.basename(file) in ['12','24','48']:
            res += [file]
        else:
            res += loop_helper(file)
    return res

def mkdir_helper(files,root,name):
    if len(files)>0:
        mkdir(os.path.join(root,name))
        for file in files:
            shutil.move(file,os.path.join(root,name))

def refine_float(lst):
    return sorted(lst, key=lambda x: int(re.findall(r"0\.(\d+)",x)[-1]))

def generate_bounding_box(mask):
    if len(mask.shape) == 3:
        mask = mask[...,0]
    verge = np.where(mask!=0)
    ly,lx = verge[0].min(),verge[1].min()
    ry,rx = verge[0].max(),verge[1].max()
    return lx,ly,rx,ry


def go_padding(mask,max_x,max_y):
    h,w = mask.shape
    lx,ly,rx,ry = generate_bounding_box(mask)
    diff_x = max_x - (rx-lx)
    diff_y = max_y - (ry-ly)
    pad_x_l,pad_x_r,pad_y_l,pad_y_r = 0,0,0,0
    pad_x = diff_x//2
    pad_x_l = pad_x
    pad_x_r = diff_x - pad_x
    pad_y = diff_y//2
    pad_y_l = pad_y
    pad_y_r = diff_y - pad_y
    lx_,ly_,rx_,ry_ = lx-pad_x_l,ly-pad_y_l,rx+pad_x_r,ry+pad_y_r
    if lx_ < 0 :
        lx_ = 0
        rx_ = max_x
    if ly_ <0 :
        ly_ = 0 
        ry_ = max_y
    if rx_ >= w:
        rx_ = w - 1
        lx_ = (w - 1 - max_x)
    if ry_ >= h:
        ry_ = h - 1
        ly_ = (h - 1 - max_y)
    return [lx_,ly_,rx_,ry_]

def bounding_box_main(file_name,args):
    mv0_name = os.path.join(file_name,'mv0')
    mv1_name = os.path.join(file_name,'mv1')
    mask_name = os.path.join(file_name,'Mask')
    image_name = os.path.join(file_name,'image')
    save_path = os.path.join(file_name,'bbox')
    name = os.path.basename(file_name)
    mv0s = jhelp_file(mv0_name) if os.path.isfile(mv0_name) else None
    mv1s = jhelp_file(mv1_name) 
    masks = jhelp_file(mask_name)
    images = jhelp_file(image_name)
    crop_pos = []
    offset = [[0,0]]
    for i in range(len(images)):
        mask = read(masks[i],type='mask')
        h,w = mask.shape
        lx_,ly_,rx_,ry_ = go_padding(mask,args.width,args.height)
        crop_pos.append([lx_,ly_,rx_,ry_])
    for i in range(len(crop_pos)-1):
        offset.append([crop_pos[i+1][0]-crop_pos[i][0],crop_pos[i+1][1]-crop_pos[i][1]])

    for i in tqdm(range(len(images)),desc=name):
        lx,ly,rx,ry = crop_pos[i]
        image = read(images[i],type='image')[ly:ry,lx:rx]
        mv0 = read(mv0s[i],type='flo')[ly:ry,lx:rx][...,:2] if mv0s is not None else None
        mv1 = read(mv1s[i],type='flo')[ly:ry,lx:rx][...,:2] if mv1s is not None else None
        mask = read(masks[i],type='mask')
        #mv denormalized
        if mv0 is not None:
            mv0[...,0] *= w
            mv0[...,1] *= -h
            mv0[...,0] += offset[i+1][0]
            mv0[...,1] += offset[i+1][1]
        if mv1 is not None:
            mv1[...,0] *= -w
            mv1[...,1] *= h
            mv1[...,0] -= offset[i][0]
            mv1[...,1] -= offset[i][1]

        mask = mask[ly:ry,lx:rx]
        fname = os.path.basename(images[i])
        mvwrite(os.path.join(save_path,'image',fname),image[...,:3])
        mvwrite(os.path.join(save_path,'Mask',fname.replace('.png','.exr')),np.repeat(mask[...,None],4,axis=2))
        if mv0 is not None:
            mv0[...,0] /= args.width
            mv0[...,1] /= -args.height
            mvwrite(os.path.join(save_path,'mv0',fname.replace('.png','.exr')),mv0,precision='half')
        if mv1 is not None:
            mv1[...,0] /= -args.width
            mv1[...,1] /= args.height
            mvwrite(os.path.join(save_path,'mv1',fname.replace('.png','.exr')),mv1,precision='half')


    
if __name__ == '__main__':
    # assert len(sys.argv)==3 ,'usage: python exr_get_mv.py root save_path'
    
    args = init_param()
    root = args.path
    file_names = loop_helper(root,key='mv1')
    assert len(file_names)>0,'error root'

    for id,file_name in enumerate(file_names):
        file_name = os.path.abspath(os.path.dirname(os.path.abspath(file_name)))
        
        print('starting camera mv calculation({}/{}) {}'.format(id+1,len(file_names),file_name))
        bounding_box_main(file_name,args)
        

