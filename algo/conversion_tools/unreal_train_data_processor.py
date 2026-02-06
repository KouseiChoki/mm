'''
Author: Qing Hong
Date: 2022-09-26 15:01:24
LastEditors: Qing Hong
LastEditTime: 2026-02-06 16:08:08
Description: file content
'''
import os,sys
from tqdm import tqdm
import numpy as np 
import cv2
import re
import shutil
import glob
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/..')
from file_utils import read,write
import OpenEXR
from collections import OrderedDict
def mkdir(path):
        if  not os.path.exists(path):
            os.makedirs(path,exist_ok=True)

def jhelp(c):
    return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]

def copyto(s,t):
    mkdir(os.path.dirname(t))
    shutil.copy(s,t)
    
def get_scene_shot(exr_path):
    exr = OpenEXR.InputFile(exr_path)
    h = exr.header()

    scene = h.get('unreal/sequenceName', b'Unknown').decode()
    shot_name = h.get('unreal/shotName', b'Unknown').decode()

    shot_frame_rel = int(
        h.get('unreal/shotFrameNumberRelative', b'0').decode()
    )

    return scene, shot_name, shot_frame_rel

if __name__ == '__main__':
    # a = [os.path.join(root,i) for i in sorted(list(filter(lambda x:x[0]!='.',os.listdir(root))))]
    assert len(sys.argv)==3 ,'usage: python xx.py root save_path'
    root = sys.argv[1]
    save_path = sys.argv[2]
    clean_root = True
    # 找到所有 png / exr
    all_pngs = glob.glob(os.path.join(root, '**/*.png'), recursive=True)
    all_exrs = glob.glob(os.path.join(root, '**/*.exr'), recursive=True)

    # 你要保留的
    keep_pngs = {
        p for p in all_pngs
        if 'finalimage.' in os.path.basename(p).lower()
    }

    keep_exrs = {
        p for p in all_exrs
        if 'finalimage' not in os.path.basename(p).lower()
    }
    if clean_root:
        # 需要删除的
        delete_pngs = set(all_pngs) - keep_pngs
        delete_exrs = set(all_exrs) - keep_exrs

        delete_files = sorted(delete_pngs | delete_exrs)
        if len(delete_files)>0:
            print(f'将要删除 {len(delete_files)} 个文件')

            for k in tqdm(range(len(delete_files)),desc='deleting...'):
                p = delete_files[k]
                try:
                    os.remove(p)
                except Exception as e:
                    print(f'删除失败: {p}, {e}')

    assert len(keep_pngs) == len(keep_exrs),f'error data,png:{len(keep_pngs)},exr:{len(keep_exrs)}'
    imgs = sorted(list(keep_pngs))
    exrs = sorted(list(keep_exrs))
   

    pre_scene = ''
    counter = 1
    for i in tqdm(range(len(imgs)),desc='copying..'):
        img = imgs[i]
        exr = exrs[i]
        scene, shot_name, shot_frame_rel = get_scene_shot(exr)


        # 新 shot 的判定
        if shot_frame_rel == 0:
            if scene != pre_scene:
                counter = 1
                pre_scene = scene
            else:
                counter+=1

        tmp = img.replace(root,save_path)
        target = os.path.join(os.path.dirname(tmp),str(counter),'image',os.path.basename(tmp).replace('FinalImage.',''))
        copyto(img,target)
        tmp = exr.replace(root,save_path)
        target = os.path.join(os.path.dirname(tmp),str(counter),'ori',os.path.basename(tmp))
        copyto(exr,target)

    # python /Users/qhong/Documents/1117test/MM/mm_NEW/mm/algo/conversion_tools/exr_processing/cal_mv.py --root /Volumes/Spica/optical_flow/Unreal_0206 --onlymv
    # for i in range(6000):
    #      img,exr = imgs[i],exrs[i]
    #      if (os.path.basename(img).replace('FinalImage.','').replace('.png','.exr') != os.path.basename(exr)):
    #           print(i)
    
    

    
    