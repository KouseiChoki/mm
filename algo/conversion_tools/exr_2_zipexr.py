'''
Author: Qing Hong
Date: 2023-07-10 11:13:33
LastEditors: QingHong
LastEditTime: 2023-07-20 13:49:03
Description: file content
'''


import os,sys
import imageio
from tqdm import tqdm
import numpy as np 
from tqdm.contrib.concurrent import process_map
def mkdir(path):
        if  not os.path.exists(path):
            os.makedirs(path,exist_ok=True)

def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))

def zip_exr(root,desc='',multipro_nums=16):
    flos = jhelp_file(root)
    process_map(subpro, flos, max_workers=multipro_nums,desc=desc)

def subpro(flo_path):
    flo = imageio.imread(flo_path)
    imageio.imwrite(flo_path,flo,flags=imageio.plugins.freeimage.IO_FLAGS.EXR_ZIP|imageio.plugins.freeimage.IO_FLAGS.EXR_FLOAT)

def zip_exr_main(root,multipro_nums=16):
    if len(jhelp_folder(root))>0:
        for scene in jhelp_folder(root):
            if os.path.basename(scene) == 'MM':
                continue
            if len(jhelp_folder(scene))>0:
                for scene2 in jhelp_folder(scene):
                    if os.path.basename(scene2) == 'MM':
                        continue
                    if len(jhelp_folder(scene2))>0:
                            for scene3 in jhelp_folder(scene2):
                                if os.path.basename(scene3) == 'MM':
                                    continue
                                zip_exr(scene2,desc=os.path.basename(scene),multipro_nums=multipro_nums)
                    else:
                        zip_exr(scene2,desc=os.path.basename(scene),multipro_nums=multipro_nums)
            else:
                zip_exr(scene,desc=os.path.basename(scene),multipro_nums=multipro_nums)
    else:
        zip_exr(root,desc=os.path.basename(root),multipro_nums=multipro_nums)


if __name__ == '__main__':
    assert len(sys.argv)==2 ,'usage: python exr_2_zipexr.py root'
    root = sys.argv[1]
    zip_exr_main(root)
    
         
