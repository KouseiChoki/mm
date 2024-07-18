'''
Author: Qing Hong
Date: 2023-03-16 17:22:20
LastEditors: QingHong
LastEditTime: 2023-03-27 18:51:07
Description: file content
'''
from myutil import *
import os
import re
from tqdm import tqdm
root = '/Users/qhong/Downloads/DoctorStrange_APR1000_comp_v1071/tmp'
save_path = '/Users/qhong/Downloads/DoctorStrange_APR1000_comp_v1071/mask'
inname = False
#2042B_0190
mkdir(save_path)
masks = jhelp(root)
mydic = {}
for mask_file in masks:
    mydic[os.path.basename(mask_file)] = jhelp(mask_file)


if inname:
    imgs = jhelp('/Users/qhong/Downloads/2042B_0190/le')
    for i in tqdm(range(len(imgs))):
        img = imgs[i]
        name = re.findall('\d+',os.path.basename(img))[-1]
        bg = np.zeros((0,0))
        for key in mydic.keys():
            for seqs in mydic[key]:
                if re.findall('\d+',os.path.basename(seqs))[-1] == name:
                    if bg.sum()==0:
                        bg = read(seqs)
                    else:
                        g = read(seqs)
                        # bg[np.where( (g>0) & (bg==0))] = g[np.where( (g>0) & (bg==0))]
                        bg[np.where((g>bg))] = g[np.where((g>bg))]
        if bg.sum()!=0:
            write(os.path.join(save_path,name+'.png'),bg)

else:
    for i in tqdm(range(len(mydic[list(mydic.keys())[0]]))):
        bg = np.zeros((0,0))
        for key in mydic.keys():
            if bg.sum()==0:
                bg = read(mydic[key][i])
            else:
                g = read(mydic[key][i])
                # bg[np.where( (g>0) & (bg==0))] = g[np.where( (g>0) & (bg==0))]
                bg[np.where((g>bg))] = g[np.where((g>bg))]
        write(os.path.join(save_path,os.path.basename(mydic[os.path.basename(list(mydic.keys())[0])][i])),bg)
