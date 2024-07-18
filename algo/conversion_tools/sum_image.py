'''
Author: Qing Hong
Date: 2023-03-16 17:22:20
LastEditors: QingHong
LastEditTime: 2023-03-23 16:32:43
Description: file content
'''
from myutil import *
import os
from tqdm import tqdm
root = '/Users/qhong/Downloads/DoctorStrange_APR1000_comp_v1071'
save_path = '/Users/qhong/Downloads/DoctorStrange_APR1000_comp_v1071/result'
mkdir(save_path)
files = jhelp(root)
mydic_image = {}
mydic_mask = {}
for file in files:
    if 'result' in file:
        continue
    if '_image' in file:
        mydic_image[os.path.basename(file).replace('_image','')] = jhelp(file)
    if '_alpha' in file:
        mydic_mask[os.path.basename(file).replace('_alpha','')] = jhelp(file)

for i in tqdm(range(len(mydic_image[list(mydic_image.keys())[0]]))):
    bg = np.zeros((0,0))
    for key in mydic_image.keys():
        if key in ['fx','sky','character']:
            continue
        if bg.sum()==0:
            bg = read(mydic_image[key][i])
        else:
            gmask= read(mydic_mask[key][i])
            g = read(mydic_image[key][i])
            # bg[np.where( (g>0) & (bg==0))] = g[np.where( (g>0) & (bg==0))]
            bg[np.where(gmask!=0)] = g[np.where(gmask!=0)]
    write(os.path.join(save_path,os.path.basename(mydic_image[os.path.basename(list(mydic_image.keys())[0])][i])),bg)
