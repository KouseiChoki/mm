'''
Author: Qing Hong
Date: 2023-03-16 17:22:20
LastEditors: QingHong
LastEditTime: 2023-03-24 18:00:21
Description: file content
'''
from myutil import *
import os
from tqdm import tqdm
objmv = '/Users/qhong/Downloads/DoctorStrange_APR1000_comp_v1071/0324/objectmv/DoctorStrange_APR1000_comp_v1071/flowformer_flowformer-32g-mask16_left_original_mv0'
maskfile = '/Users/qhong/Downloads/DoctorStrange_APR1000_comp_v1071/manu_mask'
save_path = '/Users/qhong/Downloads/DoctorStrange_APR1000_comp_v1071'
mkdir(os.path.join(save_path,os.path.basename(objmv)))
mvs = jhelp(objmv)
masks = jhelp(maskfile)
for i in tqdm(range(len(mvs))):
    mv = read(mvs[i])
    mask = read(masks[i]) if '_mv0' in objmv else read(masks[i+1])
    # mv[np.where(mask!=255)] = 0
    mv[np.where(mask==0)] = 0
    write(os.path.join(save_path,os.path.basename(objmv),os.path.basename(mvs[i])),mv)
    
