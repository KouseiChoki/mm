'''
Author: Qing Hong
Date: 2023-06-08 10:53:18
LastEditors: QingHong
LastEditTime: 2023-06-08 15:15:57
Description: file content
'''
from __future__ import print_function
from __future__ import division

from PIL import Image
import numpy as np
import argparse
import os
import glob
from scipy.interpolate import RegularGridInterpolator


def parser():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('img_path', type=str,
                        help='path to image')
    parser.add_argument('lut_path',  type=str,
                        help='path to 3DLUT')
    parser.add_argument('--save_name',  type=str, default=None,
                        help='save name')
    parser.add_argument('--lut_size',  type=int, default=64,
                        help='lut size (default 64)')
    parser.add_argument('--batch', action='store_true',
                        help='apply batch (default False)')
    parser.add_argument('--method',  type=str, default=('linear'), choices=['linear', 'nearest'],
                        help='interpolation methods (defualt linear')
    args = parser.parse_args()
    return args


def load_lut(path):
    lut = np.zeros((LUT_SIZE**3, 3))
    with open(path, 'r') as f:
        for num, l in enumerate(f.readlines()[-LUT_SIZE**3:]):
            l = np.array(l.strip().split(' ')).astype(np.float32)
            lut[num] = l
    return lut


if __name__ == "__main__":
    args = parser()
    LUT_SIZE = args.lut_size

    print('Loading 3DLUT')
    lut = load_lut(args.lut_path)

    x = np.arange(0, 65)
    interpolation_func = RegularGridInterpolator((x, x, x), lut.reshape(65, 65, 65, 3), method='linear')

    if args.batch:
        img_paths = glob.glob(args.img_path+'*')
    else:
        img_paths = [args.img_path]

    extentions = ['.png', '.jpg']
    for num, path in enumerate(img_paths):
        print('\r{}/{}'.format(num+1, len(img_paths)), end='')
        if not True in [e in path for e in extentions]:
            continue
        if '_lut' in path:
            continue

        if args.save_name is None:
            f_name, ext = os.path.splitext(args.img_path)
            save_name = ''.join([f_name, '_lut', ext])
        else:
            save_name = args.save_name
        if os.path.exists(save_name):
            print('\nFile already exists: {}'.format(save_name))
            print('Skipping...')
            continue

        img = np.array(Image.open(path))[:, :, ::-1]
        new_image = interpolation_func((img*64))
        new_image *= 255
        new_image_pil = Image.fromarray(new_image.astype(np.uint8))

        new_image_pil.save(save_name)
    print('')

def ACES2065_REDLog3():
    import imageio
    img = imageio.v2.imread('/Users/qhong/Downloads/ColorSample/ACES2065_AP0/KTC1200_comp_v416.1180.exr') 
    LUT_SIZE = 65
    lut = load_lut('/Users/qhong/Documents/1117test/MM/motionmodel/algo/lut/SV_20804_Log3G10RWG_LMT_hdr_p3d65pq300_11nits.cube')
    x = np.arange(0, 65)
    interpolation_func = RegularGridInterpolator((x, x, x), lut.reshape(65, 65, 65, 3), method='linear')
    new_image = interpolation_func(img.clip(0,1)*64)
    p_R,p_G,p_B = new_image[...,0],new_image[...,1],new_image[...,2]
    
    r_rwg =  1.26556739*p_R -0.13522952*p_G -0.13033787*p_B
    g_rwg = -0.02056862*p_R +0.94318153*p_G +0.07738710*p_B
    b_rwg =  0.06257518*p_R +0.20653821*p_G +0.73088661*p_B
    
    r_rwg_,g_rwg_,b_rwg_ = r_rwg.copy(),g_rwg.copy(),b_rwg.copy()
    r_rwg_[np.where(r_rwg<-0.01)] =0  
    g_rwg_[np.where(g_rwg<-0.01)] =0  
    b_rwg_[np.where(b_rwg<-0.01)] =0  
    rlog = (r_rwg + 0.01) * 15.1927
    rlog[np.where(r_rwg >= -0.01)] = 0.224282 * np.log((r_rwg_ + 0.01) * 155.975327 + 1.0)[np.where(r_rwg >= -0.01)]
    glog = (g_rwg + 0.01) * 15.1927
    glog[np.where(g_rwg >= -0.01)] = 0.224282 * np.log((g_rwg_ + 0.01) * 155.975327 + 1.0)[np.where(g_rwg >= -0.01)]
    blog = (b_rwg + 0.01) * 15.1927
    blog[np.where(b_rwg >= -0.01)] = 0.224282 * np.log((b_rwg_ + 0.01) * 155.975327 + 1.0)[np.where(b_rwg >= -0.01)]
    result = np.ascontiguousarray(np.concatenate((rlog[...,None],glog[...,None],blog[...,None]),axis=2))
    