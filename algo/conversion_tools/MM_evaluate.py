'''
Author: Qing Hong
Date: 2023-12-20 10:07:33
LastEditors: QingHong
LastEditTime: 2024-04-17 15:38:41
Description: file content
'''

import os,sys,sys
import numpy as np 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/..')
from file_utils import mvwrite
from threedrp.threedrp_processor import threedrp_core
import argparse
from myutil import immc,jhelp_folder,jhelp_file,write,read
import re
def gofind(c,keyword,mode = 'basename'):
    if mode =='basename':
        res = list(filter(lambda x:keyword.lower() in os.path.basename(x).lower(),c)) 
    else:
        res = list(filter(lambda x:keyword.lower() in x.lower(),c)) 
    return res 
def refine_float(lst):
    tmp = {}
    res = []
    for i in lst:
        tmp[int(re.findall('\d+',i)[-1])] = i
    for k in range(len(lst)):
        res.append(tmp[k])
    return res
def num_to_str(num):
    if isinstance(num, float):  # 检查num是否为浮点数
        return format(num, ".1f")  # 浮点数，保留1位小数
    return str(num)  # 其他类型，直接转换为字符串
def init_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',  help="your data path", required=True)
    parser.add_argument('--img_path',  help="your image data path")
    # parser.add_argument('--depth_path',  help="your depth data path")
    parser.add_argument('--depth', action='store_true', help="use extra depth")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
    parser.add_argument('--hdr', action='store_true', help="use hdr image")
    parser.add_argument('--f', action='store_true', help="force run")
    parser.add_argument('--method', default='3drp')
    parser.add_argument('--nphase', default=1)
    parser.add_argument('--edge_thr', type=float, default=2)
    args = parser.parse_args()
    return args

def find_folders_with_mv(root_dir):
    folders_with_mv = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if any('mv0' in s or 'mv1' in s for s in dirnames):
            if '/bbox/' not in dirpath:
                folders_with_mv.append(dirpath)
            # 如果你只想在第一级子目录中寻找，可以使用下面的代码代替上面的代码
            # dirnames.remove('mv0')  # 防止进一步向下搜索
    return folders_with_mv

def immc_core(datas):
    i,folder_image,folder_mv0,folder_mv1,save_path,args = datas
    dtype = 'hdr' if args.hdr else 'image'
    mv0,mv1 = read(folder_mv0[i],'flo'),read(folder_mv1[i],'flo')
    h,w,_ = mv0.shape
    #denormalize
    mv0[...,0] *= w
    mv0[...,1] *= h
    mv1[...,0] *= w
    mv1[...,1] *= h
    img_mv0 = immc(read(folder_image[i+1],type=dtype),mv0) if i!= len(folder_image) -1 else None
    img_mv1 = immc(read(folder_image[i-1],type=dtype),mv1) if i!= 0 else None
    sp_mv0 = os.path.join(save_path+'_from0',os.path.basename(folder_image[i]))
    sp_mv1 = os.path.join(save_path+'_from1',os.path.basename(folder_image[i]))
    if img_mv0 is not None:
        write(sp_mv0,img_mv0)
    else:
        write(sp_mv0,np.zeros_like(img_mv1))
    if img_mv1 is not None:
        write(sp_mv1,img_mv1)
    else:
        write(sp_mv1,np.zeros_like(img_mv0))

def evaluate_main(type='3drp'):
    if type == '3drp':
        return threedrp_core
    elif type =='immc':
        return immc_core
    else:
        raise(NotImplementedError('3drp or immc'))
    
    



if __name__ == '__main__':
    # assert len(sys.argv)==3 ,'usage: python exr_get_mv.py root save_path'
    args = init_param()
    file_names = find_folders_with_mv(args.path)
    assert len(file_names)>0,'error root'
    #init
    mv0_name = 'mv0'
    mv1_name = 'mv1'
    image_name = 'video' if args.hdr else 'image'
    for file_name in file_names:
        print(f'processing:{file_name}')
        for f in jhelp_folder(file_name): #change mv0name 
            if 'mv0' in f and 'mv0' != f:
                mv0_name = f
            if 'mv1' in f and 'mv1' != f:
                mv1_name = f
        mv0_path = os.path.join(file_name,mv0_name)
        mv1_path = os.path.join(file_name,mv1_name)
        folder_mv0 = jhelp_file(os.path.join(file_name,mv0_name)) if os.path.isdir(mv0_path) else None
        folder_mv1 = jhelp_file(os.path.join(file_name,mv1_name)) if os.path.isdir(mv1_path) else None
        depth= None
        if args.img_path is None:
            tmp = file_name
        else:
            base = os.path.basename(file_name)
            if os.path.isdir(os.path.join(args.img_path,base)):
                tmp = os.path.join(args.img_path,base)
            else:
                tmp = args.img_path

        folder_image = jhelp_file(os.path.join(tmp,image_name)) if os.path.isdir(os.path.join(tmp,image_name)) else jhelp_file(tmp)
        
        if args.depth:
            # depth = refine_float(gofind(jhelp_file(os.path.join(tmp,'ori')),keyword='PWWorldDepth'))
            depth = jhelp_file(os.path.join(tmp,'world_depth'))
            assert len(depth)>0 and len(depth) == len(folder_image),'depth error,please check your input folder'

        assert len(folder_image)>0,'error image path'
        save_path = os.path.join(file_name,f'FI_{num_to_str(args.nphase)}phase')
        data = []
        # evaluate = evaluate_main(type=args.method)
        
        
        # if args.depth_path is not None:
        #     base = os.path.basename(file_name)
        #     depth = refine_float(gofind(jhelp_file(os.path.join(args.depth_path,base,'ori')),keyword='PWWorldDepth'))
        # print(depth)
        threedrp_core(folder_image,folder_mv0,folder_mv1,save_path,args,depth,force=args.f)
        # process_map(evaluate, data, max_workers= args.core,desc='processing:{}'.format(os.path.basename(save_path)))
        

    
    