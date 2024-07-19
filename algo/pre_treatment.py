'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-07-19 10:03:43
Description: 
         ▄              ▄
        ▌▒█           ▄▀▒▌     
        ▌▒▒▀▄       ▄▀▒▒▒▐
       ▐▄▀▒▒▀▀▀▀▄▄▄▀▒▒▒▒▒▐     ,-----------------.
     ▄▄▀▒▒▒▒▒▒▒▒▒▒▒█▒▒▄█▒▐     (Wow,kousei's code)
   ▄▀▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▀██▀▒▐     `-,---------------' 
  ▐▒▒▒▄▄▄▒▒▒▒▒▒▒▒▒▒▒▒▒▀▄▒▒▌  _.-'   ,----------.
  ▌▒▒▐▄█▀▒▒▒▒▄▀█▄▒▒▒▒▒▒▒█▒▐         (surabashii)
 ▐▒▒▒▒▒▒▒▒▒▒▒▀██▀▒▒▒▒▒▒▒▒▀▄▌        `-,--------' 
 ▌▒▀▄██▄▒▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒▌      _.-'
 ▌▀▐▄█▄█▌▄▒▀▒▒▒▒▒▒░░░░░░▒▒▒▐ _.-'
▐▒▀▐▀▐▀▒▒▄▄▒▄▒▒▒▒▒░░░░░░▒▒▒▒▌
▐▒▒▒▀▀▄▄▒▒▒▄▒▒▒▒▒▒░░░░░░▒▒▒▐
 ▌▒▒▒▒▒▒▀▀▀▒▒▒▒▒▒▒▒░░░░▒▒▒▒▌
 ▐▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▐
  ▀▄▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▄▒▒▒▒▌
    ▀▄▒▒▒▒▒▒▒▒▒▒▄▄▄▀▒▒▒▒▄▀
      ▀▄▄▄▄▄▄▀▀▀▒▒▒▒▒▄▄▀
         ▒▒▒▒▒▒▒▒▒▒▀▀
When I wrote this, only God and I understood what I was doing
Now, God only knows
'''

from myutil import jhelp,jhelp_file,jhelp_folder
import os
from collections import defaultdict
'''
description: 对输入文件进行预处理,生成包含文件目录的dict文件
param {*} args
image_dir 文件名
multi_file 多文件格式，如果文件夹包含多个子文件目录 会在字典映射中添加所有子目录的遍历结果（array）
single_file 单个文件
return {*} 生成包含文件目录的dict文件
'''
def pre_treatment_3D(args,root,image_dir,multi_dir = False,single_file=False,left_only=False):
    if '/' not in image_dir:
        left_image_dir = os.path.join(args.left_dir_name,image_dir) if image_dir.lower()!='none' else args.left_dir_name
        right_image_dir = os.path.join(args.right_dir_name,image_dir) if image_dir.lower()!='none' else args.right_dir_name
    else:
        left_image_dir = image_dir
        right_image_dir = image_dir
    leroot,reroot = root,root
    if args.enable_single_input_mode:
        if not os.path.isdir(args.left_root) or not os.path.isdir(args.right_root):
            raise FileNotFoundError('[MM ERROR][input file]left eye or right eye image not exist,please check config file or your root folder')
        le = pre_treatment(args,leroot,left_image_dir,multi_dir=multi_dir,single_mode=True,single_file=args.left_root)
        re = pre_treatment(args,reroot,right_image_dir,multi_dir=multi_dir,single_mode=True,single_file=args.right_root) if not left_only else None
    elif args.enable_single_input_mode_2D:
        raise ValueError('[MM ERROR][config] 3D mode please use enable_single_input_mode(not 2d)')
    else:
        le = pre_treatment(args,root,left_image_dir,multi_dir=multi_dir)
        re = pre_treatment(args,reroot,right_image_dir,multi_dir=multi_dir) if not left_only else None
    return le,re

def pre_treatment(args,root,image_dir,multi_dir = False,single_mode=False,single_file=None):
    final_res = defaultdict(list)
    pths = jhelp_folder(root) if not single_mode else [single_file]
    core = single_core if not multi_dir else multi_core
    if multi_dir: assert args.image_file is not None,'[MM ERROR][input file]多文件mask格式不支持image文件夹名为none输入'
    assert len(pths)>0,'[MM ERROR][input file]error input,please check your root folder'
    for pth in pths:
        basename = os.path.basename(pth) if not single_mode else os.path.basename(single_file)
        final_res[basename] = core(pth,image_dir,args.n_start,args.n_limit,args.distributed_task)
    return final_res

def single_core(pth_,image_dir,n_start,n_limit,distributed_task):
    if '/' in image_dir and os.path.isdir(image_dir):
        pth = image_dir
    else:
        pth = os.path.join(pth_,image_dir) if image_dir.lower()!='none' else pth_
        pth = pth if os.path.isdir(pth) else pth_
    list_image = jhelp_file(pth)
    if len(list_image)<2:
        raise FileNotFoundError('[MM ERROR][input file]wrong root path and image size should >1 ')
    if n_start>0:
        list_image = list_image[n_start:]
    if n_limit>0:
        tmp =  list_image[:n_limit]
    else:
        tmp =  list_image
    cur_rank_start = distributed_task[0]*len(tmp)//distributed_task[1]
    next_rank_start = (1+distributed_task[0])*len(tmp)//distributed_task[1]+1
    return [tmp,[cur_rank_start,next_rank_start]]

def multi_core(pth_,image_dir,n_start,n_limit,distributed_task):
    res = defaultdict(list)
    pth = os.path.join(pth_,image_dir) if image_dir.lower()!='none' else pth_
    pth = pth if os.path.isdir(pth) else pth_
    second_folder = jhelp_folder(pth)
    assert len(second_folder)>0,'[MM ERROR][input file]error input,please check your root folder'
    for second_pth in second_folder:
        list_image = jhelp_file(second_pth)
        if len(list_image)<2:
            raise FileNotFoundError('[MM ERROR][input file]wrong root path and image size should >1 ,the path is :',pth)
        if n_start>0:
            list_image = list_image[n_start:]
        if n_limit>0:
            tmp =  list_image[:n_limit]
        else:
            tmp =  list_image
        cur_rank_start = distributed_task[0]*len(tmp)//distributed_task[1]
        next_rank_start = (1+distributed_task[0])*len(tmp)//distributed_task[1]+1
        res[os.path.basename(second_pth)] = [tmp,[cur_rank_start,next_rank_start]]
    return res

'''
description: 对深度计算的输入文件进行预处理
param {*} args
image_dir 左眼文件名
return {*} 生成包含文件目录的dict文件
'''
def pre_treatment_caldepth(args,root,image_dir):
    n_start = args.n_start
    n_limit = args.n_limit
    enable_extra_input_mode = args.enable_extra_input_mode
    enable_single_input_mode = args.enable_single_input_mode
    left_root = args.left_root
    right_root = args.right_root
    distributed_task = args.distributed_task

    le,re = None,None
    le_res,re_res = defaultdict(list),defaultdict(list)
    if not (enable_extra_input_mode or enable_single_input_mode):
        if not os.path.isdir(root):
            raise FileNotFoundError('[MM ERROR][root]%s is not a valid directory' % root)
        list_seq_file = delpoint(root,sorted(os.listdir(root)))
        for seq in list_seq_file:
            list_eye_file = delpoint(root+'/'+seq,os.listdir(root+'/'+seq))
            le_tmp,re_tmp =[],[]
            for l in list_eye_file:
                if 'left' in l.lower() or 'src_l' in l.lower() or 'le' == l.lower():
                    le = l
                if 'right' in l.lower() or 'src_r' in l.lower() or 're' == l.lower():
                    re = l
            if le is None or re is None:
                raise FileNotFoundError('[MM ERROR][input file]left eye or right eye image not exist!')

            
            lc = os.path.join(root,seq,le,image_dir)if image_dir.lower()!='none' else os.path.join(root,seq,le)
            rc = os.path.join(root,seq,re,image_dir) if image_dir.lower()!='none' else os.path.join(root,seq,re)
            le_tmp = prune_point(sorted(os.listdir(lc)))
            re_tmp = prune_point(sorted(os.listdir(rc))) 
            if n_start>0:
                le_tmp = le_tmp[n_start:]
                re_tmp = re_tmp[n_start:]
            if n_limit>0:
                tmp1,tmp2 = [os.path.join(lc,l) for l in le_tmp[:n_limit]],[os.path.join(rc,l) for l in re_tmp[:n_limit]]
            else:
                tmp1,tmp2 = [os.path.join(lc,l) for l in le_tmp],[os.path.join(rc,l) for l in re_tmp]
            if len(tmp1)!=len(tmp2):
                raise FileNotFoundError('[MM ERROR][input file]Left eye images should have the same number as right eye images but left = {}, right = {}'.format(len(tmp1),len(tmp2)))
            cur_rank_start = distributed_task[0]*len(tmp1)//distributed_task[1]
            next_rank_start = (1+distributed_task[0])*len(tmp1)//distributed_task[1]+1
            le_res[seq],re_res[seq] = [tmp1,[cur_rank_start,next_rank_start]],[tmp2,[cur_rank_start,next_rank_start]]
    else:
        if not os.path.isdir(left_root) or not os.path.isdir(right_root):
            raise FileNotFoundError('[MM ERROR][input file]left eye or right eye image not exist,please check config file or your root folder')
        if enable_extra_input_mode and enable_single_input_mode:
            raise ValueError('[MM ERROR][config]please choose one of enable_single_input_mode and enable_extra_input_mode')
        if enable_extra_input_mode:
            list_seq_file_l = delpoint(left_root,sorted(os.listdir(left_root)))
            for seq in list_seq_file_l:
                img_list = prune_point(sorted(os.listdir(left_root+'/'+seq+'/'+image_dir)))
                list_eye_file_l=[os.path.join(left_root,seq,image_dir,i) for i in img_list]
                img_list = prune_point(sorted(os.listdir(right_root+'/'+seq+'/'+image_dir)))
                list_eye_file_r=[os.path.join(right_root,seq,image_dir,i) for i in img_list]
                if n_start>0:
                    list_eye_file_l = list_eye_file_l[n_start:]
                    list_eye_file_r = list_eye_file_r[n_start:]
                if n_limit>0:
                    list_eye_file_l=list_eye_file_l[:n_limit]
                    list_eye_file_r = list_eye_file_r[:n_limit]
                if len(list_eye_file_l)!=len(list_eye_file_r):
                    raise ValueError('[MM ERROR][input file]Left eye images should have the same number as right eye images  but left = {}, right = {}'.format(len(list_eye_file_l),len(list_eye_file_r)))
                cur_rank_start = distributed_task[0]*len(list_eye_file_l)//distributed_task[1]
                next_rank_start = (1+distributed_task[0])*len(list_eye_file_l)//distributed_task[1]+1
                le_res[seq],re_res[seq] = [list_eye_file_l,[cur_rank_start,next_rank_start]],[list_eye_file_r,[cur_rank_start,next_rank_start]]
        else:
            seq = left_root.split('/')[-2]
            left_root = os.path.join(left_root,image_dir) if image_dir.lower()!='none' else left_root
            right_root = os.path.join(right_root,image_dir) if image_dir.lower()!='none' else right_root
            if not os.path.isdir(left_root) or not os.path.isdir(right_root):
                raise FileNotFoundError('[MM ERROR][input file]left eye or right eye image not exist,please check config file or your root folder')
            
            img_list = prune_point(sorted(os.listdir(left_root)))
            list_eye_file_l=[os.path.join(left_root,i) for i in img_list]
            img_list = prune_point(sorted(os.listdir(right_root)))
            list_eye_file_r=[os.path.join(right_root,i) for i in img_list]
            if n_start>0:
                list_eye_file_l = list_eye_file_l[n_start:]
                list_eye_file_r = list_eye_file_r[n_start:]
            if n_limit>0:
                list_eye_file_l=list_eye_file_l[:n_limit]
                list_eye_file_r = list_eye_file_r[:n_limit]
            if len(list_eye_file_l)!=len(list_eye_file_r):
                    raise ValueError('[MM ERROR][input file]Left eye images should have the same number as right eye images but left = {}, right = {}'.format(len(list_eye_file_l),len(list_eye_file_r)))
            cur_rank_start = distributed_task[0]*len(list_eye_file_l)//distributed_task[1]
            next_rank_start = (1+distributed_task[0])*len(list_eye_file_l)//distributed_task[1]+1
            le_res[seq],re_res[seq] = [list_eye_file_l,[cur_rank_start,next_rank_start]],[list_eye_file_r,[cur_rank_start,next_rank_start]]
    return le_res,re_res


def delpoint(root,arr):
        tmp = []
        for ar in arr:
            if ar[0] != '.' and os.path.isdir(root+'/' +ar):
                tmp.append(ar)
        return tmp

def prune_point(file):
    res = []
    for i in file:
        if i[0]!='.':
            res.append(i)
    return res


def get_frames_count(root):
    import re
    scenes = jhelp_folder(root)
    final_res = defaultdict(list)
    for scene in scenes:
        scene_name = os.path.basename(scene)
        files = jhelp_folder(scene)[0]
        flo = jhelp_file(files)[0]
        final_res[scene_name] = int(re.findall('\d+',flo)[-1])
    return final_res

if __name__ == '__main__':
    from cfg_process import init_param
    cp = os.getcwd() if '/algo' in os.getcwd() else os.path.join(os.getcwd(),'algo')
    args = init_param(os.path.join(cp,'..','config'))
    args.root = os.path.join(os.path.join(cp,'..'),args.root)
    args.model = os.path.join(os.path.join(cp,'..'),args.model)
    args.output = os.path.join(os.path.join(cp,'..'),args.output)
    args.distributed_task = [0,1]
    args.cur_rank = args.distributed_task[0] + 1
    args.use_tqdm = True

    args.root_2D =  '/Users/qhong/Documents/1117test/MM/motionmodel/pattern/1k/1'
    multi_file=args.front_mask_file
    pre_treatment(args,'/Users/qhong/Documents/1117test/MM/motionmodel/pattern/1k','mask',single_file=True,multi_dir=True)
