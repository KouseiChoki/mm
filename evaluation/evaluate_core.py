'''
Author: Qing Hong
Date: 2024-04-11 13:55:07
LastEditors: Qing Hong
LastEditTime: 2024-09-14 14:22:59
Description: file content
'''
import os,sys
cur_path = os.path.dirname(os.path.abspath(__file__))
from tqdm import tqdm
sys.path.insert(0, cur_path+'/../algo')
from file_utils import read
from openpyxl import Workbook
import datetime
import re
import numpy as np
def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))

# 提取文件名中的数字序号
def extract_number_from_filename(file_path):
    file_name = os.path.basename(file_path)
    number = re.findall(r'\d+', file_name)
    return number[-1] if number else None

def find_matching_files(target_file, file_paths):
    target_number = extract_number_from_filename(target_file)
    
    if target_number is None:
        None

    # 在路径数组中寻找与目标文件相同序号的文件
    for path in file_paths:
        number = extract_number_from_filename(path)
        if number == target_number:
            return path
    
    return None

def evaluate(mv_path,gt_path,skip=1,speed=[0,1],masks=None,fg=True):
    mvs = jhelp_file(mv_path)
    gts = jhelp_file(gt_path)
    # assert len(mvs)>skip*2 and len(gts) > skip*2 and len(gts)==len(mvs),'evluation data length should >=2'
    mvs = mvs[skip:-skip]
    # gts = gts[skip:-skip]
    if masks is not None:
        masks = jhelp_file(masks)
        masks = masks[skip:-skip]
    epe_all,px1_all,px3_all,px5_all = 0,0,0,0
    totalnum = len(mvs)
    for i in range(totalnum):
        mv = read(mvs[i],type='flo')[...,:2]
        # gt = read(gts[i],type='flo')[...,:2]
        gt_ = find_matching_files(mvs[i],gts)
        if gt_ is None or not os.path.isfile(gt_):
            print(f'can not find file:{gt_}')
            totalnum-=1
            continue
        gt = read(gt_,type='flo')[...,:2]
        if masks is not None:
            mask = read(masks[i],type='mask')
            if not fg:
                mask += 1
                mask[np.where(mask>1)] = 0
        else:
            mask = np.ones_like(mv)[...,0]
        if speed !=[0,1]:
            mag = np.sqrt(np.sum(mv**2,axis=2))
            mask_mag = np.where((mag<speed[0])|(mag>speed[1]))
            mask[mask_mag] = 0
        metrics = evaluate_single_frame(mv,gt,mask)
        epe_all += metrics['epe']
        px1_all += metrics['1px']
        px3_all += metrics['3px']
        px5_all += metrics['5px']
    return epe_all/totalnum,px1_all/totalnum,px3_all/totalnum,px5_all/totalnum
    
def evaluate_single_frame(mv,gt,valid,norm=True):
    if norm:
        h,w,_ = mv.shape
        mv[...,0] *= w
        mv[...,1] *= h
        gt[...,0] *= w
        gt[...,1] *= h
    epe = np.sqrt(np.sum((mv - gt)**2,axis=-1))
    epe = epe.view() * valid.view()
    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).mean().item(),
        '3px': (epe < 3).mean().item(),
        '5px': (epe < 5).mean().item(),
    }
    return metrics

def mm_evaluate(source_root,gt_root,obj_mode=False,skip=1,fg=True):
    sources = jhelp_folder(source_root)
    result = {}
    for scene in tqdm(sources):
        scene_name = os.path.basename(scene)
        gt = os.path.join(gt_root,scene_name)
        if not os.path.isdir(gt):
            continue
        masks = None if not obj_mode else f'{gt}/Mask'
        ff = [os.path.basename(i) for i in jhelp_folder(scene)]
        mv0_name,mv1_name = 'mv0','mv1'
        for f in ff:
            if 'mv0' in f and 'mv0' != f:
                mv0_name = f
            if 'mv1' in f and 'mv1' != f:
                mv1_name = f
        for speed in [[0,0.005],[0.005,0.01],[0.01,1]]:
            s = f's{speed[0]*1000}_{speed[1]*1000}'
            result[f'{scene_name}_mv1_{s}'] = evaluate(f'{scene}/{mv1_name}',f'{gt}/mv1',skip=skip,speed=speed,masks=masks,fg=fg)
            # if os.path.isdir(f'{gt}/mv0'):
                # result[f'{scene_name}_mv0_{s}'] = evaluate(f'{scene}/{mv0_name}',f'{gt}/mv0',skip=skip,speed=speed,masks=masks,fg=fg)
    return result

def esf(text, min_length=10):
    return f"{text:<{min_length}}"

def show_evaluation(di,algorithm='mm',sp='',excel=True):
    if excel:
        # 创建一个新的工作簿和工作表
        wb = Workbook()
        ws = wb.active
        # 更改工作表名称
        ws.title =algorithm
        # 写入一些数据
        ws.append(['Scene', 'epe', '1px', '3px', '5px'])
        for item in di.keys():
            value = di[item]
            ws.append([item, f'{value[0]:.3f}',f'{value[1]:.2%}',f'{value[2]:.2%}',f'{value[3]:.2%}'])
        wb.save(sp.replace('.txt','.xlsx'))
    else:
        with open(sp, 'w') as f:
            print(f'mm evaluation of {algorithm}:\nscene                    epe (the lower is better)        1px        3px        5px(the higher is better)',file=f)
            for item in di.keys():
                value = di[item]
                zz = esf(f'{value[0]:.3f}',10)
                xx = esf(f'{value[1]:.2%}',7)
                cc = esf(f'{value[2]:.2%}',7)
                vv = esf(f'{value[3]:.2%}',7)
                print(esf(item,20),f'    {zz}                       {xx}    {cc}    {vv}',file=f)


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

if __name__ == '__main__':
    assert len(sys.argv) == 3,'error input'
    # source_paths = find_folders_with_subfolder(sys.argv[1],keys=['mv1'])
    source_path = sys.argv[1]
    target_path = sys.argv[2]
    # for i in range(len(source_paths)):
    print(source_path,target_path)
    res = mm_evaluate(source_path,target_path)
    now = datetime.datetime.now()
    # 格式化为年月日小时分钟秒
    formatted_date = now.strftime("%Y%m%d_%H%M%S")  # 输出格式类似 '20220429_153142'
    # 使用这个字符串来命名文件
    filename = f"data_{formatted_date}.txt"  # 文件名类似 'data_20220429_153142.txt'
    filename = os.path.join(cur_path,filename)
    show_evaluation(res,'custom_algo',sp=filename)
    # show_evaluation(res,'test',sp='/Users/qhong/Documents/1117test/MM/motionmodel/evaluation/1.txt')