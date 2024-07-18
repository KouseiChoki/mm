'''
Author: Qing Hong
Date: 2024-04-11 13:55:07
LastEditors: Qing Hong
LastEditTime: 2024-06-17 15:50:12
Description: file content
'''
import os,sys
cur_path = os.path.dirname(os.path.abspath(__file__))
from tqdm import tqdm
sys.path.insert(0, cur_path+'/../algo')
from file_utils import read
from openpyxl import Workbook


import numpy as np
def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))



def evaluate(mv_path,gt_path,skip=1,speed=[0,1],masks=None,fg=True):
    mvs = jhelp_file(mv_path)
    gts = jhelp_file(gt_path)
    assert len(mvs)>skip*2 and len(gts) > skip*2 and len(gts)==len(mvs),'evluation data length should >=2'
    mvs = mvs[skip:-skip]
    gts = gts[skip:-skip]
    if masks is not None:
        masks = jhelp_file(masks)
        masks = masks[skip:-skip]
    epe_all,px1_all,px3_all,px5_all = 0,0,0,0
    for i in range(len(mvs)):
        mv = read(mvs[i],type='flo')[...,:2]
        gt = read(gts[i],type='flo')[...,:2]
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
    return epe_all/len(mvs),px1_all/len(mvs),px3_all/len(mvs),px5_all/len(mvs)
    
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
            if os.path.isdir(f'{gt}/mv0'):
                result[f'{scene_name}_mv0_{s}'] = evaluate(f'{scene}/{mv0_name}',f'{gt}/mv0',skip=skip,speed=speed,masks=masks,fg=fg)
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

if __name__ == '__main__':
    res = mm_evaluate('/Users/qhong/Documents/1117test/MM/motionmodel/evaluation/tmp', '/Users/qhong/Documents/1117test/MM/motionmodel/evaluation/data',True)
    show_evaluation(res,'test',sp='/Users/qhong/Documents/1117test/MM/motionmodel/evaluation/1.txt')