'''
Author: Qing Hong
Date: 2024-04-11 13:55:07
LastEditors: Qing Hong
LastEditTime: 2026-01-20 13:44:11
Description: file content
'''
import os,sys
cur_path = os.path.dirname(os.path.abspath(__file__))
from tqdm import tqdm
sys.path.insert(0, cur_path+'/../algo')
sys.path.insert(0, cur_path+'/..')
from file_utils import read,write
from openpyxl import Workbook
import datetime
import re
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import argparse
from evaluate_core import mm_evaluate,show_evaluation

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
        if int(number) == int(target_number):
            return path
    
    return None

def cal_psnr(image,target_image):
    psnr=peak_signal_noise_ratio(image,target_image)
    return -psnr

def cal_ssim(image,target_image):
    score, diff = ssim(image, target_image,channel_axis=2,full=True)
    return score,diff


def evaluate(src_path,gt_path,name=''):
    srcs = jhelp_file(src_path)
    gts = jhelp_file(gt_path)
    if len(srcs)==0:
        print('no src datas, try to find default data')
        srcs = jhelp_file(os.path.join(cur_path+'/..',src_path))
    if len(gts)==0:
        print('no gt datas, try to find default data')
        gts = jhelp_file(os.path.join(cur_path+'/..',gt_path))
    # assert len(mvs)>skip*2 and len(gts) > skip*2 and len(gts)==len(mvs),'evluation data length should >=2'
    # mvs = mvs[skip:-skip]
    # gts = gts[skip:-skip]
    # if masks is not None:
        # masks = jhelp_file(masks)
        # masks = masks[skip:-skip]
    epe_all,psnr_all,ssim_score_all,diff_all = 0,0,0,0
    totalnum = len(srcs)
    for i in tqdm(range(totalnum),desc=f'calculating {name}'):
        src = read(srcs[i],type='image')
        # gt = read(gts[i],type='flo')[...,:2]
        gt_ = find_matching_files(srcs[i],gts)
        gt = read(gt_,type='image')
        metrics = evaluate_single_frame(src,gt)
        epe_all += metrics['epe']
        psnr_all  += metrics['psnr']
        ssim_score_all  += metrics['ssim_score']
        diff_all  += metrics['diff']
    return epe_all/totalnum,psnr_all/totalnum,ssim_score_all/totalnum,diff_all/totalnum

def evaluate_single_frame(src,gt,valid=None,norm=True):
    # if norm:
    #     h,w,_ = mv.shape
    #     mv[...,0] *= w
    #     mv[...,1] *= h
    #     gt[...,0] *= w
    #     gt[...,1] *= h
    epe = np.sqrt(np.sum((src - gt)**2,axis=-1))
    epe = epe.view()
    psnr = cal_psnr(src,gt)
    ssim_score, diff = cal_ssim(src,gt)
    metrics = {
        'epe': epe.mean().item(),
        'psnr': psnr,
        'ssim_score': ssim_score,
        'diff': diff,
    }
    return metrics

def psnr_evaluate(sources,gts,basename='image',obj_mode=False,skip=1,fg=True):
    result = {}
    for i in sources.keys():
        scene = sources[i]
        gt = gts[i]
        if not os.path.isdir(gt):
            continue
        result[f'{i}_res'] = evaluate(scene,gt,i)
            # if os.path.isdir(f'{gt}/mv0'):
                # result[f'{scene_name}_mv0_{s}'] = evaluate(f'{scene}/{mv0_name}',f'{gt}/mv0',skip=skip,speed=speed,masks=masks,fg=fg)
    return result

def esf(text, min_length=10):
    return f"{text:<{min_length}}"

def show_evaluation_psnr(di,algorithm='mm',sp='',excel=True):
    # 创建一个新的工作簿和工作表
    wb = Workbook()
    ws = wb.active
    # 更改工作表名称
    ws.title =algorithm
    # 写入一些数据
    ws.append(['Scene', 'epe', 'psnr', 'ssim_score', 'diff'])
    for item in di.keys():
        value = di[item]
        ws.append([item, f'{value[0]:.3f}',f'{value[1]:.3f}',f'{value[2]:.3f}',f'{value[3].mean():.3f}'])
    sp = sp.replace('.txt','.xlsx')
    print(f'result saving in :{sp}')
    wb.save(sp)


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

def process_data(source_path,image_name):
    if os.path.basename(source_path) == image_name:
        source_path = os.path.dirname(source_path)
    prepares = []
    if not os.path.isdir(os.path.join(source_path,image_name)):
        prepares = []
        for dirpath, dirnames, filenames in os.walk(source_path):
            if image_name in dirnames:
                prepares.append(os.path.join(dirpath,image_name))
    else:
        prepares = [os.path.join(source_path,image_name)]
    res = {}
    for p in prepares:
        name = os.path.basename(os.path.dirname(p))
        res[name] = p
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source','--root', required=True,dest='root',type=str ,help="source root")
    parser.add_argument('--target','--gt', required=True,dest='gt',type=str ,help="ground truth root")
    parser.add_argument('--mv', action='store_true',help="motion vector mode")
    args = parser.parse_args()
    source_path = args.root
    target_path = args.gt
    if not args.mv:
        # source_paths = find_folders_with_subfolder(sys.argv[1],keys=['mv1'])
        image_name = 'image'
        # for i in range(len(source_paths)):
        print('source_path:',source_path)
        print('target_path:',target_path)
        source = process_data(source_path,image_name)
        target = process_data(target_path,image_name)


        res = psnr_evaluate(source,target)
        now = datetime.datetime.now()
        # 格式化为年月日小时分钟秒
        formatted_date = now.strftime("%Y%m%d_%H%M%S")  # 输出格式类似 '20220429_153142'
        # 使用这个字符串来命名文件
        filename = f"data_{formatted_date}.txt"  # 文件名类似 'data_20220429_153142.txt'
        filename = os.path.join(cur_path,filename)
        show_evaluation_psnr(res,'evaluate_result',sp=filename)
        # show_evaluation(res,'test',sp='/Users/qhong/Documents/1117test/MM/motionmodel/evaluation/1.txt')
    else:
        res,algo = mm_evaluate(source_path,target_path)
        now = datetime.datetime.now()
        # 格式化为年月日小时分钟秒
        formatted_date = now.strftime("%Y%m%d_%H%M%S")  # 输出格式类似 '20220429_153142'
        # 使用这个字符串来命名文件
        filename = f"data_{formatted_date}.txt"  # 文件名类似 'data_20220429_153142.txt'
        filename = os.path.join(cur_path,filename)
        show_evaluation(res,algo,sp=filename)