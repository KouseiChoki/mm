'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-10-15 11:12:32
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
#coding=utf-8 
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
import numpy as np
import os,sys
import shutil
from tqdm import tqdm
import logging
import time
import configparser
from myutil import *
from algo import * 
import argparse
from torch import distributed as dist
from pre_treatment import *
from cfg_process import init_param,adjust_weight
test_mode = False


##position
cur_time = str(time.gmtime().tm_mon) + '_' + str(time.gmtime().tm_mday)
cur_time_sec = cur_time+'/'+str(time.gmtime().tm_hour)+'/'+str(time.gmtime().tm_sec)
cur_path = os.getcwd()
##log
mkdir(cur_path+'/logs')
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d ] %(levelname)s %(message)s', #时间 文件名 line:行号  levelname logn内容
                    datefmt='%d %b %Y,%a %H:%M:%S', #日 月 年 ，星期 时 分 秒
                    filename= cur_path+'/logs/mylog{}.log'.format(cur_time),
                    filemode='w')
if len(sys.argv) == 1 or (len(sys.argv)==10 and '--ip=127.0.0.1' in sys.argv[0]) or (len(sys.argv)==11 and 'ipykernel_launcher.py' in sys.argv[0]) or 'test' in sys.argv:
    config_file = 'config_test'
    test_mode = True
    cur_path = os.path.dirname(os.path.abspath(__file__))+'/..'
else:
    config_file = sys.argv[1]
    sys.argv.pop(0)
# config_file = 'config'


def init(output):
    print('result file initiated')
    ##前景
    if os.path.exists(output):
        shutil.rmtree(output)

def fusion_mv(front_dict,front_mask_dict,back_dict,back_mask_dict,threshold=30,mv_ref=False):
    res = defaultdict(list)
    for seq in front_dict.keys():
        front,front_mask,back,back_mask = front_dict[seq],front_mask_dict[seq],back_dict[seq],back_mask_dict[seq]
        print('current sequence:{}'.format(seq))
        tmp = []
        for i in tqdm(range(len(front))):
            f,fm,b,bm = front[i],front_mask[i],back[i],back_mask[i]
            if mv_ref:
                f[np.where(fm<threshold)] = 0
                b[np.where(bm<threshold)] = 0
            else:
                f[np.where(fm>threshold)] = 0
                b[np.where(bm>threshold)] = 0
            fusion = f+b
            tmp.append(fusion)
        res[seq] = tmp
    return res

def cal_process_nums(args,scene_num):
    basenum = 2
    # basenum += 4 if args.cal_depth else 0
    # basenum += 4 if args.merge_depth else 0
    # basenum += 2 if args.dump_restored_image else 0
    basenum += 1 if (args.cal_depth and not args.threeD_mode) else 0
    # basenum += 1 if args.scene_change else 0
    # basenum += 1 if args.MM9_format else 0
    return basenum*scene_num

#传参用
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint",default='checkpoints/gma-sintel.pth')
    parser.add_argument('--model_name', help="define model name", default="GMA")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()
    return args


def save_dict_file(filename,data):
    with open(filename,'w') as f:
            f.write('average EPE: {:.2f}, average accuracy :{:.2f}% \n\n'.format(data[0],data[1]*100))


def evaluation(source_datas,ground_truth_datas,masks_dict):
    cal_res = {}
    for key in source_datas.keys():
        print('evaluate:{}'.format(key))
        source_data = source_datas[key]
        target_data = ground_truth_datas[key]
        masks = masks_dict[key]
        n = len(source_data)
        w,h,_ = source_data[0].shape
        
        #first frame is passed
        epe = 0
        acc = 0
        start = 0 if n ==1 else 1
        for i in tqdm(range(start,n)):
            #set groud truth mask  predict mask here is same with ground truth mask
            mask = masks[i]
            if mask.max()>1:
                gt_mask = np.round(mask/255)
            else:
                gt_mask = mask
            gt_mask = gt_mask.astype(bool)
            pd_mask = gt_mask
            td = target_data[i]
            height,width = td[...,0].shape
            # refine average value
            td[...,0]*=width
            td[...,1]*=height

            epe_,acc_ = flow_kitti_mask_error(td[...,0],td[...,1],gt_mask,source_data[i][...,0],source_data[i][...,1],pd_mask)
            epe+=epe_
            acc+=acc_
        if n == 1:
            cal_res[key] = (epe,acc)
        else:
            cal_res[key] = (epe/(n-1),acc/(n-1))
    return cal_res

def init_distributed_mode(cuda=False,mps=False,backend='nccl',gpus=None):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE'])>1:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        if rank==0:
            print('using distributed mode, world_size is:{}'.format(world_size))
    else:
        print('Not using distributed mode')
        return 0,1
    dist_url = 'env://'
    if cuda:
        if not mps:
            cur_device = gpus[local_rank] if gpus else local_rank
            torch.cuda.set_device(int(cur_device))
        dist_backend = backend  # 通信后端，nvidia GPU推荐使用NCCL
        # print('| distributed init (rank {}): {}'.format(
            # rank, dist_url, flush=True))
        dist.init_process_group(backend=dist_backend, init_method=dist_url,
                                world_size=world_size, rank=rank)
        dist.barrier()
    return local_rank,world_size

##load config
config = configparser.ConfigParser()
args = init_param(os.path.join(cur_path,config_file))
if len(sys.argv) >1 and not test_mode:
    args.weight_file = sys.argv[1]
    args = adjust_weight(args)
DEVICE = torch.device("cpu")
if 'cpu' not in args.gpu and args.multi_frame_algo:
    import torch
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    cudnn.deterministic = True
    if 'mps' not in args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device('mps')
args.model = args.weight_file

args.DEVICE = DEVICE
gpus = args.gpu.rstrip().split(',')

if args.IMAGEIO_USERDIR.lower()!='none':
    os.environ['IMAGEIO_USERDIR'] = args.IMAGEIO_USERDIR
if args.TORCH_HUB.lower()!='none':
    os.environ['TORCH_HOME'] = os.path.join(cur_path, args.TORCH_HUB) if '/' not in args.TORCH_HUB else args.TORCH_HUB

args.cur_rank = 1
##mutl process
args.distributed_task = init_distributed_mode(args.DEVICE.type != 'cpu',args.DEVICE.type == 'mps',args.mt_backend,gpus)
args.cur_rank = args.distributed_task[0] + 1
args.use_tqdm = True if args.cur_rank ==1 else False
#tesmode
if test_mode:
    args.root = os.path.join(cur_path,args.root)
    args.model = os.path.join(cur_path,args.model)
    args.output = os.path.join(cur_path,args.output)
    
##预处理文件
if args.cur_rank == 1:
    print('==================================================================================================== ')
    print('now loading image file and mv file')

# 获取图像
    
front_masks = args.front_mask_file.split(',') if args.mask_mode else [None]
for front_mask in front_masks:
    if args.threeD_mode: ##左右眼
        image,right_eye_image= pre_treatment_3D(args,args.root,args.image_file)
        left_mask_only = True if 'disparity' in args.weight_file else False
        masks,right_masks = pre_treatment_3D(args,args.root,front_mask,left_only=left_mask_only) if args.mask_mode else [None,None]
    else:
        image = pre_treatment(args,args.root,args.image_file,single_mode=args.enable_single_input_mode_2D,single_file=args.root_2D)
        masks = pre_treatment(args,args.root,front_mask,single_mode=args.enable_single_input_mode_2D,single_file=args.root_2D,multi_dir=args.multi_mask_file) if args.mask_mode else None
        right_eye_image = None
        right_masks = None

    if args.cur_rank == 1:
        print('pre-treatment finished,the number of sequence is {}, they are {}'.format(len(image.keys()),image.keys()))
        print('now starting {} algorithm'.format(args.algorithm))
    cur_process = 0
    scene_num = len(image.keys())
    total_process= cal_process_nums(args,scene_num)
    # 计算前景mv结果
    if args.cur_rank == 1:print('running MM9 optical flow')
    mask_extra_info = None if len(front_masks) == 1 else front_mask
    optical_flow(args,image,masks,right_images=right_eye_image,right_masks=right_masks,mask_extra_info=mask_extra_info)

if not test_mode:
    print('======================================== worker {} is finished! ======================================= totalworker={}'.format(args.cur_rank,int(os.environ['WORLD_SIZE'])))



# if args.scene_change and args.cur_rank == 1:
#     time.sleep(5)
#     print('scene change detection({}/{})'.format(cur_process,total_process))
#     cur_process+=scene_num
#     scene_change_result = scene_change_detect(image,DEVICE)
#     write_txts(scene_change_result,args.output)


# if args.MM9_format and args.cur_rank == 1:
#     print('MM9 format merging...({}/{})'.format(cur_process,total_process))
#     cur_process+=scene_num
#     combine_result(args,front_mask)
# if args.cur_rank == 1 and args.time_cost:
#     c_t,speed = [],[]
#     myiter = None
#     if args.mask_mode:
#         myiter = mv0_front_result_cost.items()
#     else:
#         myiter = mv0_origin_result_cost.items()
#     for key,value in myiter:
#         c_t.append(value[0])
#         speed.append(value[1])
#     print(algorithm)
#     print('cost time = {}'.format(c_t))
#     print('speed = {}'.format(speed))
