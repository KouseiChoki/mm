from __future__ import print_function, division
import sys,os
dir_flowformer = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir_flowformer)
sys.path.insert(0, dir_flowformer+'/core')
import argparse
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from core import optimizer

from core.loss import sequence_loss
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger
from core.utils.logger import Logger

from core.Networks import build_network
import core.datasets_kousei as datasets
import imageio


def write(path,flow,compress_method='EXR16'):
    #mkdir
    if not os.path.exists(os.path.dirname(path)):  # 判断目录是否存在
        os.makedirs(os.path.dirname(path),exist_ok=False) 
    if '.exr' in path:
        if flow.shape[2] == 2:
            flow = np.insert(flow,2,0,axis=2)
        if flow.shape[2] == 3:
            flow = np.insert(flow,2,0,axis=2)
        if compress_method.lower() == 'none':
            imageio.imwrite(path,flow[...,:4],flags=imageio.plugins.freeimage.IO_FLAGS.EXR_NONE)
        else:
            imageio.imwrite(path,flow[...,:4],flags=imageio.plugins.freeimage.IO_FLAGS.EXR_ZIP|imageio.plugins.freeimage.IO_FLAGS.EXR_FLOAT)
    elif '.flo' in path:
        write_flo_file(flow[...,:2],path)
    else:
        if len(flow.shape) == 2:
            flow = np.repeat(flow[...,None],3,axis=2)
        if flow.shape[2] == 2:
            flow = np.insert(flow,2,0,axis=2)
        cv2.imwrite(path,flow[...,:3][...,::-1])

def write_flo_file(flow, filename): # flow: H x W x 2
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    if flow.ndim == 4: # has batch
        flow = flow[0]

    outpath = os.path.dirname(filename)
    if outpath != '' and not os.path.isdir(outpath):
        os.makedirs(outpath)

    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    height, width = flow.shape[:2]
    magic.tofile(f)
    np.int32(width).tofile(f)
    np.int32(height).tofile(f)
    data = np.float32(flow).flatten()
    data.tofile(f)
    f.close()



def save_vis(imgs,flo,flo_gt,flo_gt_source,save_root,vds):
    imgs = imgs.transpose(0,2,3,1)
    for i in range(1,len(imgs)-1):
    # for i in range(len(imgs)):
        write(save_root + f'/image/img{i-1}.png',((imgs[i][...,:3]+1)/2 * 255).astype('uint8'))
    for i in range(len(flo)//2):
        flo_ = flo[i]
        flo_gt_ = flo_gt[i]
        flo_gt_source_ = flo_gt_source[i]
        write(save_root + f'/mv0/mv_{i}.flo',flo_[...,:2])
        write(save_root + f'/mv0/gt_{i}.flo',flo_gt_[...,:2])
        write(save_root + f'/mv0/gts_{i}.flo',flo_gt_source_[...,:2])
        write(save_root + f'/mv0/valid_{i}.png',vds[i][...,None]*255)
    for i in range(len(flo)//2,len(flo)):
        flo_ = flo[i]
        flo_gt_ = flo_gt[i]
        write(save_root + f'/mv1/mv_{i}.flo',flo_[...,:2])
        write(save_root + f'/mv1/gt_{i}.flo',flo_gt_[...,:2])
        write(save_root + f'/mv1/gts_{i}.flo',flo_gt_source_[...,:2])
        write(save_root + f'/mv1/valid_{i}.png',vds[i][...,None]*255)


try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)


from torchvision.utils import save_image

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(cfg):

    loss_func = sequence_loss
    if cfg.use_smoothl1:
        print("[Using smooth L1 loss]")
        loss_func = sequence_loss_smooth

    model = nn.DataParallel(build_network(cfg))
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        tmp = torch.load(cfg.restore_ckpt)
        tmp = {k.replace('update_block.mask.', 'mask.'): v for k, v in tmp.items()}
        model.load_state_dict(tmp, strict=True)

    model.cuda()
    model.train()

    train_loader = datasets.fetch_dataloader(cfg)
    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

    total_steps = -1
    pre_record = -100000
    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger = Logger(model, scheduler, cfg)

    should_keep_training = True
    single_save=True
    avgepe = 50
    epe_arr = []
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            images, flows, valids = [x for x in data_blob]
            images,flows,valids  = data_blob
            images = images.cuda() 
            flows = flows.cuda()
            valids = valids.cuda()

            output = {}
            flow_predictions = model(images, output)
            loss, metrics, NAN_flag = loss_func(flow_predictions, flows, valids, cfg)
            if NAN_flag:
                if single_save:
                    single_save = False
                    sp = os.path.dirname(os.path.abspath(__file__))+'/train_checkpoints/'+cfg.name
                    
                    mkdir(sp)
                    PATH = sp+'/%d_last_ff.pth' % (total_steps)
                    torch.save(model.state_dict(), PATH)

                print('error!')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            metrics.update(output)
            logger.push(metrics)
            
            total_steps += 1

            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break

            if len(epe_arr) == 100:
                avgepe = sum(epe_arr)/100
                epe_arr = []

            if total_steps % args.vis_iter == 0 or metrics['epe']>avgepe*2:
                if total_steps - pre_record >= min(args.vis_iter,50000): #interval
                    pre_record = total_steps
                    sp =os.path.join(os.path.dirname(os.path.abspath(__file__)),'vis',cfg.name,'{:0>6}'.format(total_steps))
                    fg = (flows[0]*valids[0][:,None,:,:]).detach().cpu().numpy().transpose(0,2,3,1)
                    fgs = flows.detach()[0].cpu().numpy().transpose(0,2,3,1)
                    fp = flow_predictions[-1].detach().cpu().numpy()[0].transpose(0,2,3,1)
                    vd = valids[0].detach().cpu().numpy()
                    save_vis(images[0].detach().cpu().numpy(),fp,fg,fgs,sp,vd)

            if total_steps > 0 and total_steps%args.save_iter == 0:
                sp = os.path.dirname(os.path.abspath(__file__))+'/train_checkpoints/'+cfg.name
                mkdir(sp)
                PATH = sp+'/%d_ff.pth' % (total_steps)
                torch.save(model.state_dict(), PATH)

    logger.close()
    PATH = cfg.log_dir + '/final'
    torch.save(model.state_dict(), PATH)

    return PATH
# python mytrain.py  --name 1109  --reload_data
# python mytrain.py --name 0822_9400_fg --save_iter 1000

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    TORCH_HOME = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+'/../../checkpoints')
    os.environ['TORCH_HOME'] = TORCH_HOME
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='MM9', help="name your experiment")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--reload_data',action='store_true')
    parser.add_argument('--data_root', default='/tt/nas/qhong/opticalflow/optical_flow_datasets', help="data root")
    parser.add_argument('--vis_iter', type=int, default=5000)
    parser.add_argument('--save_iter', type=int, default=20000)
    parser.add_argument('--kouseiversion', type=int, default=3)
    parser.add_argument('--masktype', default ='frame')

    parser.add_argument('--robin',action='store_true')
    args = parser.parse_args()
    args.pkl_root = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+'/data_pkl')
    if args.robin:
        from configs.kousei_robin import get_cfg
    else:
        if args.kouseiversion == 0:
            if args.masktype == 'frame':
                from configs.kousei_v0_train import get_cfg
            elif args.masktype == 'fg':
                from configs.kousei_v0_fg_train import get_cfg
            else:
                from configs.kousei_v0_bg_train import get_cfg
        elif args.kouseiversion == 1:
            if args.masktype == 'frame':
                from configs.kousei_v1_train import get_cfg
            elif args.masktype == 'fg':
                from configs.kousei_v1_fg_train import get_cfg
            else:
                from configs.kousei_v1_bg_train import get_cfg
        elif args.kouseiversion == 3:
            if args.masktype == 'frame':
                from configs.kousei_v3_train import get_cfg
            elif args.masktype == 'fg':
                from configs.kousei_v3_fg_train import get_cfg
            else:
                from configs.kousei_v3_bg_train import get_cfg
        else:
            print('unsupport')
            sys.exit(0)
    # from configs.kousei_v3_train import get_cfg
    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)

    torch.manual_seed(1234)
    np.random.seed(1234)
    train_datasets = ['Sintel','things','Unreal','Spring','Unreal_MRQ']

    #set data
    if cfg.reload_data:
        print('data reloading..')
        from data_prepare import train_data_prepare
        train_data_prepare(args.data_root,args.pkl_root,train_datasets,skip_frames=args.skip_frames)

    train(cfg)

