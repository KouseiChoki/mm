'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-04-24 14:03:17
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
from __future__ import print_function, division
import sys,os
dir_flowformer = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir_flowformer)
sys.path.insert(0, dir_flowformer+'/core')
import argparse
import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import importlib
from torch.utils.data import DataLoader
from core.loss import sequence_loss
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger
from core.utils.logger import Logger

from core.Networks import build_network
import core.datasets_kousei as datasets
import imageio
from config_process_train import init_param

train_root = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+'/../../train')
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
    for i in range(len(imgs)):
    # for i in range(len(imgs)):
        write(save_root + f'/image/img{i-1}.png',((imgs[i][...,:3]+1)/2 * 255).astype('uint8'))
    for i in range(len(flo)//2):
        flo_ = flo[i]
        flo_[np.where(vds[i]==0)] = 0
        flo_gt_ = flo_gt[i]
        flo_gt_source_ = flo_gt_source[i]
        write(save_root + f'/mv0/mv_{i}.flo',flo_[...,:2])
        write(save_root + f'/mv0/gt_{i}.flo',flo_gt_[...,:2])
        # write(save_root + f'/mv0/gts_{i}.flo',flo_gt_source_[...,:2])
        write(save_root + f'/mv0/valid_{i}.png',vds[i][...,None]*255)
    for i in range(len(flo)//2,len(flo)):
        flo_ = flo[i]
        flo_gt_ = flo_gt[i]
        flo_[np.where(vds[i]==0)] = 0
        flo_gt_source_ = flo_gt_source[i]
        write(save_root + f'/mv1/mv_{i-len(flo)//2}.flo',flo_[...,:2])
        write(save_root + f'/mv1/gt_{i}.flo',flo_gt_[...,:2])
        # write(save_root + f'/mv1/gts_{i-len(flo)//2}.flo',flo_gt_source_[...,:2])
        write(save_root + f'/mv1/valid_{i-len(flo)//2}.png',vds[i][...,None]*255)


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
        if cfg.restore_ckpt.lower()!='none':
            print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
            tmp = torch.load(cfg.restore_ckpt)
            if int(cfg.initial_config)!=1:
                tmp = {k.replace('update_block.mask.', 'mask.'): v for k, v in tmp.items()}
            model.load_state_dict(tmp, strict=True)

    model.to(torch.device(cfg.gpu_type))
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
            images = images.to(torch.device(cfg.gpu_type))
            flows = flows.to(torch.device(cfg.gpu_type))
            valids = valids.to(torch.device(cfg.gpu_type))

            output = {}
            flow_predictions = model(images, output)
            loss, metrics, NAN_flag = loss_func(flow_predictions, flows, valids, cfg)
            if NAN_flag:
                if single_save:
                    single_save = False
                    sp = train_root+'/train_checkpoints/'+cfg.name
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
                    sp =os.path.join(train_root,'vis',cfg.name,'{:0>6}'.format(total_steps))
                    fgs = flows.detach()[0].cpu().numpy().transpose(0,2,3,1)
                    fg = (flows[0]*valids[0][:,None,:,:]).detach().cpu().numpy().transpose(0,2,3,1)
                    fp = flow_predictions[-1].detach().cpu().numpy()[0].transpose(0,2,3,1)
                    vd = valids[0].detach().cpu().numpy()
                    save_vis(images[0].detach().cpu().numpy(),fp,fg,fgs,sp,vd)

            if total_steps > 0 and total_steps%args.save_iter == 0:
                sp = os.path.join(train_root,'train_checkpoints',cfg.name,'{:0>6}'.format(total_steps))
                mkdir(sp)
                PATH = sp+'/%d_ff.pth' % (total_steps)
                torch.save(model.state_dict(), PATH)

    logger.close()
    PATH = cfg.log_dir + '/final'
    torch.save(model.state_dict(), PATH)
    return PATH

if __name__ == '__main__':
    if len(sys.argv)<=1:
        raise ValueError('[MM ERROR][config]please specify config, usage: mmtrain your_config')
    config_file = sys.argv[1]
    args = init_param(os.path.join(train_root,'config',config_file))
    TORCH_HOME = os.path.join(train_root,'checkpoints')
    os.environ['TORCH_HOME'] = TORCH_HOME
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    args.pkl_root = os.path.join(train_root,'data_pkl')
    try:
        tmp = f'configs.kousei_initial_config_{args.initial_config}'
        cfg_module = importlib.import_module(tmp)
        cfg = cfg_module.get_cfg(args)
    except:
        raise NotImplementedError('error config version')
    train_datasets = ['Sintel','things','Spring','Unreal','Unreal_MRQ']
    # train_datasets =['Unreal_MRQ']
    if args.reload_data:
        print('data reloading..')
        from data_prepare import train_data_prepare
        if not os.path.isdir(args.data_root):
            args.data_root = os.path.join(train_root,args.data_root)
        train_data_prepare(args.data_root,args.pkl_root,train_datasets,input_frames=cfg.input_frames,skip_frames=args.skip_frames,ctype=args.img_type)

    cfg.update(vars(args))
    process_cfg(cfg)
    cfg.log_dir = os.path.join(train_root,'logs',cfg.name)
    loguru_logger.add(os.path.join(cfg.log_dir,'log.txt'))
    loguru_logger.info(cfg)
    torch.manual_seed(1234)
    np.random.seed(1234)
    

    #set data
    # print(cfg)
    train(cfg)

