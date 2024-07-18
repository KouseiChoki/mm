'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-06-26 13:02:24
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

import os,sys
import re
import importlib
import torch
from conversion_tools.algorithm_center import check_and_download_pth_file
#524新加 并行
def get_model(args):
    DEVICE=args.DEVICE
    model_name=args.algorithm
    # if 'gma' in model_name:
    #     dir_mytest = os.path.dirname(os.path.abspath(__file__))+'/../3rd/GMA/core'
    #     sys.path.insert(0, dir_mytest)
    #     from network import RAFTGMA
    #     model = torch.nn.DataParallel(RAFTGMA(args))
    #     model.load_state_dict(torch.load(args.model,map_location=DEVICE).state_dict())
    #     model = model.module
    #     model.to(DEVICE)
    #     model.eval()
    # elif 'depth' == model_name:
    #     sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/../3rd/ZoeDepth')
    #     from zoedepth.models.builder import build_model
    #     from zoedepth.utils.config import get_config
    #     conf = get_config("zoedepth", "infer")
    #     model = build_model(conf)
    #     model.to(DEVICE)
    #     model.eval()
    # elif 'deq' in model_name:
    #     dir_flowformer = os.path.dirname(os.path.abspath(__file__))+'/../3rd/deq-flow/code.v.2.0/core'
    #     sys.path.insert(0, dir_flowformer)
    #     from deq_flow import DEQFlow
    #     args.eval = True
    #     args.wnorm = True
    #     args.f_thres=40
    #     args.f_solver = 'naive_solver'
    #     args.huge = True
    #     args.eval_factor = 1.5
    #     model = torch.nn.DataParallel(DEQFlow(args))
    #     model.load_state_dict(torch.load(args.model,map_location=DEVICE), strict=False)
    #     model = model.module.to(DEVICE)
    #     model.eval()
    # elif 'msraft' in model_name:
    #     sys.path.insert(0, '/home/rg0775/QingHong/MM/MS_RAFT_plus')
    #     sys.path.insert(0, '/home/rg0775/QingHong/MM/MS_RAFT_plus/core')
    #     from raft import RAFT
    #     from config.config_loader import cpy_eval_args_to_config
    #     args.iters=[4, 6, 5, 10]
    #     args.lookup_radius = 4
    #     args.mixed_precision=False
    #     args.cuda_corr=True
    #     args.dataset = 'none'
    #     args.lookup_pyramid_levels = 2
    #     config = cpy_eval_args_to_config(args)
    #     model = RAFT(config)
    #     tl_ = torch.load(args.model)
    #     tl = {k.replace('module.', ''): v for k, v in tl_.items()}
    #     model.load_state_dict(tl)
    #     model.to(DEVICE)
    #     model.eval()
    # elif "flowformer" in model_name:
    #     dir_flowformer = os.path.dirname(os.path.abspath(__file__))+'/../3rd/flowformer/FlowFormer-Official'
    #     sys.path.insert(0, dir_flowformer)
    #     from core.FlowFormer import build_flowformer
    #     args.dataset=''
    #     args.small=False
    #     args.mixed_precision=True
    #     args.alternate_corr=True
    #     if len(re.findall('v\d+',args.weight_file))>0:
    #         tmp = 'configs.kousei_{}'.format(re.findall('v\d+',args.weight_file)[0])
    #     elif len(re.findall('v\d+small',args.weight_file))>0:
    #         tmp = 'configs.kousei_{}'.format(re.findall('v\d+small',args.weight_file)[0])
    #     else:
    #         tmp = 'configs.default'
    #     cfg_module = importlib.import_module(tmp)
    #     cfg = cfg_module.get_cfg()
    #     cfg.update(vars(args))
    #     model = torch.nn.DataParallel(build_flowformer(cfg))
    #     model.load_state_dict(torch.load(cfg.model.replace('small',''),map_location=DEVICE))
    #     model = model.module.to(DEVICE)
    #     model.eval()
    if 'kousei' in model_name:
        dir_flowformer = os.path.dirname(os.path.abspath(__file__))+'/../3rd/Kousei'
        sys.path.insert(0, dir_flowformer)
        from core.Networks import build_network
        if len(re.findall('v\d+',args.weight_file))>0:
            tmp = 'configs.kousei_{}_infer'.format(re.findall('v\d+',args.weight_file)[0])
        else:
            tmp = 'configs.kousei_default'
        # if not os.path.isfile(tmp):
        #     tmp = 'configs.kousei_default'
        cfg_module = importlib.import_module(tmp)
        cfg = cfg_module.get_cfg()
        cfg.update(vars(args))
        if args.compatibility_mode:
            cfg.MOFNetStack.decoder_depth = 4
        ckpt_path = os.path.dirname(os.path.abspath(__file__))+'/../checkpoints/'
        
        if not os.path.isfile(ckpt_path+model_name + '.pth'):
            download_url = ckpt_path+model_name+'.pth'
            md = args.server
            if args.mask_mode:
                if args.mask_type=='bg':
                    md += '/bg'
                elif args.mask_type=='fg':
                    md += '/fg'
                else:
                    md += '/mix'
            else:
                md += '/fm'
            md += '/' + model_name + '.pth'
            print(download_url,md)
            flag = check_and_download_pth_file(download_url,md)
            if not flag:
                raise NotImplementedError(f'[MM ERROR][model]model file not exists:{model_name},please use mmalgo to check')
            # raise NotImplementedError(f'[MM ERROR][model]model file loss!{model_name}')
        model_name = ckpt_path+model_name + '.pth'
        # mps not use mixed_precision
        if args.DEVICE.type == 'mps' or args.DEVICE.type == 'cpu':
            # cfg.MOFNetStack.mixed_precision = False
            cfg.mixed_precision = False
        model = torch.nn.DataParallel(build_network(cfg))
        tmp = torch.load(model_name,map_location='cpu')
        if 'v4' not in model_name:
            tmp = {k.replace('update_block.mask.', 'mask.'): v for k, v in tmp.items()}
        model.load_state_dict(tmp)
        model = model.module.to(DEVICE)
        model.eval()
    else:
        raise NotImplementedError(f'[MM ERROR][model]model file loss!{model_name}')
    return model