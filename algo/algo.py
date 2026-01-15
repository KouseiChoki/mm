'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-10-15 11:21:11
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


import cv2
import numpy as np
import os,sys
from tqdm import tqdm
import imageio
import torch
from myutil import *
from collections import defaultdict
import re
from kousei_model import get_model
from pair_data_creation import get_pairs
from kousei_inference import optical_flow_algo
FLOAT = None

# torch.backends.mps.is_available()
def overlay_optical_flow(imgs,args,model,initial_flow=None,dump_name = None,mv_offset=None):
    if imgs[0].shape[2]!=4:
        raise RuntimeError('[MM ERROR][config]config setting error,no mask to run mask method!')
    args.restrain = False
    initial_flow_fg,initial_flow_bg = None,None
    
    # if initial_flow is not None: ##初始光流分割
    #     if mask.shape!=initial_flow.shape[:2]:
    #         mask_low = cv2.resize(mask,(initial_flow.shape[1],initial_flow.shape[0]),interpolation=cv2.INTER_NEAREST)
    #     initial_flow_fg = initial_flow.copy()
    #     initial_flow_bg = initial_flow.copy()
    #     threshold = 1e-9 if args.threshold==0 else args.threshold==0
    #     initial_flow_fg[np.where(mask_low<threshold)] = 0
    #     initial_flow_bg[np.where(mask_low>=threshold)] = 0 
    dump_name_fg = dump_name
    dump_name_bg = dump_name
    threshold = args.threshold
    if args.mask_type == 'fg':
        imgs_fg = np.copy(imgs)
        opt_,initial_flow = optical_flow_core(imgs_fg,args,model,initial_flow_fg,dump_name=dump_name_fg,mv_offset=mv_offset)
        for i in range(len(opt_)):
            mask = imgs_fg[1+i%(args.num_frames-2)][...,-1]
            if threshold==1:
                opt_[i][np.where(mask<threshold)] = 0
            else:
                opt_[i][np.where(mask<=threshold)] = 0
    elif args.mask_type == 'bg':
        imgs_bg = np.copy(imgs)
        for i in range(len(imgs_bg)):
            imgs_bg[i][...,-1] = 1-imgs_bg[i][...,-1]
        opt_,initial_flow  = optical_flow_core(imgs_bg,args,model,initial_flow_bg,dump_name=dump_name_bg,mv_offset=mv_offset)
        for i in range(len(opt_)):
            mask = imgs_bg[1+i%(args.num_frames-2)][...,-1]
            if threshold==1:
                opt_[i][np.where(mask<threshold)] = 0
            else:
                opt_[i][np.where(mask<=threshold)] = 0
    elif args.mask_type == 'mix':
        imgs_fg = np.copy(imgs)
        opt_,initial_flow = optical_flow_core(imgs_fg,args,model,initial_flow_fg,dump_name=dump_name_fg,mv_offset=mv_offset)
        for i in range(len(opt_)):
            mask = imgs_fg[1+i%(args.num_frames-2)][...,-1]
            if threshold==1:
                opt_[i][np.where(mask<threshold)] = 0
            else:
                opt_[i][np.where(mask<=threshold)] = 0

        imgs_bg = np.copy(imgs)
        for i in range(len(imgs_bg)):
            imgs_bg[i][...,-1] = 1-imgs_bg[i][...,-1]
        opt_bg,initial_flow  = optical_flow_core(imgs_bg,args,model,initial_flow_bg,dump_name=dump_name_bg,mv_offset=mv_offset)
        for i in range(len(opt_bg)):
            mask = imgs_bg[1+i%(args.num_frames-2)][...,-1]
            if threshold==1:
                opt_bg[i][np.where(mask<threshold)] = 0
            else:
                opt_bg[i][np.where(mask<=threshold)] = 0

            #go merge
            opt_[i] += opt_bg[i]
    else:
        imgs_fg = np.copy(imgs)
        opt_fg,initial_flow_fg = optical_flow_core(imgs_fg,args,model,initial_flow_fg,dump_name=dump_name_fg)
        imgs_bg = np.copy(imgs)
        for i in range(len(imgs_bg)):
            imgs_bg[i][...,-1] = 1-imgs_bg[i][...,-1]
        opt_bg,initial_flow_bg = optical_flow_core(imgs_bg,args,model,initial_flow_bg,dump_name=None)
        opt_ = [None] * len(opt_fg)
        for i in range(len(opt_fg)):
            mask_fg = imgs_fg[1+i%(args.num_frames-2)][...,-1]
            mask_bg = imgs_bg[1+i%(args.num_frames-2)][...,-1]
            opt_fg[i][np.where(mask_fg<=0.5)] = 0
            opt_bg[i][np.where(mask_bg<0.5)] = 0
            opt_[i] = opt_fg[i] + opt_bg[i]
    # if initial_flow_fg is not None and initial_flow_bg is not None:
    #     initial_flow = initial_flow_fg + initial_flow_bg
    opt = np.concatenate([i[None] for i in opt_])
    return opt,initial_flow

def optical_flow_core(imgs,args,model,initial_flow=None,dump_name = None,mv_offset=None):
    #mask refine
    
    # mv_offset = [[0,0]]*(imgs.shape[0]-1)
    # #check use or not
    # if args.use_bounding_box:
    #     pad_info = pad_bounding_box(imgs,args)
    #     lx_,ly_,rx_,ry_ = pad_info[0]
    #     new_x_size = rx_-lx_
    #     new_y_size = ry_-ly_
    #     new_imgs = np.zeros((imgs.shape[0],new_y_size,new_x_size,imgs[0].shape[-1]))
    #     for i in range(new_imgs.shape[0]):
    #         lx,ly,rx,ry = pad_info[i]
    #         new_imgs[i] = imgs[i][ly:ry,lx:rx]
    #     imgs = new_imgs
    #     for i in range(imgs.shape[0]-1):
    #         mv_offset[i] = [pad_info[i+1][0]-pad_info[i+1][0],pad_info[i+1][1]-pad_info[i+1][1]]
    
    
    
    #     if x is None:
    #         use_bounding_box = False
    #     else:
    #         p1_lx,p1_ly,p1_rx,p1_ry = x
    #         tmp_x = (p1_rx - p1_lx)/pre.shape[1]
    #         tmp_y = (p1_ry - p1_ly)/pre.shape[0]
    #         use_bounding_box = args.bounding_box_mode>=3 and (tmp_x<0.25 and tmp_y<0.25)

    # if use_bounding_box:
    #     x,y = pad_bounding_box(pre[...,-1],cur[...,-1],args)
    #     if x is None:
    #         bouding_box_pos = [0,0,pre.shape[1],cur.shape[0]]
    #     else:
    #         p1_lx,p1_ly,p1_rx,p1_ry = x
    #         cf_lx,cf_ly,cf_rx,cf_ry = y
    #         pre = pre[p1_ly:p1_ry,p1_lx:p1_rx,:3]
    #         cur = cur[cf_ly:cf_ry,cf_lx:cf_rx,:3]
    #         mv_offset = [cf_lx-p1_lx,cf_ly-p1_ly]
    #         bouding_box_pos=[p1_lx,p1_ly,p1_rx,p1_ry]
        
    resize_rate_x,resize_rate_y = get_resize_rate(args.algorithm,imgs[0].shape[0],imgs[0].shape[1])
    if args.compatibility_mode: #for frame bugs
        resize_rate_x = 1
        resize_rate_y = 1
    resize_rate_x *= args.resize_rate_x
    resize_rate_y *= args.resize_rate_y
    pre_size = imgs[0].shape
    if resize_rate_x!=1 or resize_rate_y!=1:
        imgs = [cv2.resize(img,(round(img.shape[1]*resize_rate_x),round(img.shape[0]*resize_rate_y)),interpolation=cv2.INTER_CUBIC) for img in imgs]
    for i in range(len(imgs)):
        img = imgs[i]
        if img.shape[2] == 4:
            if 'disparity' in args.weight_file and i == len(imgs)//2:
                continue
            mask_  = imgs[i][...,-1] if imgs[i].shape[2] == 4 else None
            imgs[i][...,:3] = restrain(imgs[i][...,:3],mask_,args.threshold,value=-100)
    if dump_name is not None:
        name = os.path.basename(dump_name)
        for i in range(len(imgs)):
            tmp = int(re.findall(r'\d+', name)[-1])
            dump_char_file = os.path.join(os.path.dirname(dump_name),'../dumped/image/{}/{:0>8}.png'.format(tmp,tmp+i))
            write(dump_char_file,imgs[i][...,:3])
        ##bouding box 处理 

    if not args.pass_mv:
        opt,opt_initial = optical_flow_algo(imgs,args,model,flow_prev=initial_flow) 
        if resize_rate_x!=1 or resize_rate_y!=1:
            opt = F.interpolate(opt, size=(pre_size[0],pre_size[1]), mode='bilinear', align_corners=False)
            opt[:,0,:,:] /= resize_rate_x
            opt[:,1,:,:] /= resize_rate_y
        if len(opt.shape) == 4:
            opt = np.transpose(opt.numpy(),(0,2,3,1))
        # if use_bounding_box:
        #     tmp_flow = np.zeros((original_size[0],original_size[1],2)).astype('float32')
        #     opt[...,0] += mv_offset[0]
        #     opt[...,1] += mv_offset[1]
        #     tmp_flow[bouding_box_pos[1]:bouding_box_pos[3],bouding_box_pos[0]:bouding_box_pos[2]] = opt
        #     opt = tmp_flow
        if args.restrain:
            for i in range(1,opt.shape[0]//2 +1):
                mask_pre = imgs[i][...,-1] if imgs[i].shape[2] == 4 else None
                opt[i-1] = restrain(opt[i-1],mask_pre,args.threshold)
                opt[i-1+opt.shape[0]//2] = restrain(opt[i-1+opt.shape[0]//2],mask_pre,args.threshold)
        if args.use_bounding_box and args.mask_type == 'fg': #0 0 0 1 1 1
            for i in range(opt.shape[0]):
                #mv0
                if i < opt.shape[0]//2:
                    opt[i][...,0] += mv_offset[i+1][0]
                    opt[i][...,1] += mv_offset[i+1][1]
                else:
                    opt[i][...,0] -= mv_offset[i-opt.shape[0]//2][0]
                    opt[i][...,1] -= mv_offset[i-opt.shape[0]//2][1]

        opt = reprocessing(opt,args)
        return opt,opt_initial
        

def optical_flow_pre_processing(imgs,valid,model,save_name,datas_op,args,initial_flow=None,dump=False,disparity_inverse=False):
    film_border = args.film_border_arr
    dump_name = None 
    if dump:
        for item in save_name:
            if item is not None:
                dump_name = item
                break
    args.h,args.w,_ = imgs[0].shape
    #crop film board
    if sum(film_border)>0:
            imgs = [img[film_border[0]:args.h-film_border[1],film_border[2]:args.w-film_border[3]] for img in imgs]
    pass_calculation = False
    # #crop bounding box
    imgs = np.array(imgs)
    pre_bounding_size = imgs.shape[1:]
    mv_offset = [[0,0]]*(imgs.shape[0]-1)
    #check use or not
    if args.use_bounding_box and args.mask_type == 'fg':
        pad_info = pad_bounding_box(imgs,args)
        lx_,ly_,rx_,ry_ = pad_info[0]
        new_x_size = rx_-lx_
        new_y_size = ry_-ly_
        new_imgs = np.zeros((imgs.shape[0],new_y_size,new_x_size,imgs[0].shape[-1]))
        for i in range(new_imgs.shape[0]):
            lx,ly,rx,ry = pad_info[i]
            new_imgs[i] = imgs[i][ly:ry,lx:rx]
        imgs = new_imgs
        for i in range(imgs.shape[0]-1):
            mv_offset[i] = [pad_info[i+1][0]-pad_info[i][0],pad_info[i+1][1]-pad_info[i][1]]
    

    for tmp in imgs: #pass
        if tmp.sum()==0:
            ResourceWarning('[MM ERROR][image]The input image is illegal! Please check if the scene change in the image is correct !')
            pass_calculation = True
    #1030testmode
    if args.cal_disparity and args.disparity_only:
        pass_calculation = True
    if pass_calculation:
        opt = np.zeros((2*(args.num_frames-2),args.h,args.w,2))
    else:
        if '-mask' in args.algorithm and args.mask_mode:
            opt,initial_flow = overlay_optical_flow(imgs,args,model,initial_flow,dump_name=dump_name,mv_offset=mv_offset)
        else:
            opt,initial_flow = optical_flow_core(imgs,args,model,initial_flow,dump_name=dump_name)
        
    if args.cal_disparity:#计算disparity
        tmp = [None] * len(opt)
        if sum(film_border)>0:
            datas_op = [data[film_border[0]:args.h-film_border[1],film_border[2]:args.w-film_border[3]] for data in datas_op]
        for i in range(1,args.num_frames-1):
            if args.num_frames == 3:
                if 'disparity' in args.weight_file: #kata mode
                    dtmp = restrain(imgs[i][...,:3],imgs[i][...,-1],args.threshold,value=-100)
                else:
                    dtmp = imgs[i]
                combination_datas = [datas_op[i],dtmp,datas_op[i]]
                # for dd in range(2):
                #     write(f'/home/rg0775/QingHong/result/test/{dd}.exr',combination_datas[dd])
            elif args.num_frames == 5:
                combination_datas = [imgs[i],imgs[i],imgs[i],datas_op[i],datas_op[i]]
            else:
                combination_datas = [imgs[i]]
                for _ in range((args.num_frames-1)//2):
                    combination_datas.insert(0,imgs[i])
                    combination_datas.append(datas_op[i])
            assert len(combination_datas) == args.num_frames
            disparity,_ = optical_flow_core(combination_datas,args,model)
            if 'disparity' in args.weight_file:
                disparity_lr = disparity[0] #mv0
            else:
                disparity_lr = -disparity[(args.num_frames-1)//2-1] if disparity_inverse else disparity[(args.num_frames-1)//2] #mv0
            # disparity_lr = (disparity_lr + args.depth_range/2).clip(0, args.depth_range) / args.depth_range
            # opt_l = np.insert(opt[i-1],2,0,axis=2)
            opt_l  = np.concatenate((opt[i-1],disparity_lr[...,1:]),axis=2)
            opt_l  = np.concatenate((opt_l,disparity_lr[...,:1]),axis=2)
            # opt_r = np.insert(opt[i-1+(args.num_frames-2)],2,0,axis=2)
            opt_r  = np.concatenate((opt[i-1+(args.num_frames-2)],disparity_lr[...,1:]),axis=2)
            opt_r  = np.concatenate((opt_r,disparity_lr[...,:1]),axis=2)
            if args.restrain_all:
                mask_l = imgs[i][...,-1] if imgs[i].shape[2] == 4 else None
                opt_l = restrain(opt_l,mask_l,args.threshold)
                opt_r = restrain(opt_r,mask_l,args.threshold)
            if args.disparity_only:
                for z in range(len(valid)):
                    valid[z]=True
                opt_l = testmode_for_disparity(opt_l,args)
                opt_r = testmode_for_disparity(opt_r,args)
            tmp[i-1] = opt_l[None]
            tmp[i-1+(args.num_frames-2)] = opt_r[None]
        opt = np.concatenate(tmp)
    
    #recover bounding box
    if args.use_bounding_box and args.mask_type == 'fg':
        tmp = np.zeros((opt.shape[0],pre_bounding_size[0],pre_bounding_size[1],2))
        for i in range(tmp.shape[0]):
            lx,ly,rx,ry = pad_info[i%(len(pad_info)-2)+1]
            tmp[i][ly:ry,lx:rx] = opt[i]
        opt = tmp

    #recover film board
    if sum(film_border)>0:  
        opt_ = np.zeros((2*(args.num_frames-2),args.h,args.w,opt.shape[-1])).astype('float32')
        for i in range(2*(args.num_frames-2)):
            opt_[i,film_border[0]:opt.shape[1]+film_border[0],film_border[2]:opt.shape[2]+film_border[2]] = opt[i]
        opt = opt_
    save_mv_file(save_name,opt,valid,args)
    return initial_flow

def testmode_for_disparity(opt,baseline=20,focal_length=1):
    #copy 4 channel to x channel
    assert opt.shape[2] == 4,'error'
    h,w,_ = opt.shape
    #depth = (baseline * focal_length) / disparity)
    opt[...,0] = opt[...,3] * w
    opt[...,3] = (baseline * focal_length) / opt[...,0]
    return opt
'''
description: 光流计算代码
param {*} args
param {*} images 输入图片dict
param {*} append 保存名
param {*} front_mask_dict 前景mask的dict
param {*} zero_one 是否为01相位
param {*} using_mask 使用哪种mask
return {*} 光流结果地址dict
'''
# def optical_flow(args,images,masks,right_images=None,right_masks=None,mask_extra_info=None):
#     algorithm_check(args.algorithm,args.all_algorithm)
#     if '-v' not in args.algorithm:
#         algorithm_pick(images,args) #pick algo base on image size
#     save_res = []
#     model = None if cpu_algorithm(args.algorithm) else get_model(args)
#     cams = ['left','right'] if args.threeD_mode else ['monocular']
#     imgs_op = None
#     for seq_,data in images.items():
#         seq = seq_
#         seq_images,[start,end] = data
#         seq_images_right = None if right_images is None else right_images[seq][0]
#         input_valid_check(seq_images,args)
#         #film border checker
#         if 'auto' in args.film_border.lower():
#             from conversion_tools.filmborder_detection import cal_border_length 
#             args.film_border_arr = cal_border_length(seq_images[0])
#         else:
#             args.film_border_arr = [int(num) for num in args.film_border.split(',')]
#         for step in args.frame_step:
#             for cam in cams:
#                 initial_opt = None #初始光流
#                 task_name = '{}_{}_'.format(args.algorithm_fullname,cam) if masks is None else '{}_{}_object_'.format(args.algorithm_fullname,cam)
#                 if mask_extra_info is not None and cam !='right':
#                     if '/' in mask_extra_info:
#                         mask_extra_info = mask_extra_info.split('/')[-2]+'_'+mask_extra_info.split('/')[-1]
#                     seq += f'_mask_{mask_extra_info}'
#                 task_name += '$$'
#                 append = os.path.join(args.output,seq,task_name,'mv_{}.'+ args.savetype)
#                 #from start-1 to end +1 because the videoflow can't output first and last img mv
#                 task_indexes = np.arange(start,end-1)
#                 args.num_frames = min(end-start+2,args.num_frames) ## for valid num_frames
#                 if args.multi_output and step==1:
#                     slide_indexes = sliding_window(task_indexes,args.num_frames,args.num_frames-2,1,step)
#                 else:
#                     slide_indexes = sliding_window(task_indexes,3,1,args.num_frames//2,step)
#                 total_range = range(len(slide_indexes))
#                 if args.use_tqdm:
#                     total_range = tqdm(total_range,desc='{}_{}'.format(task_name.replace('_$$','').replace('_##',''),seq))
#                 for i in total_range:
#                     slide_index = slide_indexes[i]
#                     mask = masks[seq_][0] if masks is not None else None
#                     imgs,valid,names = get_pairs(slide_index,seq_images,append,mask,args,step)
#                     imgs_op = None
#                     if seq_images_right is not None:
#                         right_mask = right_masks[seq_][0] if right_masks is not None  else None
#                         imgs_op,valid_op,names_op = get_pairs(slide_index,seq_images_right,append,right_mask,args,step)
#                         if cam == 'right':
#                             imgs_op,valid_op,names_op,imgs,valid,names = imgs,valid,names,imgs_op,valid_op,names_op
#                             if 'disparity' in args.weight_file:
#                                 continue
#                     if len(imgs)==0: #pass when exists
#                         if imgs_op is not None and len(imgs_op)>0:
#                             if len(imgs_op[0])==0:
#                                 continue
#                         else:
#                             continue
#                     initial_opt = optical_flow_pre_processing(imgs,valid,model,names,imgs_op,args,initial_opt,dump=args.enable_dump,disparity_inverse = cam=='right')
#     return save_res
import traceback
from tqdm import tqdm

def optical_flow(args, images, masks,
                 right_images=None, right_masks=None,
                 mask_extra_info=None):

    errors = []  # 汇总错误

    algorithm_check(args.algorithm, args.all_algorithm)
    if '-v' not in args.algorithm:
        algorithm_pick(images, args)

    save_res = []
    model = None if cpu_algorithm(args.algorithm) else get_model(args)
    cams = ['left', 'right'] if args.threeD_mode else ['monocular']
    imgs_op = None

    for seq_, data in images.items():
        seq = seq_
        seq_images, [start, end] = data
        seq_images_right = None if right_images is None else right_images[seq][0]

        try:
            input_valid_check(seq_images, args)

            # film border
            if 'auto' in args.film_border.lower():
                from conversion_tools.filmborder_detection import cal_border_length
                args.film_border_arr = cal_border_length(seq_images[0])
            else:
                args.film_border_arr = [int(num) for num in args.film_border.split(',')]

        except Exception as e:
            errors.append({
                "stage": "sequence_init",
                "seq": seq,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            continue  # 整个序列直接跳过

        for step in args.frame_step:
            for cam in cams:
                initial_opt = None
                task_name = (
                    f'{args.algorithm_fullname}_{cam}_'
                    if masks is None
                    else f'{args.algorithm_fullname}_{cam}_object_'
                )

                if mask_extra_info is not None and cam != 'right':
                    info = mask_extra_info
                    if '/' in info:
                        info = info.split('/')[-2] + '_' + info.split('/')[-1]
                    seq = f'{seq}_mask_{info}'

                task_name += '$$'
                append = os.path.join(args.output, seq, task_name, 'mv_{}.' + args.savetype)

                task_indexes = np.arange(start, end - 1)
                args.num_frames = min(end - start + 2, args.num_frames)

                if args.multi_output and step == 1:
                    slide_indexes = sliding_window(
                        task_indexes, args.num_frames, args.num_frames - 2, 1, step
                    )
                else:
                    slide_indexes = sliding_window(
                        task_indexes, 3, 1, args.num_frames // 2, step
                    )

                total_range = range(len(slide_indexes))
                if args.use_tqdm:
                    total_range = tqdm(
                        total_range,
                        desc=f'{task_name.replace("_$$", "").replace("_##", "")}_{seq}'
                    )

                for i in total_range:
                    slide_index = slide_indexes[i]

                    try:
                        mask = masks[seq_][0] if masks is not None else None
                        imgs, valid, names = get_pairs(
                            slide_index, seq_images, append, mask, args, step
                        )

                        imgs_op = None
                        if seq_images_right is not None:
                            right_mask = (
                                right_masks[seq_][0] if right_masks is not None else None
                            )
                            imgs_op, valid_op, names_op = get_pairs(
                                slide_index, seq_images_right, append,
                                right_mask, args, step
                            )

                            if cam == 'right':
                                imgs_op, valid_op, names_op, imgs, valid, names = (
                                    imgs, valid, names,
                                    imgs_op, valid_op, names_op
                                )
                                if 'disparity' in args.weight_file:
                                    continue

                        if len(imgs) == 0:
                            if imgs_op is not None and len(imgs_op) > 0:
                                if len(imgs_op[0]) == 0:
                                    continue
                            else:
                                continue

                        initial_opt = optical_flow_pre_processing(
                            imgs, valid, model, names,
                            imgs_op, args, initial_opt,
                            dump=args.enable_dump,
                            disparity_inverse=(cam == 'right')
                        )

                    except Exception as e:
                        errors.append({
                            "stage": "frame",
                            "seq": seq,
                            "cam": cam,
                            "step": step,
                            "index": i,
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        })
                        if args.use_tqdm:
                            total_range.write(
                                f"[ERROR] skip frame | seq={seq}, cam={cam}, step={step}, i={i}"
                            )
                        continue

    # ===== 统一打印错误 =====
    if errors:
        print("\n========== OPTICAL FLOW ERROR SUMMARY ==========")
        for idx, err in enumerate(errors):
            print(f"\n[{idx}] Stage: {err.get('stage')}")
            print(f"Seq: {err.get('seq')}")
            print(f"Cam: {err.get('cam', '-')}, Step: {err.get('step', '-')}, Index: {err.get('index', '-')}")
            print(err["error"])
            print(err["traceback"])
        print(f"\nTotal failed items: {len(errors)}")

    return save_res



'''
description: 深度核心计算
param {*} image 当前帧
param {*} algo 算法
return {*} 计算结果
'''
def depth_algo(img,args,model):
    DEVICE = args.DEVICE
    image = torch.from_numpy(mask_read(img,None,args)/255).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    depth = model.infer(image).squeeze().detach().cpu().numpy()
    depth -= np.min(depth)
    depth /= np.max(depth)
    # depth = (baseline * focal length) / disparity)
    baseline = 20
    focal_length = 1
    disparity = (baseline * focal_length) / depth
    # depth= np.repeat(depth[...,None],4,2)
    return disparity





'''
description: 通过mv场对结果进行恢复
param {*} image_dic
param {*} mv_dic
param {*} args
param {*} append
param {*} disparity
return {*}
'''
def recover_image(image_dic,mv_dic,args,append='recover',disparity=True,left=True):
     for seq in image_dic.keys():
        tmp = []
        images = image_dic[seq]
        mvs = mv_dic[seq]
        mkdir(os.path.join(args.output,seq,append))
        total_range = range(len(images)) if not args.use_tqdm else tqdm(range(len(images)), desc='{}:{}'.format(append,seq))
        for i in total_range:
            save_path = args.output+'/'+seq+'/'+append+'/image_'+re.findall(r'\d+', (images[i].split('/')[-1]))[-1]+ '.png'
            # if args.pass_when_exist and os.path.isfile(save_path):
            #     continue
            image = read(images[i],type='image')
            mv = read(mvs[i])
            mv[...,0] = mv[...,3]
            mv[...,1] = mv[...,2]
            mv = mv[...,:2]*args.depth_range-args.depth_range//2
            if disparity:
                mv[...,1]=0
            if left:
                mv*=-1
            result = immc(image,mv[...,:2])
            write(save_path,result)

'''
description: 通过生成的mv场对depth进行校正并生成disparity
param {*} image
param {*} mvd
param {*} left
return {*}
'''
def trans_depth_to_disparity(args,image_dic,target_image_dic,mv_dic,append,left=True,search_method_range=0):
    for seq in image_dic.keys():
        threshold = 0.03
        images = image_dic[seq]
        target_images = target_image_dic[seq]
        mvs = mv_dic[seq]
        img_name = args.image_file
        depth_root = images[0][:-images[0][::-1].find('/')].replace(img_name,args.depth_name)
        depths=[os.path.join(depth_root,i) for i in sorted(list(filter(lambda x:x[0]!='.' and 'exr' in x,os.listdir(depth_root))))]
        assert os.path.isfile(depths[0]),'depth file not exist!'
        mkdir(os.path.join(args.output,seq,append))
        total_range = range(len(mvs)) if not args.use_tqdm else tqdm(range(len(mvs)), desc='{}:{}'.format(append,seq))
        x_arr = np.array([])
        y_arr = np.array([])
        for i in total_range:
            x_arr = np.array([])
            y_arr = np.array([])
            #先载入depth和gt（光流深度结果） 做拟合（最小二乘法）disparity = Bf*depth+c
            image = read(images[i],type='image')
            depth = read(depths[i])[...,0]
            gt = read(mvs[i])[...,3]*args.depth_range-args.depth_range//2 #解除归一化
            # gt[np.where(depth==0)]=0#对depth数据中为0对部分进行过滤
            #最小二乘法拟合Bf和c
            image_ = image.copy()[...,0]
            mask = np.where((image_>2)&(depth>threshold))
            x_arr = np.append(x_arr,np.array(depth)[mask])
            y_arr = np.append(y_arr,np.array(gt)[mask])
            Bf,c = n_poly(2, x_arr, y_arr)  # 一阶
            # BBf,Bf,c = n_poly(2, x_arr, y_arr)  # 2阶
            
            # if search_method_range>0:
            #     print('调整前的参数',Bf,c)
            #     mask_ = np.where((image_<=2)|(depth<=threshold))
            #     Bf,c = search_method(images[i],target_images[i],depths[i],Bf,c,10,left,mask_)
            #     print('调整后的参数',Bf,c)
            #生成disparity
            depth = read(depths[i])[...,:2]
            disparity = Bf*depth+c
            # disparity = depth**2*BBf + Bf*depth + c
            #恢复图像并保存
            #y轴设为0
            if left:
                disparity[...,0]*=-1
            disparity[...,1] = 0
            result = immc(image,disparity)
            save_disparity=os.path.join(args.output,seq,append,'disparity_'+re.findall(r'\d+', (images[i].split('/')[-1]))[-1]+ '.exr')
            save_cover_image=os.path.join(args.output,seq,append,'recovered_image_'+re.findall(r'\d+', (images[i].split('/')[-1]))[-1]+ '.'+images[i].split('/')[-1].split('.')[-1])
            disparity = (disparity+args.depth_range//2) /args.depth_range #归一化
            disparity = np.repeat(disparity[...,0][...,None],4,2)
            write(save_cover_image,result)
            imageio.imwrite(save_disparity,disparity)
            
def search_method(image_,target_image_,depth_,Bf_,c_,search_method_step,left,mask):
    #先搜索大步长获取最优区间
    iter_num = 3
    Bf_list = [Bf_ + i*search_method_step for i in range(-5,6)]
    c_list  = [c_ + i*search_method_step for i in range(-5,6)]
    image = read(image_,type='image')
    target_image = read(target_image_,type='image')
    depth = read(depth_)[...,:2]
    print(Bf_list)
    for iter in range(iter_num):
        min_epe = [None,None,1e9]
        for Bf in Bf_list:
            for c in c_list:
                disparity = Bf*depth+c
                # disparity = depth**2*BBf + Bf*depth + c
                #恢复图像并保存
                #y轴设为0
                if left:
                    disparity[...,0]*=-1
                disparity[...,1] = 0
                restored_image = immc(image,disparity)
                target_image[mask] =0
                restored_image[mask] =0
                # epe = cal_epe(result,target_image)
                epe = cal_psnr(restored_image,target_image)
                if  epe < min_epe[2]:
                    min_epe = [Bf,c,epe]
        print('第{}次调整'.format(iter+1),min_epe)
        Bf_list = [min_epe[0]+i*search_method_step/(5**(iter+1)) for i in range(-4,5)]
        c_list  = [min_epe[1]+i*search_method_step/(5**(iter+1)) for i in range(-4,5)]
    print('最佳信噪比',-min_epe[2])
    return min_epe[:2]



'''
description: 根据mask位置生成bounding box信息
param {*} mask 
param {*} threshold mask的阈值
return {*} bouding box的xy信息
'''
def generate_bounding_box(mask):
    if len(mask.shape) == 3:
        mask = mask[...,0]
    verge = np.where(mask!=0)
    ly,lx = verge[0].min(),verge[1].min()
    ry,rx = verge[0].max(),verge[1].max()
    return lx,ly,rx,ry
    # bounding_box = mask[ly:ry,lx:rx]

def go_padding(mask,max_x,max_y):
    h,w = mask.shape
    lx,ly,rx,ry = generate_bounding_box(mask)
    diff_x = max_x - (rx-lx)
    diff_y = max_y - (ry-ly)
    pad_x_l,pad_x_r,pad_y_l,pad_y_r = 0,0,0,0
    if diff_x >0 :
        pad_x = diff_x//2
        pad_x_l = pad_x
        pad_x_r = diff_x - pad_x
    if diff_y >0 :
        pad_y = diff_y//2
        pad_y_l = pad_y
        pad_y_r = diff_y - pad_y
    lx_,ly_,rx_,ry_ = lx-pad_x_l,ly-pad_y_l,rx+pad_x_r,ry+pad_y_r
    if lx_ < 0 :
        lx_ = 0
        rx_ = max_x
    if ly_ <0 :
        ly_ = 0 
        ry_ = max_y
    if rx_ >= w:
        rx_ = w - 1
        lx_ = (w - 1 - max_x)
    if ry_ >= h:
        ry_ = h - 1
        ly_ = (h - 1 - max_y)
    return [lx_,ly_,rx_,ry_]
'''
description: 对bouding box中变长较小的进行padding
param {*} p1 mask
param {*} cf mask
return {*} padding后信息
'''
def pad_bounding_box(masks,args):
    n_frames =  masks.shape[0]
    for i in range(n_frames):
        if masks[i][...,-1].sum() <= masks[i][...,-1].max() *40 * 40 :
            return None,None
    lx,ly,rx,ry = 1e9,1e9,-1e9,-1e9
    max_x,max_y = 0,0
    if args.bounding_box_mode==2:
        for i in range(n_frames):
            lx_,ly_,rx_,ry_ = generate_bounding_box(masks[i][...,-1])
            lx = min(lx,lx_)
            ly = min(ly,ly_)
            rx = max(rx,rx_)
            ry = max(ry,ry_)
        return [[lx,ly,rx,ry]]*n_frames
    else:
        for i in range(n_frames):
            lx_,ly_,rx_,ry_ = generate_bounding_box(masks[i][...,-1])
            max_x = max(max_x,rx_-lx_)
            max_y = max(max_y,ry_-ly_)
        res = []
        for i in range(n_frames):
            res.append(go_padding(masks[i][...,-1],max_x,max_y))
        return res
            # p1_dx = p1_rx - p1_lx
            # cf_dx = cf_rx - cf_lx
            # p1_dy = p1_ry - p1_ly
            # cf_dy = cf_ry - cf_ly
            # if p1_dx != cf_dx:
            #     diff_x = abs(p1_dx-cf_dx) #差值
            #     #扩展较小边的边长
            #     if p1_dx < cf_dx:
            #         p1_lx -= diff_x
            #         if p1_lx <0: #边缘检测
            #             p1_lx = 0
            #             p1_rx = cf_dx
            #     else:
            #         cf_lx -= diff_x
            #         if cf_lx <0: #边缘检测
            #             cf_lx = 0
            #             cf_rx = p1_dx
            
            # if p1_dy != cf_dy:
            #     diff_y = abs(p1_dy-cf_dy) #差值
            #     #扩展较小边的边长
            #     if p1_dy < cf_dy:
            #         p1_ly -= diff_y
            #         if p1_ly <0: #边缘检测
            #             p1_ly = 0
            #             p1_ry= cf_dy
            #     else:
            #         cf_ly -= diff_y
            #         if cf_ly <0: #边缘检测
            #             cf_ly = 0
            #             cf_ry = p1_dy
        return [p1_lx,p1_ly,p1_rx,p1_ry],[cf_lx,cf_ly,cf_rx,cf_ry]






if __name__ == '__main__':
    from cfg_process import init_param
    args = init_param(os.path.join(os.getcwd(),'..','config'))
    args.root = os.path.join(os.path.join(os.getcwd(),'..'),args.root)
    args.model = os.path.join(os.path.join(os.getcwd(),'..'),args.model)
    args.output = os.path.join(os.path.join(os.getcwd(),'..'),args.output)
    args.use_tqdm = True
    imgs = torch.zeros((5,540,1024,4))
