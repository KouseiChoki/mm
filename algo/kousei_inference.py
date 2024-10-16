'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-10-16 09:52:07
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
import torch
import cv2
import torch.nn.functional as F
import numpy as np
import os
import sys
'''
description: 光流核心计算
param {*} pre 前一帧
param {*} cur 当前帧
param {*} algo 算法
return {*} 计算结果
'''


def forward_interpolate(flow):
    from scipy import interpolate
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()

class InputPadderMul:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel',coarsest_scale=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // coarsest_scale) + 1) * coarsest_scale- self.ht) % coarsest_scale
        pad_wd = (((self.wd // coarsest_scale) + 1) * coarsest_scale - self.wd) % coarsest_scale
        self.mode = mode
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2, 0, 0]
        elif mode == "downzero":
            self._pad = [0, pad_wd, 0, pad_ht, 0, 0]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht, 0, 0]

    def pad(self, input):
        if self.mode == "downzero":
            return F.pad(input, self._pad)
        else:
            return F.pad(input, self._pad, mode='replicate')

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

@torch.no_grad()
def optical_flow_algo(imgs,args,model=None,flow_prev=None,rgb = True):
    flow_lr = None
    if rgb:
        imgs = [img[...,:3] for img in imgs]
    if args.algorithm in ['farneback','deepflow','disflow','deepflow_cuda']: #to gray
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    if args.multi_frame_algo:
        if 'kousei' in args.algorithm:
            scale = 16 if 'v0' in args.algorithm else 8
            if args.empty_cache:
                torch.cuda.empty_cache()
            imgs = torch.stack([torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]).to(args.DEVICE)
            input_images = imgs[None].to(args.DEVICE)
            padder = InputPadderMul(input_images.shape,coarsest_scale=scale)
            input_images = padder.pad(input_images)
            if flow_prev is not None:
                flow_prev = torch.FloatTensor(flow_prev).to(args.DEVICE)
            try:
                flow,flow_lr = model(input_images,{},flow_init=flow_prev)
            except:
                sys.exit('[MM ERROR][memory]Insufficient GPU or CPU memory, please confirm the specifications and reselect the model')
            if args.usewarm:        
                flow_lr = forward_interpolate(flow_lr[0])[None].cpu().numpy()
            else:
                flow_lr = None
            flow = flow.squeeze(0).cpu().detach()
            flow = padder.unpad(flow)
        else:
            raise NotImplementedError
    else:
        # raise NotImplementedError #老代码弃用
        pre,cur = imgs[1],imgs[0] #mv1
        opt_mv1,opt_lr_mv1 = optical_flow_algo_one_step(pre,cur,args,model,flow_prev=None)
        pre,cur = imgs[1],imgs[2] #mv0
        opt_mv0,opt_lr_mv0 = optical_flow_algo_one_step(pre,cur,args,model,flow_prev=None)
        flow = torch.concat((torch.FloatTensor(opt_mv0)[None,...],torch.FloatTensor(opt_mv1[None,...])),axis=0)
        if args.usewarm:        
            opt_lr_mv0 = forward_interpolate(opt_lr_mv0[0])[None].cpu()
            opt_lr_mv1 = forward_interpolate(opt_lr_mv1[0])[None].cpu()
            flow_lr = np.concatenate((opt_lr_mv0[None,...],opt_lr_mv1[None,...]),axis=0)
        else:
            flow_lr = None
    return flow,flow_lr


@torch.no_grad()
def optical_flow_algo_one_step(pre,cur,args,model=None,flow_prev=None):
    if 'LOCAL_RANK' in os.environ.keys():
        torch.manual_seed(int(os.environ['LOCAL_RANK']))
        np.random.seed(int(os.environ['LOCAL_RANK']))
    algo = args.algorithm
    DEVICE = args.DEVICE
    flow_lr = None
    if algo =='farneback':
        flow = cv2.calcOpticalFlowFarneback(prev=pre, next=cur, flow=None, pyr_scale=0.5, levels=5,
                                            winsize=15,
                                            iterations=3, poly_n=3, poly_sigma=1.2,
                                            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flow = np.transpose(flow,(2,0,1))
    elif algo == 'deepflow':
        inst = cv2.optflow.createOptFlow_DeepFlow()
        # inst.downscaleFactor = 0.9
        flow = inst.calc(pre, cur, None)
        flow = np.transpose(flow,(2,0,1))
    elif algo == 'deepflow_cuda':
        # inst = cv2.cuda_BroxOpticalFlow.create(0.2, 50.0, 0.8, 5, 150, 10) 
        # inst = cv2.cuda_BroxOpticalFlow.create(0.2, 50.0, 0.8, 5, 50, 10) 
        # inst = cv2.cuda_BroxOpticalFlow.create(0.2, 50.0, 0.8, 5, 150, 5) 
        inst = cv2.cuda_BroxOpticalFlow.create(0.2, 50.0, 0.6, 5, 50, 5) 
        # inst = cv2.cuda_BroxOpticalFlow.create(0.2, 50.0, 0.6, 3, 10, 3) 
        # inst = cv2.cuda_BroxOpticalFlow.create(0.2, 50.0, 0.6, 2, 10, 2)
        # inst = cv2.cuda_BroxOpticalFlow.create(0.2, 50.0, 0.4, 1, 10, 1)
        #(alpha ,gamma ,scale ,inner_iterations ,outer_iterations ,solver_iterations)
        gpu_prev = cv2.cuda_GpuMat()
        gpu_cur = cv2.cuda_GpuMat()
        gpu_prev.upload(np.float32(pre)/255.0)
        gpu_cur.upload(np.float32(cur)/255.0)
        flow = inst.calc(gpu_prev, gpu_cur, None)
        flow = flow.download()
        flow = np.transpose(flow,(2,0,1))
    elif algo == 'simpleflow':
        flow = cv2.optflow.calcOpticalFlowSF(pre, cur, 2, 2, 4)
        flow = np.transpose(flow,(2,0,1))
    elif algo == 'stdf':
        flow = cv2.optflow.calcOpticalFlowSparseToDense(pre, cur)
        flow = np.transpose(flow,(2,0,1))
    elif algo == 'pcaflow':
        inst = cv2.optflow.createOptFlow_PCAFlow()
        flow = inst.calc(pre, cur, None)
        flow = np.transpose(flow,(2,0,1))
    elif algo == 'rlof':
        flow = cv2.optflow.calcOpticalFlowDenseRLOF(pre, cur,None)
        flow = np.transpose(flow,(2,0,1))
    #388 584  392 584
    elif 'gma' in algo:
        if args.empty_cache:
            torch.cuda.empty_cache()
        pre = torch.from_numpy(pre).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        cur = torch.from_numpy(cur).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        padder = InputPadder(pre.shape)
        pre, cur = padder.pad(pre, cur)
        with torch.no_grad():
            flow_low, flow_up = model(pre, cur, iters=12, test_mode=True)
        # flow = gma_viz(flow_low, flow_up)
        flow = flow_up.squeeze(0).cpu().detach()
        flow = padder.unpad(flow)
        flow_lr = flow_low.squeeze(0).cpu().detach()
        flow_lr = padder.unpad(flow_lr)

    elif algo =='gma_4k':
        #patch number, if  partition ==2, then image will devide into 4 image
        partition = 2
        #cover size
        cover_size_h = 8 + (8 - (pre.shape[0]//2)%8)
        cover_size_w = 8 + (8 - (pre.shape[1]//2)%8)
        def regroup(data):
            h,w,_ = data.shape
            assert min(h,w)// partition > cover_size_w and min(h,w)// partition > cover_size_h, 'cover size can not larger than image size!'
            res = np.zeros((4,h//2+cover_size_h,w//2+cover_size_w,3))
            # for i in range(partition**2):
            #     res[i,...] = data[max(0,i*hh):(i+1)*hh]
            #先默认partition=2 写死，后续要扩展再拓展代码
            res[0] = data[:h//2+cover_size_h,:w//2+cover_size_w,:]
            res[1] = data[:h//2+cover_size_h,w//2-cover_size_w:,:]
            res[2] = data[h//2-cover_size_h:,:w//2+cover_size_w:,:]
            res[3] = data[h//2-cover_size_h:,w//2-cover_size_w:,:]
            # res = np.zeros((partition**2,hh,ww,3))
            # res[0] = data[:hh,:ww,:]
            # res[1] = data[hh:,ww:,:]
            # res[2] = data[:hh,:ww,:]
            # res[3] = data[hh:,ww:,:]
            res = torch.from_numpy(res).permute(0, 3, 1, 2).float()
            return res

        def restore(data):
            #4,2,594 1056
            b,c,hhh,www = data.shape
            h = (hhh - cover_size_h) * 2
            w = (www - cover_size_w) * 2
            res = np.zeros((2,h,w))
            dup = np.zeros((h,w))
            res[:,:h//2+cover_size_h,:w//2+cover_size_w] = data[0]
            res[:,:h//2+cover_size_h,w//2-cover_size_w:] = data[1]
            res[:,h//2-cover_size_h:,:w//2+cover_size_w:] = data[2]
            res[:,h//2-cover_size_h:,w//2-cover_size_w:] = data[3]
            ##加权 求平均
            dup[:h//2+cover_size_h,:w//2+cover_size_w]+= 1
            dup[:h//2+cover_size_h,w//2-cover_size_w:] += 1
            dup[h//2-cover_size_h:,:w//2+cover_size_w:]  += 1
            dup[h//2-cover_size_h:,w//2-cover_size_w:] += 1
            res[0,...] = res[0,...] / dup
            res[1,...] = res[1,...] / dup
            return  np.transpose(res,(1,2,0))

        pre = regroup(pre).to(DEVICE)
        cur = regroup(cur).to(DEVICE)
        flow = []
        with torch.no_grad():
            for i in range(4):
                flow_low, flow_up = model(pre[i].unsqueeze(0), cur[i].unsqueeze(0), iters=12, test_mode=True)
                flow_ = flow_up.squeeze(0).cpu().detach().numpy().astype('float32')
                flow.append(flow_)
        # flow = gma_viz(flow_low, flow_up)
        flow = np.stack(flow)
        flow = restore(flow)
    elif 'deq' in algo:
        if args.empty_cache:
            torch.cuda.empty_cache()
        pre = torch.from_numpy(pre).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        cur = torch.from_numpy(cur).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        padder = InputPadder(pre.shape)
        pre, cur = padder.pad(pre, cur)
        with torch.no_grad():
            _,flow,info = model(pre, cur)
        flow = padder.unpad(flow[0]).cpu().detach()
        flow = padder.unpad(flow),(1,2,0)
    elif 'msraft' in algo:
        from utils.utils import forward_interpolate
        if args.empty_cache:
            torch.cuda.empty_cache()
        pre = torch.from_numpy(pre).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        cur = torch.from_numpy(cur).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        padder = InputPadder(pre.shape,coarsest_scale=16)
        pre, cur = padder.pad(pre, cur)
        init_flow_for_next_low,flow = model(pre, cur,iters=args.iters,flow_init=flow_prev, test_mode=True)
        flow_lr = forward_interpolate(init_flow_for_next_low[0])[None].to(DEVICE) if args.usewarm else None
        flow = flow[0].squeeze(0).cpu().detach()
        flow = padder.unpad(flow)
    elif 'flowformer' in algo:
        if args.empty_cache:
            torch.cuda.empty_cache()
        pre = torch.from_numpy(pre).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        cur = torch.from_numpy(cur).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        padder = InputPadder(pre.shape)
        pre, cur = padder.pad(pre, cur)
        if flow_prev is not None:
            flow_prev = torch.FloatTensor(flow_prev).to(DEVICE)
        flow,flow_lr = model(pre, cur,flow_init=flow_prev)
        if args.usewarm:        
            from utils.utils import forward_interpolate
            flow_lr = forward_interpolate(flow_lr[0])[None].cpu().numpy()
        else:
            flow_lr = None
        flow = flow.squeeze(0).cpu().detach()
        flow = padder.unpad(flow)
    else:
        raise NotImplementedError
    return flow,flow_lr

