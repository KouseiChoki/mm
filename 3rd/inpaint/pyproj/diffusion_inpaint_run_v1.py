import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import random
import math
from copy import deepcopy
from torch.autograd import Variable


import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from diffusion_gen_v5 import SimpleDiffusion, get
from diffusion_gen_v5 import Model

# this is a simplified version of the code that is used for training and evaluation.
# it can read multiple mask files so that you can experiment with artifact reduction
# and harmonization of the image that has been generated from multiple frames
# for production, you would want to port some of this code over to the diffusion_inpaint_v8b.py
# and diffusion_data_v2.py so that you can run it on multiple GPU's.
# show_image => inpaint_v8b
# get_data => data_v2.py

convert_tensor = transforms.ToTensor()

def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    #plt.title(title)
    plt.show()

def mask_rgb(rgbImage, mask):
    # create 3 channel of the mask
    mask3chnl = torch.cat([mask,mask,mask], dim=0)
    # mask is already 3 channel???
    #return rgbImage*mask3chnl+0.5*(1-mask3chnl)
    # let's put hole in as black
    return rgbImage*mask3chnl

def saveImg(image,path,folder,name,ext,bits,transform=None):
    img_dir = os.path.join(path,folder)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print(f"The new directory {path} is created!")        
    img_path = os.path.join(img_dir,name + ext)
    if transform is not None:
        image = transform(image)
    elif not (ext == '.exr'):
        gain = 2**bits - 1
        image = torch.clamp(image*gain+0.5,0,gain)
        if bits > 8:
            image = image.to('cpu')
            #pil_image = Image.fromarray(np.uint16(image.numpy()))
            #pil_image.save(img_path, format=ext)
            image = torch.permute(image,(1,2,0)).numpy()
            imgInt = image.astype('uint16')
            cv2.imwrite(img_path,imgInt)
        else:
            image = image.to('cpu')
            image = torch.permute(image,(1,2,0)).numpy()
            imgInt = image.astype('uint8')
            cv2.imwrite(img_path,imgInt)
    else:
        image = image.to('cpu')
        #pil_image = Image.fromarray(np.float32(image.numpy()))
        #pil_image.save(img_path, format=ext)
        image = torch.permute(image,(1,2,0)).numpy()
        imgInt = image.astype('float32')
        cv2.imwrite(img_path,imgInt)
    print(img_path)

def bgr2rgb_batch(bgrImg):
    #yuv is the cv2 ycbcr with the cbcr offset removed
    output = Variable(bgrImg.data.new(*bgrImg.size()))
    # need to fix below with the correct equations
    # red channel
    output[:, 0, :, :] = bgrImg[:, 2, :, :]
    output[:, 1, :, :] = bgrImg[:, 1, :, :]
    output[:, 2, :, :] = bgrImg[:, 0, :, :]
    # similarly write output[:, 1, :, :] and output[:, 2, :, :] using formulas from https://en.wikipedia.org/wiki/YCbCr
    return output

def scaleImg(img):
    if img.shape[1] > 3:
        tmp = 2*img[:,0:3,:,:]-1
        img = torch.concat([tmp,img[:,3:,:,:]],dim=1)
    else:
        img = 2*img - 1
    return img

def invScaleImg(img):
    if img.shape[1] > 3:
        tmp = (img[:,0:3,:,:]+1.)*0.5
        img = torch.concat([tmp,img[:,3:,:,:]],dim=1)
    else:
        img = (img+1.)*0.5
    return img

def inverse_transform(tensors,maxVal=255.0):
    """Convert tensors from [-1., 1.] to [0., 255.]"""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * maxVal

def get_data(args):
    imgPath = os.path.join(args.dataset_root,args.img_folder)
    imgName = args.img_name
    imgExt = args.img_ext
    maskName = args.mask_name
    maskExt = args.mask_ext
    maskSuffix = args.mask_suffix
    imgPath = os.path.join(imgPath,(imgName+imgExt))
    tempPic = cv2.imread(imgPath,6)
    tempPic = tempPic.astype('float32')
    if imgExt != '.exr':
        if args.imgIs16:
            tempPic = tempPic/65535
        else:
            tempPic = tempPic/255
    if len(tempPic.shape) == 3:
        #tempPic = cv2.cvtColor(tempPic, cv2.COLOR_BGR2YCrCb)
        image = convert_tensor(tempPic)
        #image = ycrcb2yuv(image,self.cGain) # so that 0 color has CbCr = 0
    else:
        temp = convert_tensor(tempPic)
        image = torch.concat([temp,temp,temp],0)
    maskPath = os.path.join(args.dataset_root,args.mask_folder)
    for i in range(len(maskSuffix)):
        cMaskPath = os.path.join(maskPath,(maskName+maskSuffix[i]+maskExt))
        tempMask = cv2.imread(cMaskPath,6)
        tempMask = tempMask.astype('float32')
        if maskExt != '.exr':
            if args.maskIs16:
                tempMask = tempMask/65535
            else:
                tempMask = tempMask/255
        if len(tempMask.shape) == 3:
            tempMask = cv2.cvtColor(tempMask, cv2.COLOR_BGR2GRAY)
        mask = convert_tensor(tempMask)
        if not args.maskIsPV: # then invert mask so that any pad is considered the mask
            validPix = 1-mask
        else:
            validPix = mask
        validPix = validPix.clamp(0,1)
        if i == 0:
            preValidPix = validPix
        elif i == 1:
            imgVpix = torch.maximum(preValidPix,validPix)
            preValidPix = F.max_pool2d(preValidPix,3,stride=1,padding=1)
            validPix = F.max_pool2d(validPix,3,stride=1,padding=1)
            imgBpix = torch.minimum(preValidPix,validPix) # intersection of the two expanded areas could have issues
            preValidPix = validPix
        else:
            imgVpix = torch.maximum(imgVpix , validPix) # essentially, image valid pixels are the union of all the valid pix areas
            validPix = F.max_pool2d(validPix,3,stride=1,padding=1)
            tempImgBpix = torch.minimum(preValidPix,validPix) # intersection of the two expanded areas could have issues
            imgBpix = torch.maximum(tempImgBpix,imgBpix)          
            preValidPix = validPix
    if args.chromaKey is not None: # has to be done before any transforms and result is a pixel valid
        tempMask = torch.zeros_like(image)
        for i in range(3):
            #tempMask[i,:,:] = image[i,:,:].masked_fill(image[i,:,:] == self.chromaKey[i], 2) # so mask value is 2x max image value
            tempMask[i,:,:] = tempMask[i,:,:].masked_fill(image[i,:,:] == args.chromaKey[2-i], 2) # so mask value is 2x max image value
        mask = tempMask[0:1,:,:] + tempMask[1:2,:,:] + tempMask[2:3,:,:]
        mask = F.threshold(mask,5.5,0)/6 # two channel match max value = 2+2+1 = 5, so 5.5 threshold
        imgVpix = 1-mask #expand mask and change to
        #mask = 1-F.max_pool2d(mask,3,stride=1,padding=1) #expand mask and change to
        # mask is actually pixel valid so that during transform any padding = 0 is an area to inpaint

    # need to crop the data to imgSize, which is really just padding it.
    crop = transforms.Compose([
        transforms.CenterCrop(args.imgSize),
        ])

    image = crop(image)
    imgVpix = crop(imgVpix)
    imgBpix = crop(imgBpix)

    return image, imgVpix, imgBpix

def load_train_objs(args,device):

    
    model = Model(base_channels=32,
                channel_mult=[16,16,16,8,4,2,1],
                attn_layers=[True,True,True,False,False,False],
                num_atten=[4,2,1,1,1,1],
                #patchSizes=[[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]],
                patchSizes=[[1,1],[2,2],[4,4],[1,1],[1,1],[1,1]],
                #patchSizes=[[1,1],[1,1],[2,2],[1,1],[1,1],[1,1]],
                con_attn_num=8, # with 3x2 in the cond and 3x2 in the decode, that makes 20 total
                attn_channels=1024,
                num_heads=4,
                mlp_ratio=4.0,
                dropout_rate=args.dropout_p,
                time_emb_dims=512)
    
    checkpoint_pre = torch.load(args.chk_load_pre + '_ema_m_' + args.chk_lowRes_ver + '.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint_pre['ema_m_state_dict'], strict=False)
    
    ema_m = deepcopy(model)

    checkpoint_pre = torch.load(args.chk_load_pre + '_ema_m_' + args.chk_HiRes_ver + '.pth',map_location='cpu',weights_only=True)
    ema_m.load_state_dict(checkpoint_pre['ema_m_state_dict'], strict=False)

    ema_m.eval().to(device)
    model.eval().to(device)

    return model, ema_m


def p_sample(sd,x,ats,z,ts,model):
    
    predicted_noise = model(x, ats)

    beta_t                            = get(sd.beta, ts)
    one_by_sqrt_alpha_t               = get(sd.one_by_sqrt_alpha, ts)
    sqrt_one_minus_alpha_cumulative_t = get(sd.sqrt_one_minus_alpha_cumulative, ts) 
    x = (
        one_by_sqrt_alpha_t
        * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
        + torch.sqrt(beta_t) * z
    )
    return x


@torch.no_grad()
def show_img(args,mLowRes,mHiRes,image, imgVpix, imgBpix, device):

    image = image.to(device)

    #img_apl = torch.mean(image,dim=[1,2,3],keepdim=True)
    #img_apl_exp = torch.log(img_apl)/math.log(0.3)
    #img_apl_exp = torch.clamp(img_apl_exp,1,3) # if image is too dark, then we apply gamma to lighten up the dark areas
    #in_img = torch.pow(image,1.0/img_apl_exp)
    #in_img = scaleImg(in_img)
    img = scaleImg(image)

    imgVpix = imgVpix.to(device)
    imgBpix = imgBpix.to(device)
    batch_size, C, H, W = image.shape
    imgSave = False
    # there are some instances of the mask not being quite right so expand by a few pixel
    #p0s = smpl_inpaint(img*mask,mask,self.args.sizeLayers) # the inference does this step before, so it needs to be part of training too
    if args.makeMaskBinary:
        mask = torch.floor(imgVpix) # make sure that it's binary and only valid pixels are marked as such
    hole = 1-mask
    for i in range(args.dilateHoleMult):
        hole = F.max_pool2d(hole,3,stride=1,padding=1)
    for i in range(args.dilate_art_reduce):
        imgBpix = F.max_pool2d(imgBpix,3,stride=1,padding=1)
    init_img = img*mask # this is my raw input, I need this because mask = 0 should be zero and it's -1 if img has a hole
                        # this shouldn't make a difference as it is always mulitplied by inpaint_mask in the code below
    start_mask = 1-hole
    imgGpix = torch.minimum(start_mask,1-imgBpix)

    #p0s = smpl_inpaint(inpaint_mask*img,inpaint_mask,self.args.sizeLayers) # the inference does this step before, so it needs to be part of training too
    #init_img = inpaint_mask*img + p0s*hole
    init_img = start_mask*img
    #mask_img2use = img*img_mask
    one_ts = torch.ones(batch_size, dtype=torch.long, device=device)
    #if args.dif_rng_seed == 0 or (not args.inPaint):
    if not args.inPaint:
        torch.manual_seed(random.randint(1, 10000))
    else:
        torch.manual_seed(args.dif_rng_seed)
        #z.repeat(batch_size,1,1,1)
    res_values = [[512,512],[H,W]]
    start_time_step = [args.start_sample_steps,256]
    multi_stage = True
    for res_step in range(2):
        sd = SimpleDiffusion(num_diffusion_timesteps=args.time_steps, device=device)
        if not multi_stage:
            init_img = img
            inpaint_mask = start_mask
            init_img = init_img*inpaint_mask
            ip_model = mHiRes
            z = torch.randn_like(init_img[0:1,:,:,:])
            #z = torch.roll(z,[yOff,xOff],dims=[-2,-1]) # shifting the noise to match the shift in the background
            z.repeat(batch_size,1,1,1)
            if not args.inPaint:
                x = z
            else:
                #x, _gtn = sd.forward_diffusion(init_img,one_ts*(start_time_step[res_step]-1),noise2use=z)
                x, _gtn = sd.forward_diffusion(init_img,one_ts*(start_time_step[res_step]-1),noise2use=z)            
        elif res_step == 0:
            init_img = F.interpolate(img,res_values[res_step],mode='bilinear',antialias=False)
            inpaint_mask = F.interpolate(start_mask,res_values[res_step],mode='bilinear',antialias=False)
            inpaint_mask = torch.floor(inpaint_mask)
            init_img = init_img*inpaint_mask
            ip_model = mLowRes
            z = torch.randn_like(init_img[0:1,:,:,:])
            #z = torch.roll(z,[yOff,xOff],dims=[-2,-1]) # shifting the noise to match the shift in the background
            z.repeat(batch_size,1,1,1)
            if not args.inPaint:
                x = z
            else:
                #x, _gtn = sd.forward_diffusion(init_img,one_ts*(start_time_step[res_step]-1),noise2use=z)
                x, _gtn = sd.forward_diffusion(init_img,one_ts*(start_time_step[res_step]-1),noise2use=z)
        else:
            z = torch.randn_like(img[0:1,:,:,:])
            #z = torch.roll(z,[yOff,xOff],dims=[-2,-1]) # shifting the noise to match the shift in the background
            z.repeat(batch_size,1,1,1)
            x = F.interpolate(x,res_values[res_step],mode='bilinear',antialias=False)
            #x, _gtn = sd.forward_diffusion(init_img,one_ts*(start_time_step[res_step]-1),noise2use=z)
            x, _gtn = sd.forward_diffusion(x,one_ts*(start_time_step[res_step]-1),noise2use=z)
            init_img = img
            inpaint_mask = start_mask
            ip_model = mHiRes

        time_step = start_time_step[res_step]
        show_index = 0
        if device.index == 0:
            pbar = tqdm(total=start_time_step[res_step]-2)
        sd_ver = 0
        repeat_count = 0
        variable_steps = True
        #sd = SimpleDiffusion(num_diffusion_timesteps=1024, device=device)
        step_size = 1
        steps_back = 20
        repeat = 0
        min_t = 0
        while time_step > 1: # needs to be > 1 so that indexing to get betas works below
            if time_step < args.start_art_reduce and res_step > 0:
                inpaint_mask = imgGpix
            if variable_steps:
                if time_step > 768:
                    if sd_ver == 0:
                        #step_size = 8
                        step_size = 8
                        sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=device)
                        sd_ver = 1
                        #steps_back = 16 # first is 1000->500
                        steps_back = 8
                        repeat = 3
                        min_t = 0
                elif time_step > 512:
                    if sd_ver <= 1:
                        #step_size = 4
                        step_size = 8
                        sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=device)
                        sd_ver = 2
                        #steps_back = 8 # first is 500 -> 250 - 20%
                        #steps_back = 12 # first is 500 -> 250 - 20%
                        steps_back = 16 # first is 500 -> 250 - 20%
                        repeat = 3
                        min_t = 0
                elif time_step > 256:
                    if sd_ver <= 2:
                        #step_size = 4
                        step_size = 4
                        sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=device)
                        sd_ver = 3
                        steps_back = 16 # first is 150/5 = 20 
                        repeat = 0
                        min_t = 0
                elif time_step > 128:
                    if sd_ver <= 3:
                        step_size = 4
                        sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=device)
                        sd_ver = 4
                        steps_back = 8 #100->50
                        repeat = 0
                        min_t = 0
                elif time_step > 64:
                    if sd_ver <= 4:
                        step_size = 4
                        sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=device)
                        sd_ver = 5
                        steps_back = 4
                        repeat = 0
                        min_t = 0
                elif time_step > 32:
                    if sd_ver <= 5:
                        step_size = 2
                        sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=device)
                        sd_ver = 5
                        steps_back = 2
                        repeat = 0
                        min_t = 0
                else:
                    if sd_ver <= 6:
                        step_size = 1
                        steps_back = 1
                        sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=device)
                        repeat = 0
                        sd_ver = 6

            adj_time_step = (time_step-min_t)//step_size
            #ts = one_ts * adj_time_step * step_size # the model restore has to be in the 0:999 range
            ts = one_ts * time_step # the model restore has to be in the 0:999 range
            
            num_steps = min(adj_time_step,steps_back)

            #if self.args.steps_back_range[0] >= time_step and self.args.steps_back_range[1] <= time_step:
            #    num_steps = min(adj_time_step,self.args.steps_back)
            #else:
            #    num_steps = min(adj_time_step,1)
            if not args.inPaint:
                z = torch.randn_like(x[0:1,:,:,:]) if adj_time_step > 1 else torch.zeros_like(x)
                z.repeat(batch_size,1,1,1)
                ts = one_ts * (adj_time * step_size)
                x = p_sample(sd,x,ts-1,z,one_ts*(adj_time_step-1),ip_model)
            elif num_steps >= 0:
                for i in range(num_steps):
                    z = torch.randn_like(x[0:1,:,:,:]) if adj_time_step > 1 else torch.zeros_like(x)
                    #z = torch.roll(z,[yOff,xOff],dims=[-2,-1]) # shifting the noise to match the shift in the background
                    z.repeat(batch_size,1,1,1)
                    adj_time = adj_time_step - i# 
                    #ts = one_ts * (adj_time * step_size + step_size -1)
                    ts = one_ts * (adj_time * step_size)
                    x = p_sample(sd,x,(ts-1),z,one_ts*(adj_time-1),ip_model)
                    #adj_time_step = adj_time*t_scale
                    if adj_time > 0:
                        # use same noise for all the steps
                        #kwnImg, gt_noise = self.sd.forward_diffusion(mask_img,one_ts*(adj_time-1))
                        #z = torch.randn_like(mask_img2use[0:1,:,:,:])
                        #z.repeat(batch_size,1,1,1)
                        #kwnImg, _gtn = sd.forward_diffusion(init_img,one_ts*(adj_time-1),noise2use=z) # this was the original, but I don't think I need to subtract 1
                        kwnImg, _gtn = sd.forward_diffusion(init_img,one_ts*(adj_time-1),noise2use=z) # this is correct!!! proved experimentally 10/30/2024
                    else:
                        kwnImg = init_img.detach().clone()
                    # then blend with what the original should be
                    x = kwnImg*inpaint_mask + x*(1-inpaint_mask) # this coiuld be zero, but there are more diffusion steps
                    #x = kwnImg
                # going forward in time one less than going backward
                if repeat_count == repeat:
                    num_steps -= 1
                    repeat_count = 0
                else:
                    repeat_count += 1
                num_steps = max(0,num_steps)
                for j in reversed(range(num_steps)):
                    #ts += t_scale # increment ts to next output position
                    #beta_t = get(self.sd.beta, one_ts*(adj_time-1))
                    beta_t = get(sd.beta, one_ts*(adj_time-1))
                    z = torch.randn_like(init_img[0:1,:,:,:])
                    #z = torch.roll(z,[yOff,xOff],dims=[-2,-1]) # shifting the noise to match the shift in the background
                    z.repeat(batch_size,1,1,1)
                    #x = x*torch.sqrt(1-beta_t) + beta_t*z
                    x = x*torch.sqrt(1-beta_t) + torch.sqrt(beta_t)*z # in the repaint code it's sqrt, in paper it's not.
                    #kwnImg, gt_noise = self.sd.forward_diffusion(mask_img,one_ts*adj_time)
                    #x = kwnImg*img_mask + x*nimg_mask
                    adj_time += 1
                    #x = kwnImg
                #x, gt_noise = self.sd.forward_diffusion(x,one_ts*max(0,(num_steps)))
                #x, gt_noise = self.sd.forward_diffusion(x,one_ts*max(0,(time_step-2)))
                #kwnImg, gt_noise = self.sd.forward_diffusion(x,one_ts*max(0,time_step-2))
                #x = kwnImg*img_mask + x*nimg_mask
            else:
                x = p_sample(sd,x,ts,z,one_ts*0,ip_model)
                #x = kwnImg*mask + x*(1-mask) # use the original mask because this is the last step
                x = kwnImg*inpaint_mask + x*(1-inpaint_mask) # this is safer and can always apply original mask later
            if time_step <= args.show_time_steps[show_index] and args.showImg:
                #write the images using the show_time_step in the file name
                #fake_data = self.vae.decode(x.detach().clone() / 0.18215).sample
                #fake_data = mImg[:,0:3,:,:]*img_mask + fake_data*(1-img_mask)
                fake_data = torch.clip(invScaleImg(x),0,1)
                # undo the gamma adjustment and then merge with the original
                #fake_data = torch.pow(fake_data,img_apl_exp)
                #fake_data = image*mask + fake_data*(1-mask)
                outputs = fake_data.to('cpu')
                outputs = bgr2rgb_batch(outputs)
                inputs = bgr2rgb_batch(invScaleImg(init_img)).to('cpu')
                #refImages = bgr2rgb_batch(invScaleImg(ref)).to('cpu')
                # reorganize the data to display: first row content, second,depth, third vpix
                #RGBvPix = 1-torch.cat([inputStack[:,3:4,:,:],inputStack[:,3:4,:,:],inputStack[:,3:4,:,:]],1)
                debugImg = torch.cat([inputs,outputs],0)
                img_out = torchvision.utils.make_grid(debugImg,nrow=batch_size,normalize=False)
                #hr_out = torchvision.utils.make_grid(img_inputs[hr_img])
                imshow(img_out)
                show_index += 1
            if repeat_count == 0:
                time_step -= step_size
                if device.index == 0:
                    pbar.update(step_size)
    if args.saveImg:
        x = torch.clip(x,-1.0,1.0)
        x = invScaleImg(x)
        #x = torch.pow(x,img_apl_exp)
        #x = invScaleImg(stack[:,0:3,:,:])*stack[:,3:4,:,:] + x*(1-stack[:,3:4,:,:])
        #x = invScaleImg(refImages)
        #mask_img = inverse_transform(mask_img,1)
        #img = inverse_transform(img,1)
        #for i in range(len(args.img_name)):
            #if args.img_subfolder is None:
        savePath = args.saveImg_path
            #else:
            #    savePath = imgDirs[i]
        if args.saveOutput_folder is not None:
            saveImg(x[i,:,:,:],savePath,args.saveOutput_folder,args.img_name,args.saveExt,args.saveImgBPC)
        #if last_batch:
        #    print(f'last batch {self.device}')
            #mytensor = torch.tensor([torch.distributed.get_rank()], dtype=torch.int32,device=self.device.type)
            ##tensors need to be dense and cuda for gather
            #if torch.distributed.get_rank() == 0:
            #    gpu_list =  [mytensor.clone() for _ in range(torch.distributed.get_world_size())]
            #    torch.distributed.gather(tensor=mytensor,gather_list=gpu_list)
            #    print(f'GPUs used:{gpu_list}')
            #else:
            #    torch.distributed.gather(tensor=mytensor)

    pbar.close()
    return None     

if __name__ == '__main__':
    if(torch.cuda.is_available()):
        device = torch.device('cuda',0)
    elif(torch.backends.mps.is_available()):
        device = torch.device('mps',0)
        #device = torch.device('cpu',0) # debug on CPU so I don't slow down training
    else:
        device = torch.device('cpu')

    
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--location', default='home', help='version of checkpt to save(default: OFSRCNN14)')
    parser.add_argument('--chk_load_pre', default='checkPoint/ddpm', help='version of checkpt to save(default: OFSRCNN14)')
    parser.add_argument('--chk_lowRes_ver', default='mdl512_v8a', help='version of checkpt to load')
    parser.add_argument('--chk_HiRes_ver', default='mdl512_v8a', help='version of checkpt to load')
    
    parser.add_argument('--time_steps', default=1024, type=int, help='number of diffusion time steps (default: 1024)')
    parser.add_argument('--dropout_p', default=.125, type=float, help='number of diffusion time steps (default: 1000)')
    #parser.add_argument('--time_emb_mult', default=4, type=int, help='number of diffusion time steps (default: 1000)')
    #parser.add_argument('--dif_rng_seed',default=0, type=int, help='rng seed for diffustion, 0=>rand seed (default: 0)')
    parser.add_argument('--dif_rng_seed',default=4713, type=int, help='rng seed for diffustion, 0=>rand seed (default: 0)')
    parser.add_argument('--img_folder', default="toNeil_1023/render_out_1023_JM3",help='root folder for datasets (default: /Volumes/neil_SSD/datasets/)')
    parser.add_argument('--img_name', default="Out_1122",help='root folder for datasets (default: /Volumes/neil_SSD/datasets/)')
    parser.add_argument('--img_ext', default=".exr",help='leading period to be compatible with extension found via parsing')
    parser.add_argument('--mask_folder', default="toNeil_1023/AIpainting_mask", help='folder for masks')
    parser.add_argument('--mask_name', default="1122", help='folder for masks')
    parser.add_argument('--mask_ext', default=".exr", help='leading period to be compatible with extension found via parsing')
    parser.add_argument('--mask_suffix', default=["_P1","_CF","_F1"],type=list, help='folder for masks')
    #parser.add_argument('--imgSize', default=(512,512), type=int, nargs=2, help='pixel mulitple image needs to be)')
    #parser.add_argument('--imgSize', default=(1024,1024), type=int, nargs=2, help='pixel mulitple image needs to be)')
    parser.add_argument('--imgSize', default=(1088,2048), type=int, nargs=2, help='pixel mulitple image needs to be)')
    parser.add_argument('--imgIs16', default=False, type=bool, help='load input CNN section (default=False))')
    parser.add_argument('--imgChnl', default=3, type=int, help='number of color channels to use (default=1)')
    #parser.add_argument('--chromaKey', default=None, help='set to None to disable')
    parser.add_argument('--chromaKey', default=[0,0,0], help='set to None to disable')
    parser.add_argument('--maskIsPV', default=True, type=bool, help='high value is pixel valid (default: False)')
    parser.add_argument('--maskIs16', default=False, type=bool, help='mask is 16 bit (default: False)')
    #parser.add_argument('--getRandMask', default=False, type=bool, help='get a random mask and not masks paired with image')
    parser.add_argument('--makeMaskBinary', default=True, type=bool, help='use mask information for training or inference')
    parser.add_argument('--dilateHoleMult',default=1,type=int,help='number of times to do 3x3 max operation on hole')

    parser.add_argument('--start_art_reduce',default=256,type=int,help='number of times to do 3x3 max operation on hole')
    parser.add_argument('--dilate_art_reduce',default=1,type=int,help='number of times to do 3x3 max operation on hole')
    parser.add_argument('--showImg', default=True, type=bool, help='display a batch of images (default=False)')
    parser.add_argument('--inPaint', default=True, type=bool, help='display a batch of images (default=False)')
    parser.add_argument('--steps_back', default = 8, type=int, help='number of steps forward in time when gen image')
    parser.add_argument('--steps_back_range', default =[1024,1], nargs=2, type=int, help='range for going back and forth in time')
    
    parser.add_argument('--start_sample_steps', default =1024,type=int, help='starting time step for generating image')
    #parser.add_argument('--num_sample_steps', default =1000, type=int, help='number of steps in generating an image')
    #parser.add_argument('--show_time_steps', default=[999,800,500,249,128,64,32,16,8,4,2,1,0], type=int, help='time steps to show progress')
    parser.add_argument('--num_sample_steps', default =256, type=int, help='number of steps in generating an image')
    #parser.add_argument('--show_time_steps', default=[1023,768,512,256,128,64,4,2,1,0], nargs='*', type=int, help='time steps to show progress, set to zero to skip')
    #parser.add_argument('--show_time_steps', default=[1023,768,512,256,128,64,4,2,1,0], nargs='*', type=int, help='time steps to show progress, set to zero to skip')
    #parser.add_argument('--show_time_steps', default=[1023,4,2,1,0], nargs='*', type=int, help='time steps to show progress, set to zero to skip')
    parser.add_argument('--show_time_steps', default=[0], nargs='*', type=int, help='time steps to show progress, set to zero to skip')
    parser.add_argument('--showImg_num', default=2, type=int, help='number of images to generate')

    parser.add_argument('--saveImg', default=True, type=bool, help='save a batch of images (default=False)')
    parser.add_argument('--saveImg_path', default='Output4713f_16_16_8', help='version of checkpt to save(default: v7sa)')
    parser.add_argument('--saveImgBPC', default = 16, type=int, help='number of bits per color channel, ignored for exr extension')
    parser.add_argument('--saveExt', default='.exr', help='folder for reference image (None=>dont save)')
    parser.add_argument('--saveInput_folder', default='Input', help='folder for input image (None=>dont save)')
    parser.add_argument('--saveOutput_folder', default='Output', help='folder for output image (None=>dont save)')
    parser.add_argument('--saveRef_folder', default='Reference', help='folder for reference image (None=>dont save)')

    
    args = parser.parse_args()

    if args.location == 'home':
        args.chk_save_pre='checkpoints/base/ddpm'
        args.chk_load_pre='checkpoints/base/ddpm'
        args.dataset_root='/Volumes/neil_SSD/datasets/'
    elif args.location =='truecut':
        args.chk_save_pre='/tt/ssd0/checkPoint/ddpm'
        args.chk_load_pre='/tt/ssd0/checkPoint/ddpm'
        args.dataset_root='/tt/ssd0/datasets/'
    elif args.location =='custom':
        args.chk_save_pre='checkpoints/base/ddpm'
        args.chk_load_pre='checkpoints/base/ddpm'
        args.dataset_root='~/Desktop/'

    if device.type != 'cuda':
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Set random seed for reproducibility
    #os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    if args.dif_rng_seed == 0:
        manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random Seed (random): ", manualSeed)
    else:
        manualSeed = args.dif_rng_seed
        print("Random Seed (fixed): ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # load the models
    model_low_res, model_high_res = load_train_objs(args,device)
    # load the data
    image, imgVpix, imgGpix = get_data(args)
    # show the image
    # convert image to look like a batch of 1 because get_data only returns 3 dim.
    show_img(args,model_low_res,model_high_res,image.unsqueeze(0), imgVpix.unsqueeze(0), imgGpix.unsqueeze(0), device)