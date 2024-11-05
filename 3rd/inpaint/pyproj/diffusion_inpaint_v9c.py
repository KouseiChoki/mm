import time
import math
import torch
import torch.distributed
import torchvision.transforms.functional as TF
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
#from torch.optim import lr_scheduler
#import torch.cuda.amp as amp 
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.autograd import Variable
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
#from IPython.display import display
from PIL import Image
from copy import deepcopy
from collections import OrderedDict
import cv2

from diffusion_data_v2 import get_dataset, inverse_transform
#from diffusion_data import inpaint_train_dataset as inpaint_dataset
from diffusion_data_v2 import inpaint_mask_dataset as inpaint_dataset
from diffusion_gen_v9 import SimpleDiffusion, get
from diffusion_gen_v9 import Model_T as Model
#from lion_opt import Lion

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    
def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    #plt.title(title)
    plt.show()
"""
def saveImg8bit(image,path,folder,name,ext,transform=None):
    img_path = os.path.join(path,folder,name + '.' + ext)
    print(img_path)
    if transform is not None:
        image = transform(image)
    else:
        image = image*255.0
    image = image.to('cpu')
    #pil_image = Image.fromarray(np.uint8(image.numpy()))
    image = torch.permute(image,(1,2,0)).numpy()
    imgInt = image.astype('uint8')
    #outImg = cv2.cvtColor(np.uint8(image.numpy()), cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path,imgInt)
    #pil_image.save(img_path, format=ext)

def saveImg16bit(image,path,folder,name,ext,transform=None):
    img_path = os.path.join(path,folder,name + '.' + ext)
    if transform is not None:
        image = transform(image)
    else:
        image = image*65535.0
    image = image.to('cpu')
    pil_image = Image.fromarray(np.uint16(image.numpy()))
    pil_image.save(img_path, format=ext)
"""
def saveImg(image,path,folder,name,ext,bits,transform=None):
    img_dir = os.path.join(path,folder)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print(f"The new directory {path} is created!")        
    img_path = os.path.join(img_dir,name + '.' + ext)
    if transform is not None:
        image = transform(image)
    elif not (ext == 'exr'):
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

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def ddp_setup(rank: int, world_size: int, device):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if device.type == 'cuda':
        #init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        init_process_group(backend="gloo", rank=rank, world_size=world_size)

def prepare_dataloader(args, worldSize, dataset: Dataset, rng):
    if args.n_epochs == 1:
        shuffle_data = False # shuffling is not exclusive between GPU's which causes issues at evaluation
    else:
        shuffle_data = True
    if worldSize == 1:
        outLoader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True, 
            generator=rng,
            shuffle=True,
        )
    else:
        outLoader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True, 
            shuffle=False,
            sampler=DistributedSampler(dataset,num_replicas=worldSize,shuffle = shuffle_data)
        )
    return outLoader
def load_train_objs(args):

    '''
    image_transforms = transforms.Compose([
        transforms.Resize(args.imgSize[-1],antialias=True), # resize the smallest edge to this dimension
        transforms.RandomCrop(args.imgSize,pad_if_needed=True), # then random crop
        #transforms.Resize([args.imgH,args.imgW]),
        transforms.RandomHorizontalFlip(), # for inpainting vertical and 90 degree flip creates unnatural images
        ])
    mask_transforms = transforms.Compose([
        #transforms.Resize([args.imgH,args.imgW],transforms.InterpolationMode.NEAREST), # resize to CNN input
        transforms.Resize(args.imgSize,antialias=True), # resize to CNN input
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
        ])
    '''
    crop_transforms = transforms.Compose([
        transforms.CenterCrop(args.imgSize), # then random crop
        ])
    scale_transforms = transforms.Compose([
        transforms.Resize(args.imgSize,antialias=True), # resize the smallest edge to this dimension
        ])
    
    train_img_transforms = transforms.Compose([
        transforms.Resize(args.imgSize[0],antialias=True), # resize the smallest edge to this dimension
        transforms.ColorJitter(brightness=[0.25,1],contrast=[0.25,1]), # backgrounds can have much lower brightness/gain than total image
        transforms.RandomCrop(args.imgSize,pad_if_needed=True), # then random crop
        transforms.RandomHorizontalFlip(), # for inpainting vertical and 90 degree flip creates unnatural images
        ])

    train_mask_transforms = transforms.Compose([
        transforms.Resize(args.imgSize,antialias=True), # resize to CNN input
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
        ])
    if args.showImg or args.saveImg:
        if args.imgScale:
            image_transforms = scale_transforms
            mask_transforms = scale_transforms
        else:
            image_transforms = crop_transforms
            mask_transforms = crop_transforms
    else:
        image_transforms = train_img_transforms
        mask_transforms = train_mask_transforms

    if args.dataset_folder is None:
        train_dataset = get_dataset(dataset_name=args.dataset_name,imgSize=args.imgSize)
    else:
        imgTrainFolder = []
        maskTrainFolder = []
        for i in range(len(args.dataset_folder)):
            tempTrainFolder = args.dataset_root + args.dataset_folder[i]
            imgTrainFolder.append(tempTrainFolder)
            print(f'[last path: {tempTrainFolder}] All paths: {imgTrainFolder}')
        imgTrainFolder = imgTrainFolder # don't ask, me: += gives me a list of characters and append adds a level in the heirarchy
        for i in range(len(args.mask_folder)):
            tempMaskFolder = args.dataset_root + args.mask_folder[i]
            maskTrainFolder.append(tempMaskFolder)
            print(f'[last path: {tempMaskFolder}] All paths: {maskTrainFolder}')
        maskTrainFolder = maskTrainFolder # don't ask, me: += gives me a list of characters and append adds a level in the heirarchy
        #maskTrainFolder = args.dataset_root + 'inPaintMask/train/random_masks/'
        train_dataset = inpaint_dataset(args, imgTrainFolder,maskTrainFolder, imgTransform=image_transforms,maskTransform=mask_transforms)
    
    model = Model(base_channels=32,
                channel_mult=[16,16,16,8,4,2,1],
                attn_layers=[True,True,True,False,False,False],
                num_atten=[4,2,1,1,1,1],
                patchSizes=[[1,1],[2,2],[4,4],[1,1],[1,1],[1,1]],
                con_attn_num=8, # with 3x2 in the cond and 3x2 in the decode, that makes 20 total
                attn_channels=1024,
                num_heads=4,
                mlp_ratio=4.0,
                dropout_rate=args.dropout_p,
                time_emb_dims=512)
    
    if args.loadModel:
        checkpoint_pre = torch.load(args.chk_load_pre + '_model_' + args.chk_load_ver + '.pth',map_location='cpu')
        model.load_state_dict(checkpoint_pre['model_state_dict'], strict=False)

    ema_m = deepcopy(model)

    if args.loadEMA:
        checkpoint_pre = torch.load(args.chk_load_pre + '_ema_m_' + args.chk_load_ver + '.pth',map_location='cpu')
        ema_m.load_state_dict(checkpoint_pre['ema_m_state_dict'], strict=False)

    requires_grad(ema_m, False)
    ema_m.eval()

    if args.showImg:
        model.eval()
    else:
        model.train()

    if os.path.exists(args.chk_save_pre + '_model_' + args.chk_save_ver + '.pth'):
        # successful load for training, back up
        shutil.copy(args.chk_save_pre + '_model_' + args.chk_save_ver + '.pth',args.chk_save_pre + '_model_' + args.chk_save_ver + '.bak')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.imgLR)
    #optimizer = Lion(model.parameters(), lr=args.imgLR)
    loss = nn.MSELoss()
    #loss = RMSELoss()
    #gScaler = amp.GradScaler()

    return train_dataset, model, ema_m, optimizer, loss

class Trainer:
    def __init__(
        self,
        args,
        img_data: DataLoader,
        #sd: torch.nn.Module,
        model: torch.nn.Module,
        ema_m: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss,
        tDevice: torch.device,
        world_size
    ) -> None:
        self.args = args
        self.device = tDevice
        self.world_size = world_size
        self.img_data = img_data
        self.optimizer = optimizer
        self.model = model.to(tDevice)
        self.ema_m = ema_m.to(tDevice)
        self.loss = loss
        self.aLoss = None
        self.sd = SimpleDiffusion(num_diffusion_timesteps=args.time_steps, device=tDevice)
        #self.sd = sd
        if world_size > 1:
            #self.model = DDP(model,device_ids=[tDevice.index],find_unused_parameters=True)
            self.model = DDP(model,device_ids=[tDevice.index],find_unused_parameters=True)

    def _run_batch(self,x0s,xm0s, ts, lastMB):
        
        mask = xm0s[:,3:4,:,:]
        mask = 1-F.max_pool2d(1-mask,3,stride=1,padding=1) # using this in training and I don't want have an issue with boundary effects
        xts, gt_noise = self.sd.forward_diffusion(x0s, ts)

        #pred_noise, ec = self.model(xts, ts, min_mask,xm0s[:,0:3,:,:],mask,ec=None)
        pred_noise,ec = self.model(xts, ts,xm0s[:,0:3,:,:],mask, ec=None)
        bloss = self.loss(gt_noise,pred_noise)
        self.aLoss = bloss/self.args.num_batches
        self.aLoss.backward()
        if lastMB:
            self.optimizer.step()
            self.optimizer.zero_grad()

        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        #self.optimizer.step()
        if self.device.index == 0:
            if self.world_size > 1:
                update_ema(self.ema_m,self.model.module)
            else:
                update_ema(self.ema_m,self.model)
        
        loss_value = bloss.detach().item()
        #loss_record.update(loss_value)

        #tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")
 
        return loss_value
    
    def _run_epoch(self, epoch):
        if epoch == 0:
            print(f'[GPU: {self.device}] Epoch: {epoch} | B_SZ: {self.args.batch_size} | Steps: {len(self.img_data)}')
        # next three lines for multi-GPU
        if self.world_size != 1:
            epoch_rnd = random.randint(0,255)
            self.img_data.sampler.set_epoch(epoch_rnd) # sampler shuffles based on epoch number
        n_total_steps = len(self.img_data)
        startTime = time.time()
        cLoss = 0.0
        last_batch = False
        for i, (refImages, inputStacks, imgNames, imgDirs) in enumerate(self.img_data):
            refImages = scaleImg(refImages.to(self.device))
            inputStacks = scaleImg(inputStacks.to(self.device))
            if i == n_total_steps-1:
                last_batch = True
            if self.args.showImg or self.args.saveImg:
                self.show_img(refImages,inputStacks,imgNames,imgDirs,last_batch)
                #self.show_img(refImages, inputStacks)
            else:
                if (i+1) % self.args.num_batches == 0  or (i + 1 == len(self.img_data)):
                    lastMB = True
                else:
                    lastMB = False
                ts = torch.randint(low=1, high=self.args.time_steps, size=(refImages.shape[0],), device=self.device) # line
                loss = self._run_batch(refImages,inputStacks,ts,lastMB) # high res is always trained with critic
                #loss = self._run_batch(refImages, inputStacks,ts) # high res is always trained with critic
                cLoss += loss
                
                if self.device.index == 0 and (i+1) % self.args.save_every == 0:
                    if not (self.args.showImg or self.args.saveImg):
                        self._save_checkpoint(self.args.chk_save_pre,self.args.chk_save_ver)

                if self.device.index == 0 and (i+1) % self.args.stats_every == 0:
                    endTime = time.time()
                    timePerImage = (endTime-startTime)*1000/self.args.stats_every/self.args.batch_size/self.world_size
                    print(f'Epoch [{epoch+1}/{self.args.n_epochs}]  Step [{i+1}/{n_total_steps}]  Time/image(mS) {timePerImage:.4f} mse_loss: {cLoss/self.args.stats_every:.4f}')
                    cLoss = 0.0
                    startTime = time.time()

    def _save_checkpoint(self, chk_save_pre, chk_save_ver):
        if not self.args.loadEMA: # we didn't load the EMA, so let's force the ema to equal the model
            if self.world_size > 1:
                update_ema(self.ema_m,self.model.module,decay=0)
            else:
                update_ema(self.ema_m,self.model,decay=0)
            self.args.loadEMA = True # then set it as if we loaded it
        torch.save({
            'ema_m_state_dict': self.ema_m.state_dict(),
            }, chk_save_pre + '_ema_m_' + chk_save_ver + '.pth')
        if self.world_size > 1:
            torch.save({
                'model_state_dict': self.model.module.state_dict(),
                }, chk_save_pre + '_model_' + chk_save_ver + '.pth')
        else:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                }, chk_save_pre + '_model_' + chk_save_ver + '.pth')
    """
    def show_img(self,noisy_images):
        _, ax = plt.subplots(1, len(noisy_images), figsize=(10, 5), facecolor='white')
        
        for i, (timestep, noisy_sample) in enumerate(zip(self.args.show_time_steps, noisy_images)):
            ax[i].imshow(noisy_sample.squeeze(0).permute(1, 2, 0))
            ax[i].set_title(f"t={timestep}", fontsize=8)
            ax[i].axis("off")
            ax[i].grid(False)
        
        plt.suptitle("Forward Diffusion Process", y=0.9)
        plt.axis("off")
        plt.show()        

    """
    def p_sample(self,sd,x,ats,z,ts,img,mask,ec):
        if self.args.loadEMA:
            #predicted_noise, ec = self.ema_m(x, ats,m,img,mask,ec)
            predicted_noise, ec = self.ema_m(x, ats,img,mask,ec)
        else:
            #predicted_noise, ec = self.model(x, ats,m,img,mask,ec)
            predicted_noise, ec = self.model(x, ats,img,mask,ec)

        beta_t                            = get(sd.beta, ts)
        one_by_sqrt_alpha_t               = get(sd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(sd.sqrt_one_minus_alpha_cumulative, ts) 
        x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )
        return x, ec

    @torch.no_grad()
    def show_img(self,ref_img,stack,imgNames,imgDirs,last_batch):
        batch_size = ref_img.shape[0]
        mask = stack[:,3:4,:,:]
        img = stack[:,0:3,:,:].detach().clone()

        img_temp = invScaleImg(img)
        img_apl = torch.mean(img_temp,dim=[1,2,3],keepdim=True)
        img_apl_exp = torch.log(img_apl)/math.log(0.3)
        img_apl_exp = torch.clamp(img_apl_exp,1,3) # if image is too dark, then we apply gamma to lighten up the dark areas
        img = torch.pow(img_temp,1.0/img_apl_exp)
        img = scaleImg(img)

        # there are some instances of the mask not being quite right so expand by 1 pixel
        #mask = 1-F.max_pool2d(1-mask,3,stride=1,padding=1)
        if self.args.makeMaskBinary:
            mask = torch.floor(mask) # make sure that it's binary and only valid pixels are marked as such
        hole = 1-mask
        for i in range(self.args.dilateHoleMult):
            hole = F.max_pool2d(hole,3,stride=1,padding=1)
        #p0s = smpl_inpaint(img*mask,mask,self.args.sizeLayers) # the inference does this step before, so it needs to be part of training too
        mask = 1-hole
        mask_img = img*mask
        #mask_img = img*mask + p0s*(1-mask)
        model_mask = torch.ones_like(mask)
        #mask_img = smpl_inpaint(mask_img,mask,self.args.sizeLayers)
        #inner_mask = 1-F.max_pool2d(mask,65,stride=1,padding=32)
        #first_mask = inner_mask+mask
        org_mask = mask.clone()
        #mask = 1-F.max_pool2d(1-mask,3,stride=1,padding=1)
        #mask = 1-F.max_pool2d(1-mask,3,stride=1,padding=1)
        #nimg_mask = torch.ones_like(mask)-mask
        #mask_img2use = img*img_mask
        mask_img2use = mask_img
        self.model.eval()
        self.ema_m.eval()
        one_ts = torch.ones(batch_size, dtype=torch.long, device=self.device)
        if (self.args.dif_rng_seed == 0 or (not self.args.inPaint)) and not self.args.saveImg:
            torch.manual_seed(random.randint(1, 10000))
            z = torch.randn_like(mask_img)
        else:
            torch.manual_seed(self.args.dif_rng_seed)
            z = torch.randn_like(mask_img[0:1,:,:,:])
            z.repeat(batch_size,1,1,1)
        if not self.args.inPaint:
            x = z
        else:
            x, _gtn = self.sd.forward_diffusion(mask_img,one_ts*(self.args.start_sample_steps-1),noise2use=z)
        
        time_step = self.args.start_sample_steps
        show_index = 0
        pbar = tqdm(total=self.args.start_sample_steps-2)
        sd_ver = -1
        repeat_count = 0
        if not self.args.inPaint:
            variable_steps = False
            step_size = 1
            steps_back = 1
            skip_forward = 1
        else:
            variable_steps = True
            step_size = 1024//self.args.num_sample_steps
            steps_back = self.args.steps_back
        sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=self.device)
        repeat = 0
        min_t = 0
        ec = None # set this to none so that it is generated the first time through.
        while time_step > 1: # needs to be > 1 so that indexing to get betas works below
            if variable_steps:
                if time_step > 768:
                    if sd_ver == -1:
                        #step_size = 2
                        step_size = 4
                        sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=self.device)
                        sd_ver = 0
                        steps_back = 4 # first is 1000->500
                        #steps_back = 4 # first is 1000->500
                        skip_forward = 1 # must be GE 1
                        repeat = 1
                        min_t = 0
                        #mask = second_mask
                        #nimg_mask = torch.ones_like(mask)-mask
                elif time_step > 512:
                    if sd_ver == 0:
                        #step_size = 2
                        step_size = 4
                        sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=self.device)
                        sd_ver = 1
                        #steps_back = 8 # first is 1000->500
                        steps_back = 4 # first is 1000->500
                        skip_forward = 1
                        repeat = 0
                        min_t = 0
                        #mask = second_mask
                        #nimg_mask = torch.ones_like(mask)-mask
                elif time_step > 256:
                    if sd_ver <= 1:
                        #step_size = 2
                        step_size = 4
                        sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=self.device)
                        sd_ver = 2
                        #steps_back = 8 # first is 500 -> 250 - 20%
                        steps_back = 4 # first is 500 -> 250 - 20%
                        skip_forward = 1
                        repeat = 0
                        min_t = 0
                        #mask = second_mask
                        #nimg_mask = torch.ones_like(mask)-mask
                elif time_step > 128:
                    if sd_ver <= 2:
                        step_size = 2
                        sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=self.device)
                        sd_ver = 3
                        steps_back = 4 # first is 150/5 = 20 
                        skip_forward = 1
                        repeat = 0
                        min_t = 0
                        #mask = second_mask
                        #nimg_mask = torch.ones_like(mask)-mask
                elif time_step > 64:
                    if sd_ver <= 3:
                        step_size = 2
                        sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=self.device)
                        sd_ver = 4
                        steps_back = 2 #100->50
                        skip_forward = 1
                        repeat = 0
                        min_t = 0
                        #mask = second_mask
                        #nimg_mask = torch.ones_like(mask)-mask
                elif time_step > 32:
                    if sd_ver <= 4:
                        step_size = 1
                        sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=self.device)
                        sd_ver = 5
                        steps_back = 1
                        skip_forward = 1
                        repeat = 0
                        min_t = 0
                        #mask = second_mask
                        #nimg_mask = torch.ones_like(mask)-mask
                else:
                    if sd_ver <= 5:
                        step_size = 1
                        sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=self.device)
                        sd_ver = 6
                        steps_back = 1
                        skip_forward = 1
                        repeat = 0
                        min_t = 0
                        #mask = org_mask
                        #nimg_mask = torch.ones_like(mask)-mask

            adj_time_step = (time_step-min_t)//step_size - 1
            #ts = one_ts * adj_time_step * step_size # the model restore has to be in the 0:999 range
            ts = one_ts * time_step - 1 # the model restore has to be in the 0:1023 range
            
            num_steps = min(adj_time_step,steps_back)

            #if self.args.steps_back_range[0] >= time_step and self.args.steps_back_range[1] <= time_step:
            #    num_steps = min(adj_time_step,self.args.steps_back)
            #else:
            #    num_steps = min(adj_time_step,1)
            if not self.args.inPaint:
                z = torch.randn_like(x[0:1,:,:,:]) if adj_time_step > 1 else torch.zeros_like(x)
                z.repeat(batch_size,1,1,1)
                x, ec = self.p_sample(sd,x,ts,z,one_ts*adj_time_step,img,mask,ec)
                #x = self.p_sample(sd,x,ts,mask,z,one_ts*adj_time_step)
                #min_mask = (1024. - torch.reshape(ts,[-1,1,1,1]))/1024.
                #min_mask = torch.maximum(mask,min_mask)
                #if self.args.loadEMA:
                #    x0s = self.ema_m(x, ts, min_mask)
                #else:
                #    x0s = self.model(x, ts, min_mask)
                #x = mask_img*mask + x0s*nimg_mask
                #x = x0s
                #adj_time = adj_time_step# 
                #if adj_time > 0:
                    # use same noise for all the steps
                    #kwnImg, gt_noise = self.sd.forward_diffusion(mask_img,one_ts*(adj_time-1))
                    #z = torch.randn_like(mask_img2use[0:1,:,:,:])
                    #z.repeat(batch_size,1,1,1)
                #    x, _gtn = sd.forward_diffusion(x,one_ts*(adj_time-1),noise2use=z)

            elif num_steps > 0:
                for i in range(num_steps):
                    z = torch.randn_like(x[0:1,:,:,:]) if adj_time_step > 1 else torch.zeros_like(x)
                    z.repeat(batch_size,1,1,1)
                    adj_time = adj_time_step - i# 
                    ts = one_ts * (adj_time * step_size + step_size -1) # hmm, should this be 0,4,...1020 or 3, 7,... 1023
                    x,ec = self.p_sample(sd,x,ts,z,one_ts*adj_time_step,img,mask,ec)
                    #x = self.p_sample(sd,x,ts,mask,z,one_ts*adj_time)
                    #adj_time_step = adj_time*t_scale
                    if adj_time > 0:
                        # use same noise for all the steps
                        #kwnImg, gt_noise = self.sd.forward_diffusion(mask_img,one_ts*(adj_time-1))
                        #z = torch.randn_like(mask_img2use[0:1,:,:,:])
                        #z.repeat(batch_size,1,1,1)
                        #kwnImg, _gtn = sd.forward_diffusion(mask_img,one_ts*(adj_time-1),noise2use=z)
                        kwnImg, _gtn = sd.forward_diffusion(mask_img,one_ts*(adj_time),noise2use=z) # seems a little better than the above
                    else:
                        kwnImg = mask_img.detach().clone()
                    # then blend with what the original should be
                    #x = kwnImg*mask + x*nimg_mask # this coiuld be zero, but there are more diffusion steps
                    x = kwnImg*org_mask + x*(1-org_mask)
                # going forward in time one less than going backward
                if repeat_count == repeat:
                    num_steps -= skip_forward
                    repeat_count = 0
                else:
                    repeat_count += 1
                num_steps = max(0,num_steps)
                for j in reversed(range(num_steps)):
                    #ts += t_scale # increment ts to next output position
                    #beta_t = get(self.sd.beta, one_ts*(adj_time-1)) # definitely not the correct method!
                    beta_t = get(sd.beta, one_ts*(adj_time))
                    z = torch.randn_like(mask_img[0:1,:,:,:])
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
                adj_time = adj_time_step-i# 
                ts = one_ts * (adj_time * step_size + step_size - 1)
                x,ec = self.p_sample(sd,x,ts,z,one_ts*adj_time_step,img,mask,ec)
                #x = self.p_sample(sd,x,ts,mask,z,one_ts*adj_time_step)
                x = mask_img*org_mask + x*(1-org_mask)
            #"""
            if time_step <= self.args.show_time_steps[show_index] and self.args.showImg:
                #write the images using the show_time_step in the file name
                #fake_data = self.vae.decode(x.detach().clone() / 0.18215).sample
                #fake_data = mImg[:,0:3,:,:]*img_mask + fake_data*(1-img_mask)
                fake_data = torch.clip(invScaleImg(x),0,1)
                # undo the gamma adjustment and then merge with the original
                #fake_data = torch.pow(fake_data,img_apl_exp)
                #fake_data = stack[:,0:3,:,:]*stack[:,3:4,:,:] + fake_data*(1-stack[:,3:4,:,:])
                outputs = fake_data.to('cpu')
                outputs = bgr2rgb_batch(outputs)
                inputs = bgr2rgb_batch(invScaleImg(mask_img)).to('cpu')
                refImages = bgr2rgb_batch(invScaleImg(ref_img)).to('cpu')
                # reorganize the data to display: first row content, second,depth, third vpix
                #RGBvPix = 1-torch.cat([inputStack[:,3:4,:,:],inputStack[:,3:4,:,:],inputStack[:,3:4,:,:]],1)
                debugImg = torch.cat([inputs,outputs,refImages],0)
                img_out = torchvision.utils.make_grid(debugImg,nrow=batch_size,normalize=False)
                #hr_out = torchvision.utils.make_grid(img_inputs[hr_img])
                imshow(img_out)
                show_index += 1
            if repeat_count == 0:
                time_step -= step_size*skip_forward
                pbar.update(step_size*skip_forward)
                #mask = F.max_pool2d(mask,3,1,padding=1)
        if self.args.saveImg:
            x = torch.clip(x,-1.0,1.0)
            x = invScaleImg(x)
            x = torch.pow(x,img_apl_exp)
            #x = invScaleImg(stack[:,0:3,:,:])*stack[:,3:4,:,:] + x*(1-stack[:,3:4,:,:])
            x = invScaleImg(stack[:,0:3,:,:])*mask + x*(1-mask)
            mask_img = inverse_transform(mask_img,1)
            img = inverse_transform(img,1)
            for i in range(len(imgNames)):
                if self.args.img_subfolder is None:
                    savePath = self.args.saveImg_path
                else:
                    savePath = imgDirs[i]
                if self.args.saveInput_folder is not None:
                    saveImg(mask_img[i,:,:,:],savePath,self.args.saveInput_folder,imgNames[i],self.args.saveExt,self.args.saveImgBPC)
                if self.args.saveOutput_folder is not None:
                    saveImg(x[i,:,:,:],savePath,self.args.saveOutput_folder,imgNames[i],self.args.saveExt,self.args.saveImgBPC)
                if self.args.saveRef_folder is not None:
                    saveImg(img[i,:,:,:],savePath,self.args.saveRef_folder,imgNames[i],self.args.saveExt,self.args.saveImgBPC)
            if last_batch:
                print(f'last batch {self.device}')
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
         
    def train(self):
        for epoch in range(self.args.n_epochs):
            torch.cuda.empty_cache()
            #gc.collect()
            
            # Algorithm 1: Training
            self._run_epoch(epoch=epoch)
            if self.device.index == 0:
                self._save_checkpoint(self.args.chk_save_pre,self.args.chk_save_ver)
    """
                if epoch % 20 == 0:
                    #save_path = os.path.join(log_dir, f"{epoch}{ext}")
                    
                    # Algorithm 2: Sampling
                    self.sd.reverse_diffusion(self.model, timesteps=self.args.time_steps, num_images=32, generate_video=False,
                        img_shape=self.args.imgShape, device=self.device,
                    )
    """
def train_model(rank: int, world_size: int, args, device):
    myRNG = torch.Generator()
    if args.dif_rng_seed != 0:
        myRNG.manual_seed(args.dif_rng_seed)
    new_device = torch.device(device.type,rank) # this creates the device such that cuda and mps look the same
    ddp_setup(rank, world_size, new_device) # for multi-GPU
    train_dataset, model, ema_m, optimizer, loss = load_train_objs(args)
    trainLoader = prepare_dataloader(args, world_size, train_dataset, myRNG)
    trainer = Trainer(
            args,
            trainLoader,
            model,
            ema_m,
            optimizer,
            loss,
            new_device,
            world_size)
    trainer.train()
    destroy_process_group()

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
    parser.add_argument('--location', default='home', help='location of server(default: home)')
    parser.add_argument('--max_wSize', default=8, type=int, help='max number of GPUs to use (default: 8)')
    parser.add_argument('--n_epochs', default=10, type=int, help='Total epochs to train the model (default: 10)')
    parser.add_argument('--save_every', default=1024, type=int, help='How often to save a snapshot (default: 1024)')
    parser.add_argument('--stats_every', default=128, type=int, help='How often to print statistics (default: 128)')
    parser.add_argument('--batch_size', default=1, type=int, help='Input batch size on each device (default: 1)')
    parser.add_argument('--num_batches', default=1, type=int, help='number of batches to accumulate before backward (default: 2)')
    parser.add_argument('--chk_save_pre', default='checkPoint/ddpm', help='location of checkpoint file and prefix, overriden by location')
    parser.add_argument('--chk_load_pre', default='checkPoint/ddpm', help='location of checkpoint file and prefix, overriden by location')
    # note that check points have a fixed name "_model_" or "_ema_m_" inserted between prefix and the version below
    parser.add_argument('--chk_save_ver', default='mdl1k_v9c_temp', help='version of checkpt to save')
    parser.add_argument('--chk_load_ver', default='mdl1k_v9c_base', help='version of checkpt to load')
    parser.add_argument('--loadModel', default=True, type=bool, help='load CNN model (default=True))')
    parser.add_argument('--loadEMA', default=True, type=bool, help='load EMA CNN model (default=True))')
    parser.add_argument('--trainClassifier', default=True, type=bool, help='The classifier part of the model (default=True))')
    parser.add_argument('--imgLR', default=1e-5, type=float, help='MV learning rate (default: 0.00005)')
    parser.add_argument('--num_workers', default=2, type=int, help='number of dataloader processes(default: 0)')
    
    # value below might be a mistake. In inference I will set the ts as high as 1023, not 999, so it could cause issues
    parser.add_argument('--time_steps', default=1024, type=int, help='number of diffusion time steps (default: 1000)')
    parser.add_argument('--dropout_p', default=.125, type=float, help='number of diffusion time steps (default: 1000)')
    parser.add_argument('--dif_rng_seed',default=0, type=int, help='rng seed for diffusion, 0=>rand seed except during inference (default: 0)')
    parser.add_argument('--dataset_name', default='flowers', help='name of pytorch built-in dataset to use if a path isnt given')
    parser.add_argument('--dataset_root', default='dummy', help='root folder for datasets (default: /Volumes/neil_SSD/datasets/)')
    #parser.add_argument('--dataset_folder', default='Unreal/train/clean/', help='root folder for datasets (default: /Volumes/neil_SSD/datasets/)')
    #parser.add_argument('--dataset_folder', default=["imagenet-1k/train","raw_celeba/img_celeba","DIV2K/images","AIM500/original","woodall"],type=list, help='root folder for datasets (default: /Volumes/neil_SSD/datasets/)')
    #parser.add_argument('--dataset_folder', default=["imagenet-1k/train","raw_celeba/img_celeba"],type=list, help='root folder for datasets (default: /Volumes/neil_SSD/datasets/)')
    #parser.add_argument('--mask_folder', default='inPaintMask/train/random_masks/', help='folder for masks')
    # below is what I use for inference inpainting testing with the image that you supplied
    parser.add_argument('--dataset_folder', default=["AV2/2039A_0020_v0795/2d_toMesh/Rendering_Pattern/render_out_0930_JM0_binary"],type=list, help='root folder for dataset')
    parser.add_argument('--mask_folder', default=["AV2/2039A_0020_v0795/2d_toMesh/Rendering_Pattern/render_out_0930_JM0_binary"], help='folder for masks')
    parser.add_argument('--img_subfolder', default="None", help='sub folder for masks, None = search all sub folders')
    parser.add_argument('--mask_subfolder', default="None", help='folder for masks, None = search all sub folders')
    # not sure if below is used, so make sure that --imgSize parameter is a multiple
    parser.add_argument('--sizeMult', default=64, type=int, help='pixel mulitple image needs to be)')
    # next two lines are used for inference, the two lines after that for training
    parser.add_argument('--imgSize', default=(1088,2048), type=int, nargs=2, help='pixel mulitple image needs to be)')
    parser.add_argument('--imgScale', default=False, type=bool, help='scale image, otherwise center crop')
    #parser.add_argument('--imgSize', default=(512,512), type=int, nargs=2, help='pixel mulitple image needs to be)')
    #parser.add_argument('--imgScale', default=True, type=bool, help='scale image, otherwise center crop')
    parser.add_argument('--imgIs16', default=False, type=bool, help='bit width of color channel, ignored if .exr (default=False))')
    parser.add_argument('--imgChnl', default=3, type=int, help='number of color channels to use (default=3)')

    parser.add_argument('--maskIsPV', default=False, type=bool, help='high value is pixel valid (default: False)')
    parser.add_argument('--maskIs16', default=False, type=bool, help='mask is 16 bit (default: False)')
    parser.add_argument('--chromaKey', default=None, help='set to None to disable')
    #parser.add_argument('--getRandMask', default=True, type=bool, help='get a random mask and not masks paired with image')
    #parser.add_argument('--chromaKey', default=[1,0,0], help='[R,G,B] format, values based on file format, set to None to disable')
    parser.add_argument('--getRandMask', default=False, type=bool, help='get a random mask and not masks paired with image')
    #parser.add_argument('--inMaskG', default=512., type=float, help='mask gain value (default: 512)')
    #parser.add_argument('--maskOff', default=[0.375,0], type=float, nargs=2, help='mask offset only [0] is used(default: [0.375,0])')
    parser.add_argument('--useMask', default=False, type=bool, help='use mask information for training or inference')
    parser.add_argument('--makeMaskBinary', default=False, type=bool, help='use mask information for training or inference')
    parser.add_argument('--dilateHoleMult',default=4,type=int,help='number of times to do 3x3 max operation on hole')
    
    parser.add_argument('--showImg', default=True, type=bool, help='display a batch of images (default=False)')
    parser.add_argument('--inPaint', default=True, type=bool, help='inpaint a batch of images (default=False)')
    parser.add_argument('--steps_back', default = 8, type=int, help='number of steps forward in time when gen image')
    parser.add_argument('--steps_back_range', default =[1024,1], nargs=2, type=int, help='range for going back and forth in time')
    
    parser.add_argument('--start_sample_steps', default =1024,type=int, help='starting time step for generating image')
    #parser.add_argument('--start_sample_steps', default =16,type=int, help='starting time step for generating image')
    #parser.add_argument('--num_sample_steps', default =1000, type=int, help='number of steps in generating an image')
    #parser.add_argument('--show_time_steps', default=[999,800,500,249,128,64,32,16,8,4,2,1,0], type=int, help='time steps to show progress')
    parser.add_argument('--num_sample_steps', default =256, type=int, help='number of steps in generating an image')
    parser.add_argument('--show_time_steps', default=[1023,768,512,256,128,64,4,2,1,0], nargs='*', type=int, help='time steps to show progress, set to zero to skip')
    #parser.add_argument('--show_time_steps', default=[1023,4,2,1,0], nargs='*', type=int, help='time steps to show progress, set to zero to skip')
    #parser.add_argument('--show_time_steps', default=[0], nargs='*', type=int, help='time steps to show progress, set to zero to skip')
    parser.add_argument('--showImg_num', default=2, type=int, help='number of images to generate')

    parser.add_argument('--saveImg', default=True, type=bool, help='save a batch of images (default=False)')
    #parser.add_argument('--saveImg_path', default='Images', help='version of checkpt to save(default: v7sa)')
    parser.add_argument('--saveImg_path', default='/tt/ssd0/neil/images', help='version of checkpt to save(default: v7sa)')
    parser.add_argument('--saveExt', default='exr', help='folder for reference image (None=>dont save)')
    parser.add_argument('--saveImgBPC', default = 16, type=int, help='number of bits per color channel, ignored for exr extension')
    parser.add_argument('--saveInput_folder', default=None, help='folder for input image (None=>dont save)')
    parser.add_argument('--saveOutput_folder', default='inpaint', help='folder for output image (None=>dont save)')
    parser.add_argument('--saveRef_folder', default=None, help='folder for reference image (None=>dont save)')

    args = parser.parse_args()

    if args.location == 'home':
        args.chk_save_pre='checkpoints/base/ddpm'
        args.chk_load_pre='checkpoints/base/ddpm'
        args.dataset_root='/Volumes/AI_TrainingData/2Dto3D_Rendering'
    elif args.location =='truecut':
        args.chk_save_pre='/tt/ssd0/checkPoint/ddpm'
        args.chk_load_pre='/tt/ssd0/checkPoint/ddpm'
        args.dataset_root='/tt/ssd0/datasets/'

    if args.showImg:
        args.max_wSize = 1
        args.batch_size = args.showImg_num        
        #args.time_steps = args.num_sample_steps
    
    if args.showImg or args.saveImg:
        args.n_epochs = 1

    if device.type == 'cuda':
    #if device.is_cuda():
        world_size = min(args.max_wSize,torch.cuda.device_count())
    else:
        world_size = 1
    #train_model(CHECK_PT_SAVE,NUM_EPOCHS,B_SZ)
    if device.type != 'cuda':
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


    # Set random seed for reproducibility
    #os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    if args.dif_rng_seed == 0 and not args.saveImg:
        manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random Seed (random): ", manualSeed)
    else:
        manualSeed = args.dif_rng_seed
        print("Random Seed (fixed): ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    #inputDebug()

    mp.spawn(train_model, args=(world_size, args, device), nprocs=world_size)
