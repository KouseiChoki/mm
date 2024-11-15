import time
#import math
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
##from diffusion_data import inpaint_unreal_dataset as inpaint_dataset # train with the unreal dataset
from diffusion_gen_v5 import SimpleDiffusion, Model, get
#from lion_opt import Lion
#from dif_smpl_inpaint import smpl_inpaint

# this is a good template to use for any AI training. It will run on either mac studio or a multi-GPU
# cluster without any code changes. Where the code needs to be different, it checks for the number
# of GPU's being used (world size) to make the appropriate changes

def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    #plt.title(title)
    plt.show()

# this is getting to be pretty generic, so it can probably move to diffusion utilities
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
    return output
#diffusion is usually trained with image content that is between -1 and 1. This provides the scaling to do that
def scaleImg(img):
    if img.shape[1] > 3:
        tmp = 2*img[:,0:3,:,:]-1
        img = torch.concat([tmp,img[:,3:,:,:]],dim=1)
    else:
        img = 2*img - 1
    return img

# go from image scaled for diffusion to image scaled for display and saving
def invScaleImg(img):
    if img.shape[1] > 3:
        tmp = (img[:,0:3,:,:]+1.)*0.5
        img = torch.concat([tmp,img[:,3:,:,:]],dim=1)
    else:
        img = (img+1.)*0.5
    return img

# For diffusion specifically, but also for any training that zero's out channels during training
# better results are achieved by low pass filtering the weights over time
# this was changed from a function to a class so that I can have a persistent value
# of how many times it has been called. This allows quicker convergence if you restart
# it by first doing the equivalent of a unweighted average before switching to a more
# IIR low pass filter method
class update_ema_class:
    def __init__(self, min_ema, max_ema):
        self.ema_decay = min_ema # inverse of IIR weight to start the training
        self.max_ema_decay = max_ema # inverse of min IIR weight during training

    @torch.no_grad()
    def forward(self, ema_model, model, init_ema=False):
        if init_ema:
            decay = 0
        else:
            decay = 1-1./self.ema_decay
        
        self.ema_decay += 1
        self.ema_decay = min(self.ema_decay,self.max_ema_decay)

        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())

        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            #ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
            ema_params[name] = param.data + decay*(ema_params[name]-param.data) # this will have less precision issues


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

# I'm not sure that the MASTER_PORT value needs to be what is specified
# because if you use different values, you can actually get two programs to share the GPU on the mac
def ddp_setup(rank: int, world_size: int, device):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if device.type == 'cuda':
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        init_process_group(backend="gloo", rank=rank, world_size=world_size)

def prepare_dataloader(args, worldSize, dataset: Dataset, rng):
    if args.inPaint or args.saveImg: # then doing evaluation and need to disable shuffling
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
    # called by the train model function which is called by multi-processor
    # spawn method this loads all the objects and their weights/parameters into memory
    if args.showImg or args.saveImg:
        eval_mdl = True
    else:
        eval_mdl = False
    
    crop_transforms = transforms.Compose([
        transforms.CenterCrop(args.imgSize), # then random crop
        ])
    scale_transforms = transforms.Compose([
        transforms.Resize(args.imgSize,antialias=True), # resize the smallest edge to this dimension
        ])
    
    train_img_transforms = transforms.Compose([
        transforms.Resize(args.imgSize[0],antialias=True), # resize the smallest edge to this dimension
        #transforms.ColorJitter(brightness=[0.25,1],contrast=[0.5,1]), # backgrounds can have much lower brightness/gain than total image
        transforms.RandomCrop(args.imgSize,pad_if_needed=True), # then random crop
        transforms.RandomHorizontalFlip(), # for inpainting vertical and 90 degree flip creates unnatural images
        ])

    train_mask_transforms = transforms.Compose([
        transforms.Resize(args.imgSize,antialias=True), # resize to CNN input
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
        ])
    if eval_mdl:
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
        # This takes the list of the folders and creates a list of absolute paths for 
        # the dataset program to use.
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
                #patchSizes=[[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]],
                patchSizes=[[1,1],[2,2],[4,4],[1,1],[1,1],[1,1]],
                #patchSizes=[[1,1],[1,1],[2,2],[1,1],[1,1],[1,1]],
                con_attn_num=8, # with 3x2 in the cond and 3x2 in the decode, that makes 20 total
                attn_channels=1024,
                num_heads=4,
                mlp_ratio=4.0,
                dropout_rate=args.dropout_p,
                time_emb_dims=512)

    # during eval you always want to use the low pass filtered set of weights
    # and the show image function allows the use of two different resolutions during
    # the inpainting, so this just loads the appropriate files based on whether you are
    # doing qualitative evaluation (show or saving an image) or doing training  
    if args.loadModel and (not eval_mdl):
        checkpoint_pre = torch.load(args.chk_load_pre + '_model_' + args.chk_load_ver + '.pth',map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint_pre['model_state_dict'], strict=False)
    elif args.loadEMA or eval_mdl:
        checkpoint_pre = torch.load(args.chk_load_pre + '_ema_m_' + args.chk_load_ver + '.pth',map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint_pre['ema_m_state_dict'], strict=False)

    ema_m = deepcopy(model)

    if args.loadEMA and (not eval_mdl):
        checkpoint_pre = torch.load(args.chk_load_pre + '_ema_m_' + args.chk_load_ver + '.pth',map_location='cpu',weights_only=True)
        ema_m.load_state_dict(checkpoint_pre['ema_m_state_dict'], strict=False)
    elif eval_mdl:
        checkpoint_pre = torch.load(args.chk_load_pre + '_ema_m_' + args.chk_load_hd_ver + '.pth',map_location='cpu',weights_only=True)
        ema_m.load_state_dict(checkpoint_pre['ema_m_state_dict'], strict=False)

    requires_grad(ema_m, False)
    ema_m.eval()

    if eval_mdl:
        model.eval()
    else:
        model.train()


    if os.path.exists(args.chk_save_pre + '_model_' + args.chk_save_ver + '.pth') and not eval_mdl:
        # successful load for training, back up because nothing worse than a crash that corrupts the 
        # checkpoint file
        shutil.copy(args.chk_save_pre + '_model_' + args.chk_save_ver + '.pth',args.chk_save_pre + '_model_' + args.chk_save_ver + '.bak')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.imgLR)
    #optimizer = Lion(model.parameters(), lr=args.imgLR)
    loss = nn.MSELoss()
    #gScaler = amp.GradScaler()

    return train_dataset, model, ema_m, optimizer, loss

# this is the main class where everything is done
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
        #self.sd1000 = SimpleDiffusion(num_diffusion_timesteps=1000, device=tDevice)
        #self.sd = sd
        if world_size > 1:
            self.model = DDP(model,device_ids=[tDevice.index],find_unused_parameters=True)
        self.update_ema = update_ema_class(args.min_ema_decay, args.max_ema_decay)

    def _run_batch(self,x0s, ts,lastMB):
        # implements the ability to accumulate multiple batches before doing back projection
        # use this to keep the number of images used for back projection constant as memory
        # constraints and number of GPU changes. Typically use 256 images before back projection

        xts, gt_noise = self.sd.forward_diffusion(x0s, ts)

        pred_noise = self.model(xts, ts)
        bloss = self.loss(gt_noise,pred_noise)
        self.aLoss = bloss/self.args.num_batches
        self.aLoss.backward()
        if lastMB:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.device.index == 0:
                if self.world_size > 1:
                    self.update_ema.forward(self.ema_m,self.model.module)
                else:
                    self.update_ema.forward(self.ema_m,self.model)
        
        loss_value = bloss.detach().item()
 
        return loss_value
    
    def _run_epoch(self, epoch):
        if epoch == 0:
            print(f'[GPU: {self.device}] Epochs: {self.args.n_epochs} | B_SZ: {self.args.batch_size} | Steps: {len(self.img_data)}')
        # next three lines for multi-GPU
        if self.world_size != 1: # then we want to randomize the images sent to each GPU
            epoch_rnd = random.randint(0,255)
            self.img_data.sampler.set_epoch(epoch_rnd) # sampler shuffles based on epoch number
        n_total_steps = len(self.img_data)
        startTime = time.time()
        cLoss = 0.0
        for i, (refImages, inputStacks, imgNames, imgDirs) in enumerate(self.img_data):
            # format and send images to the GPU
            refImages = scaleImg(refImages.to(self.device))
            inputStacks = scaleImg(inputStacks.to(self.device))
            if self.args.showImg or self.args.saveImg: # then we are doing evaluation, so run show image instead of batch
                self.show_img(refImages,inputStacks,imgNames,imgDirs)
                #self.show_img(refImages, inputStacks)
            else:
                if (i+1) % self.args.num_batches == 0  or (i + 1 == len(self.img_data)):
                    lastMB = True
                else:
                    lastMB = False
                ts = torch.randint(low=1, high=self.args.time_steps, size=(refImages.shape[0],), device=self.device) # line
                loss = self._run_batch(refImages,ts,lastMB) # high res is always trained with critic
                cLoss += loss
                
                if self.device.index == 0 and (i+1) % self.args.save_every == 0:
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
                self.update_ema.forward(self.ema_m,self.model.module,init_ema=True)
            else:
                self.update_ema.forward(self.ema_m,self.model,init_ema=True)
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

    def p_sample(self,sd,x,ats,z,ts,model): # this is the denoising step of the diffusion network
        
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
    def show_img(self,ref_img,stack,imgNames,imgDirs):
        mask = stack[:,3:4,:,:]
        img = stack[:,0:3,:,:].detach().clone()
        batch_size, C, H, W = img.shape

        if self.args.makeMaskBinary:
            mask = torch.floor(mask) # make sure that it's binary and only valid pixels are marked as such
        hole = 1-mask
        for i in range(self.args.dilateHoleMult):
            hole = F.max_pool2d(hole,3,stride=1,padding=1)
        #for i in range(args.dilate_art_reduce):
        #    imgBpix = F.max_pool2d(imgBpix,3,stride=1,padding=1)
        mask_img = img*mask # this is my raw input, I need this because mask = 0 should be zero and it's -1 if img has a hole
                            # this shouldn't make a difference as it is always mulitplied by inpaint_mask in the code below
        start_mask = 1-hole
        #imgGpix = torch.minimum(start_mask,1-imgBpix)

        #p0s = smpl_inpaint(inpaint_mask*img,inpaint_mask,self.args.sizeLayers) # the inference does this step before, so it needs to be part of training too
        #init_img = inpaint_mask*img + p0s*hole
        init_img = start_mask*img
        #mask_img2use = img*img_mask
        one_ts = torch.ones(batch_size, dtype=torch.long, device=self.device)
        #if args.dif_rng_seed == 0 or (not args.inPaint):
        if not self.args.inPaint:
            torch.manual_seed(random.randint(1, 10000))
        else:
            torch.manual_seed(self.args.dif_rng_seed)
            #z.repeat(batch_size,1,1,1)
        res_values = [[512,512],[H,W]]
        start_time_step = [self.args.start_sample_steps,256]
        multi_stage = False
        for res_step in range(2):
            sd = SimpleDiffusion(num_diffusion_timesteps=self.args.time_steps, device=self.device)
            if not multi_stage:
                init_img = img
                inpaint_mask = start_mask
                init_img = init_img*inpaint_mask
                ip_model = self.ema_m
                z = torch.randn_like(init_img[0:1,:,:,:])
                #z = torch.roll(z,[yOff,xOff],dims=[-2,-1]) # shifting the noise to match the shift in the background
                z.repeat(batch_size,1,1,1)
                if not self.args.inPaint:
                    x = z
                else:
                    #x, _gtn = sd.forward_diffusion(init_img,one_ts*(start_time_step[res_step]-1),noise2use=z)
                    x, _gtn = sd.forward_diffusion(init_img,one_ts*(start_time_step[res_step]-1),noise2use=z)            
            elif res_step == 0:
                init_img = F.interpolate(img,res_values[res_step],mode='bilinear',antialias=False)
                inpaint_mask = F.interpolate(start_mask,res_values[res_step],mode='bilinear',antialias=False)
                inpaint_mask = torch.floor(inpaint_mask)
                init_img = init_img*inpaint_mask
                ip_model = self.model
                z = torch.randn_like(init_img[0:1,:,:,:])
                #z = torch.roll(z,[yOff,xOff],dims=[-2,-1]) # shifting the noise to match the shift in the background
                z.repeat(batch_size,1,1,1)
                if not self.args.inPaint:
                    x = z
                else:
                    x, _gtn = sd.forward_diffusion(init_img,one_ts*(start_time_step[res_step]-1),noise2use=z)
            else:
                z = torch.randn_like(img[0:1,:,:,:])
                #z = torch.roll(z,[yOff,xOff],dims=[-2,-1]) # shifting the noise to match the shift in the background
                z.repeat(batch_size,1,1,1)
                x = F.interpolate(x,res_values[res_step],mode='bilinear',antialias=False)
                x, _gtn = sd.forward_diffusion(x,one_ts*(start_time_step[res_step]-1),noise2use=z)
                init_img = img
                inpaint_mask = start_mask
                ip_model = self.ema_m

            time_step = start_time_step[res_step] 
            show_index = 0
            if self.device.index == 0:
                pbar = tqdm(total=start_time_step[res_step]-2)
            sd_ver = 0
            repeat_count = 0
            variable_steps = True
            sd = SimpleDiffusion(num_diffusion_timesteps=1024, device=self.device)
            step_size = 1
            steps_back = 20
            repeat = 0
            min_t = 0
            while time_step > 1: # needs to be > 1 so that indexing to get betas works below
                #if time_step < self.args.start_art_reduce and res_step > 0:
                #    inpaint_mask = imgGpix
                # based on the time step you can change how much repetition you do to harmonize the
                # generated diffusion image with the surround image
                # step size is how many time steps you increment each time 1024/step_size is how many steps it would take
                # to generate a random image using diffusion
                # steps_back is the number of steps you go forward before going backward
                # repeat is how often you do that before subtracting 1 from steps back (that is decrementing the time_step value with step_size)
                # min_t is the min time that you allow the process to use (modifies steps_back)
                if variable_steps:
                    if time_step > 768: #range 1024->769
                        if sd_ver == 0:
                            #step_size = 8
                            step_size = 8
                            sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=self.device)
                            sd_ver = 1
                            #steps_back = 16 # first is 1000->500
                            steps_back = 8
                            repeat = 0
                            min_t = 0
                    elif time_step > 512:
                        if sd_ver <= 1:
                            #step_size = 4
                            step_size = 4
                            sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=self.device)
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
                            sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=self.device)
                            sd_ver = 3
                            steps_back = 16 # first is 150/5 = 20 
                            repeat = 0
                            min_t = 0
                    elif time_step > 128:
                        if sd_ver <= 3:
                            step_size = 4
                            sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=self.device)
                            sd_ver = 4
                            steps_back = 8 #100->50
                            repeat = 0
                            min_t = 0
                    elif time_step > 64:
                        if sd_ver <= 4:
                            step_size = 4
                            sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=self.device)
                            sd_ver = 5
                            steps_back = 4
                            repeat = 0
                            min_t = 0
                    elif time_step > 32:
                        if sd_ver <= 5:
                            step_size = 2
                            sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=self.device)
                            sd_ver = 5
                            steps_back = 2
                            repeat = 0
                            min_t = 0
                    else:
                        if sd_ver <= 6:
                            step_size = 1
                            steps_back = 1
                            sd = SimpleDiffusion(num_diffusion_timesteps=1024//step_size, device=self.device)
                            repeat = 0
                            sd_ver = 6

                adj_time_step = (time_step-min_t)//step_size # time value based on the step size
                #ts = one_ts * adj_time_step * step_size # the model restore has to be in the 0:999 range
                ts = one_ts * time_step # convert scaler to tensor because during training each image can have a different time_step
                
                num_steps = min(adj_time_step,steps_back)

                #if self.args.steps_back_range[0] >= time_step and self.args.steps_back_range[1] <= time_step:
                #    num_steps = min(adj_time_step,self.args.steps_back)
                #else:
                #    num_steps = min(adj_time_step,1)
                if not self.args.inPaint: # just generate a random image
                    z = torch.randn_like(x[0:1,:,:,:]) if adj_time_step > 1 else torch.zeros_like(x)
                    z.repeat(batch_size,1,1,1)
                    # one is subtracted from the time values below because of the indexing requirements
                    x = self.p_sample(sd,x,(ts-1),z,one_ts*(adj_time_step-1),ip_model)
                elif num_steps > 0:
                    for i in range(num_steps):
                        z = torch.randn_like(x[0:1,:,:,:]) if adj_time_step > 1 else torch.zeros_like(x)
                        # if you want to move the inpaint results to match the background, use torch.roll() here
                        # and after any randn call
                        z.repeat(batch_size,1,1,1)
                        adj_time = adj_time_step - i# 
                        ts = one_ts * (adj_time * step_size)
                        x = self.p_sample(sd,x,(ts-1),z,one_ts*(adj_time-1),ip_model)
                        #adj_time_step = adj_time*t_scale
                        if adj_time > 0:
                            # use same noise for all the steps
                            #kwnImg, gt_noise = self.sd.forward_diffusion(mask_img,one_ts*(adj_time-1))
                            #z = torch.randn_like(mask_img2use[0:1,:,:,:])
                            #z.repeat(batch_size,1,1,1)
                            kwnImg, _gtn = sd.forward_diffusion(init_img,one_ts*(adj_time-1),noise2use=z) ### do I really need to adj time???
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
                        # this portion adds back noise to go backward
                        #ts += t_scale # increment ts to next output position
                        #beta_t = get(self.sd.beta, one_ts*(adj_time-1))
                        beta_t = get(sd.beta, one_ts*(adj_time-1))
                        z = torch.randn_like(init_img[0:1,:,:,:])
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
                    x = self.p_sample(sd,x,ts,z,one_ts*0,ip_model)
                    #x = kwnImg*mask + x*(1-mask) # use the original mask because this is the last step
                    x = kwnImg*inpaint_mask + x*(1-inpaint_mask) # this is safer and can always apply original mask later
                if time_step <= self.args.show_time_steps[show_index]:
                    #write the images using the show_time_step in the file name
                    #fake_data = self.vae.decode(x.detach().clone() / 0.18215).sample
                    #fake_data = mImg[:,0:3,:,:]*img_mask + fake_data*(1-img_mask)
                    #temp = kwnImg*mask + x*(1-mask)
                    fake_data = torch.clip(invScaleImg(x),0,1)
                    outputs = fake_data.to('cpu')
                    outputs = bgr2rgb_batch(outputs)
                    inputs = bgr2rgb_batch(invScaleImg(init_img)).to('cpu')
                    refImages = bgr2rgb_batch(invScaleImg(ref_img)).to('cpu')
                    # reorganize the data to display: first row content, second,depth, third vpix
                    #RGBvPix = 1-torch.cat([inputStack[:,3:4,:,:],inputStack[:,3:4,:,:],inputStack[:,3:4,:,:]],1)
                    debugImg = torch.cat([inputs,outputs,refImages],0)
                    img_out = torchvision.utils.make_grid(debugImg,nrow=self.args.batch_size,normalize=False)
                    #hr_out = torchvision.utils.make_grid(img_inputs[hr_img])
                    imshow(img_out)
                    show_index += 1
                if repeat_count == 0:
                    time_step -= step_size
                    if self.device.index == 0:
                        pbar.update(step_size)
        if self.args.saveImg:
            x = torch.clip(x,-1.0,1.0)
            x = invScaleImg(x)
            #x = torch.pow(x,img_apl_exp)
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
        if self.device.index == 0:
            pbar.close()
        return None  

    def train(self):
        for epoch in range(self.args.n_epochs):
            torch.cuda.empty_cache()
            #gc.collect()
            
            # Algorithm 1: Training
            self._run_epoch(epoch=epoch)
            torch.distributed.barrier() # wait for all the processes to finish the epoch
            if self.device.index == 0:
                self._save_checkpoint(self.args.chk_save_pre,self.args.chk_save_ver)

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
    parser.add_argument('--location', default='home', help='location of GPU used: home, truecut overide path settings (default: home)')
    parser.add_argument('--max_wSize', default=4, type=int, help='max number of GPUs to use (default: 4)')
    parser.add_argument('--visableCUDA', default = '0,1,2,3,4,5,6,7', help='string with list of GPUs (eg 4, 5, 6, 7)')
    parser.add_argument('--n_epochs', default=500, type=int, help='Total epochs to train the model (default: 1)')
    parser.add_argument('--save_every', default=1024, type=int, help='How often to save a snapshot (default: 200)')
    parser.add_argument('--stats_every', default=64, type=int, help='How often to print statistics (default: 50)')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 1)')
    parser.add_argument('--num_batches', default=8, type=int, help='number of batches to accumulate before backward (default: 10)')
    parser.add_argument('--chk_save_pre', default='checkPoint/ddpm', help='version of checkpt to save(default: OFSRCNN14)')
    parser.add_argument('--chk_load_pre', default='checkPoint/ddpm', help='version of checkpt to save(default: OFSRCNN14)')
    parser.add_argument('--chk_save_ver', default='mdl1024_v8a', help='version of checkpt to save(default: v7sa)')
    parser.add_argument('--chk_load_ver', default='mdl512_v8a', help='version of checkpt to save(default: v7sa)')
    parser.add_argument('--chk_load_hd_ver', default='mdl1024_v8a', help='version of checkpt to load')
    parser.add_argument('--loadModel', default=False, type=bool, help='load CNN model (default=True))')
    parser.add_argument('--loadEMA', default=True, type=bool, help='load EMA CNN model (default=True))')
    parser.add_argument('--min_ema_decay', default=256, type=float, help='decay rate for ema model, default is equivalent of adding 1e-4')
    parser.add_argument('--max_ema_decay', default=8196, type=float, help='decay rate for ema model, default is equivalent of adding 1e-4')
    parser.add_argument('--imgLR', default=4e-5, type=float, help='MV learning rate (default: 5e-6)')
    parser.add_argument('--num_workers', default=2, type=int, help='number of dataloader processes(default: 0)')
    
    parser.add_argument('--time_steps', default=1024, type=int, help='number of diffusion time steps (default: 1024)')
    parser.add_argument('--dropout_p', default=.0625, type=float, help='number of random layers to drop (default: 0.0625)')
    parser.add_argument('--dif_rng_seed',default=0, type=int, help='seed to use for diffusion inference 0=>rand seed for all other(default: 0)')
    parser.add_argument('--dataset_name', default='flowers', help='version of checkpt to save(default: OFSRCNN14)')
    parser.add_argument('--dataset_root', default='dummy', help='path to dataset folder when location is not home or truecut')
    #parser.add_argument('--dataset_folder', default=["external/R5"],type=list, help='root folder for datasets (default: /Volumes/neil_SSD/datasets/)')
    #parser.add_argument('--mask_folder', default='inPaintMask/train/random_masks', help='folder for masks')
    #parser.add_argument('--dataset_folder', default=["RPresult_JA130/Right"],type=list, help='root folder for datasets (default: /Volumes/neil_SSD/datasets/)')
    #parser.add_argument('--mask_folder', default='RPresult_JA130/Right', help='folder for masks')
    #parser.add_argument('--dataset_folder', default=["aba_r5_3220/Source_Left/source"],type=list, help='root folder for dataset')
    #parser.add_argument('--mask_folder', default='aba_r5_3220/fg_r5_3220/mask_fg', help='folder for masks')
    #parser.add_argument('--dataset_folder', default=["imagenet-1k/train","raw_celeba/img_celeba","DIV2K/images","AIM500/original","woodall"],type=list, help='root folder for datasets (default: /Volumes/neil_SSD/datasets/)')
    parser.add_argument('--dataset_folder', default=["imagenet-1k/train","raw_celeba/img_celeba"],type=list, help='root folder for datasets (default: /Volumes/neil_SSD/datasets/)')
    parser.add_argument('--mask_folder', default=["inPaintMask/train/random_masks/"], help='folder for masks')
    parser.add_argument('--img_subfolder', default=None, help='sub folder for masks, None = search all sub folders')
    parser.add_argument('--mask_subfolder', default=None, help='folder for masks, None = search all sub folders')
    parser.add_argument('--sizeMult', default=64, type=int, help='pixel mulitple image needs to be)')
    parser.add_argument('--sizeLayers', default=6, type=int, help='log2 of sizeMult')
    #parser.add_argument('--imgSize', default=(512,1024), type=int, nargs=2, help='pixel mulitple image needs to be)')
    #parser.add_argument('--imgSize', default=(1024,1024), type=int, nargs=2, help='pixel mulitple image needs to be)')
    #parser.add_argument('--imgSize', default=(1152,2176), type=int, nargs=2, help='pixel mulitple image needs to be)')
    parser.add_argument('--imgSize', default=(512,512), type=int, nargs=2, help='pixel mulitple image needs to be)')
    #parser.add_argument('--imgSize', default=(1024,2048), type=int, nargs=2, help='pixel mulitple image needs to be)')
    #parser.add_argument('--imgSize', default=(512,1024), type=int, nargs=2, help='pixel mulitple image needs to be)')
    parser.add_argument('--imgScale', default=True, type=bool, help='optionally scale image before center crop')
    parser.add_argument('--imgIs16', default=False, type=bool, help='image is 16 bits/color (default=True)')
    parser.add_argument('--imgChnl', default=3, type=int, help='number of color channels to use (default=3)')

    parser.add_argument('--maskIsPV', default=False, type=bool, help='high value is pixel valid (default: False)')
    parser.add_argument('--maskIs16', default=False, type=bool, help='mask is 16 bit (default: False)')
    parser.add_argument('--chromaKey', default=None, help='set to None to disable otherwise 3 value tuple range 0:1')
    parser.add_argument('--getRandMask', default=True, type=bool, help='get a random mask and not masks paired with image')
    parser.add_argument('--makeMaskBinary', default=True, type=bool, help='only 100% valid pixels marked as valid pixels')
    parser.add_argument('--dilateHoleMult',default=1,type=int,help='number of times to do 3x3 max operation on hole')
    parser.add_argument('--inMaskG', default=512., type=float, help='mask gain value (default: 512)')
    #parser.add_argument('--maskOff', default=[0.375,0], type=float, nargs=2, help='mask offset only [0] is used(default: [0.375,0])')

    parser.add_argument('--showImg', default=True, type=bool, help='display a batch of images (default=False)')
    parser.add_argument('--inPaint', default=True, type=bool, help='do inpainting instead of normal diffusion (default=True)')
    #parser.add_argument('--steps_back', default = 25, type=int, help='number of steps forward in time when gen image')
    #parser.add_argument('--steps_back_range', default =[1024,1], nargs=2, type=int, help='range for going back and forth in time')
    
    parser.add_argument('--start_sample_steps', default =1024,type=int, help='starting time step for generating image')
    #parser.add_argument('--show_time_steps', default=[999,800,500,249,128,64,32,16,8,4,2,1,0], type=int, help='time steps to show progress')
    parser.add_argument('--show_time_steps', default=[1023,512,256,128,64,32,4,2,1,0], nargs='*', type=int, help='time steps to show progress, set to zero to skip')
    #parser.add_argument('--show_time_steps', default=[1023,4,2,1,0], nargs='*', type=int, help='time steps to show progress, set to zero to skip')
    #parser.add_argument('--show_time_steps', default=[0], nargs='*', type=int, help='time steps to show progress, set to zero to skip')
    parser.add_argument('--showImg_num', default=2, type=int, help='number of images to generate if showImg is true')

    parser.add_argument('--saveImg', default=False, type=bool, help='save a batch of images (default=False)')
    parser.add_argument('--saveAs8bit', default=False, type=bool, help='save as an 8 bit/color image, ignored for exr (default is true)')
    parser.add_argument('--saveImg_path', default='images', help='folder where images are save')
    parser.add_argument('--saveExt', default='tif', help='type of image to save (default: jpg)')
    parser.add_argument('--saveInput_folder', default=None, help='subfolder for input image (defult=Input)')
    parser.add_argument('--saveOutput_folder', default='Output', help='subfolder for output image (defult=Output)')
    parser.add_argument('--saveRef_folder', default=None, help='subfolder for reference image (defult=Reference)')

    args = parser.parse_args()

    if args.location == 'home':
        args.chk_save_pre='checkpoints/base/ddpm'
        args.chk_load_pre='checkpoints/base/ddpm'
        args.dataset_root='/Volumes/neil_SSD/datasets/'
        args.saveImg_path = 'images'
    elif args.location =='truecut':
        args.chk_save_pre='/tt/ssd0/checkPoint/ddpm'
        args.chk_load_pre='/tt/ssd0/checkPoint/ddpm'
        args.dataset_root='/tt/ssd0/datasets/'
        args.saveImg_path = '/tt/ssd0/neil/images'

    if args.showImg:
        args.max_wSize = 1
        args.batch_size = args.showImg_num
    else:
        if args.saveImg:
            args.show_time_steps = [0]      
        #args.time_steps = args.num_sample_steps

    if args.showImg or args.saveImg:
        args.n_epochs = 1

    if device.type == 'cuda':
    #if device.is_cuda():
        world_size = min(args.max_wSize,torch.cuda.device_count())
        if args.visableCUDA is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.visableCUDA
    else:
        world_size = 1
    #train_model(CHECK_PT_SAVE,NUM_EPOCHS,B_SZ)
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

    #inputDebug()

    mp.spawn(train_model, args=(world_size, args, device), nprocs=world_size)
