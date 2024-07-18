import numpy as np
import torch
import torch.utils.data as data
import random
import os.path as osp

from torch.utils.data.dataset import ConcatDataset, Dataset
from utils.augmentor_multiframes import FlowAugmentor
import pickle
import sys,os
import cv2
dir_mytest = os.path.dirname(os.path.abspath(__file__))+'/../../../algo/'
sys.path.insert(0, dir_mytest)
from file_utils import read
from myutil import mask_adjust,img_equal,motion_blur
# from myutil import mask_processor
class FlowDatasetKousei(data.Dataset):
    def __init__(self, data_root,aug_params=None, sparse=False, input_frames=5,norm=False,reverse=False,OPENEXR=True):
        self.data_root = data_root
        self.augmentor = None
        self.sparse = sparse
        self.input_frames = input_frames
        max_mv_x = aug_params['max_mv_x'] if 'max_mv_x' in aug_params else 0.5
        max_mv_y = aug_params['max_mv_y'] if 'max_mv_y' in aug_params else 0.5
        self.motion_blur_rate = aug_params['motion_blur_rate'] if 'motion_blur_rate' in aug_params else 0
        self.film_grain = aug_params['film_grain']
        self.max_mv_x=max_mv_x
        self.max_mv_y=max_mv_y
        self.img_type = 'image' if aug_params['img_type'].lower()=='image' else 'ldr' #hdr or image(ldr)
        self.reverse_rate = aug_params['reverse_rate']
        self.reverse = reverse
        self.norm = norm
        if aug_params is not None:
            self.augmentor = FlowAugmentor(aug_params)
        self.init_seed = False
        self.image_list = None
        self.mv0_list = None
        self.mv1_list = None
        self.mask_list = None
        self.mask_enhance_rate = aug_params['mask_enhance_rate']
        self.mask_enhance_range = aug_params['mask_enhance_range']
        #ldr or hdr
        self.image_range = [-2,2] if self.img_type != 'image' else [0,1]
        #mixed type
        if aug_params['mask_type'] == 'mix':
            if np.random.rand() < 0.5:
                self.mask_type = 'fg'
            else:
                self.mask_type = 'bg'
        else:
            self.mask_type = aug_params['mask_type'] if aug_params['mask_type'].lower()!='frame' else None
        self.kata_mask_mode = aug_params['kata_mask_mode'] if 'kata_mask_mode' in list(aug_params.keys()) else False
        self.OPENEXR = OPENEXR
        self.repeat_frame_rate = aug_params['repeat_frame_rate']
        self.mask_treshold = aug_params['mask_treshold']

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                #print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        imgs = [read(self.data_root+path,type=self.img_type,OPENEXR=self.OPENEXR) for path in self.image_list[index]]
        mv0s = [read(self.data_root+path,type='flo',OPENEXR=self.OPENEXR)[...,:2] if path is not None else None for path in self.mv0_list[index]]
        mv1s = [read(self.data_root+path,type='flo',OPENEXR=self.OPENEXR)[...,:2] if path is not None else None for path in self.mv1_list[index]]
        masks = [(255*read(self.data_root+path,type='mask',OPENEXR=self.OPENEXR)).astype('uint8') if (path is not None and self.mask_type is not None) else None for path in self.mask_list[index]]
        h,w,_ = imgs[0].shape
        if np.random.rand() < self.repeat_frame_rate: #add repeat frame case
            if self.input_frames >3:
                # pass invalid frames
                if self.image_list[0]!= self.image_list[1] and self.image_list[-1]!= self.image_list[-2]:
                    start_pos = np.random.randint(1,len(imgs)-1)    
                    if np.random.rand()<0.5: #go left
                        for i in range(start_pos):
                            imgs[i] = imgs[start_pos]
                            masks[i] = masks[start_pos]
                    else: # go right
                        for i in range(start_pos+1,len(imgs)):
                            imgs[i] = imgs[start_pos]
                            masks[i] = masks[start_pos]
            else:
                #mock disparity input
                if np.random.rand()<0.5: #go left
                    imgs[0] = imgs[2].copy()
                    masks[0] = masks[2].copy() if masks[2] is not None else None
                    mv1s[0] = mv0s[0].copy() if mv0s[0] is not None else None
                else:
                    imgs[2] = imgs[0].copy()
                    masks[2] = masks[0].copy() if masks[0] is not None else None
                    mv0s[0] = mv1s[0].copy() if mv1s[0] is not None else None
                    

        #mv refine
        for i in range(1,len(imgs)-1):
            if img_equal(imgs[i-1],imgs[i]):
                if mv1s[i-1] is not None:
                    mv1s[i-1] = np.zeros((imgs[i].shape[0],imgs[i].shape[1],2)).astype('float32')
            if img_equal(imgs[i],imgs[i+1]):
                if mv0s[i-1] is not None:
                    mv0s[i-1] = np.zeros((imgs[i].shape[0],imgs[i].shape[1],2)).astype('float32')
        #UE4 denormalized
        if self.norm:
            for i in range(len(mv0s)):
                mv0 = mv0s[i]
                mv1 = mv1s[i]
                if mv0 is not None:
                    h,w,_ = mv0.shape
                    mv0[...,0] *= w
                    mv0[...,1] *= h
                    if self.reverse:
                        mv0[...,1] *= -1
                    mv0s[i] = mv0
                if mv1 is not None:
                    h,w,_ = mv1.shape
                    mv1[...,0] *= w
                    mv1[...,1] *= h
                    if self.reverse:
                        mv1[...,0] *= -1
                    mv1s[i] = mv1
        #mask processor
        if all([mask.max()>0 if mask is not None else None for mask in masks]):#prune diff mask
            mask_ = [np.zeros_like(tmp) for tmp in masks]
            char = sorted(list(filter(lambda x:x>self.mask_treshold,np.unique(masks[0]))))[::-1]
            num_choice = len(char)
            pick = np.random.randint(1,num_choice+1) if num_choice>1 else 1
            for i in range(pick):
                for ii in range(len(mask_)):
                    mask_[ii][np.where(masks[ii] == char[i])] = 1
            masks = mask_
        else: # not using mask inf
            masks = [np.ones_like(imgs[0])[...,-1]]*len(imgs)
        
        #change to bg mask
        if self.mask_type == 'bg': 
            for i in range(len(masks)):
                masks[i] += 1
                masks[i][np.where(masks[i]>1)] = 0

        #add mask and valid
        if self.mask_type is not None:  
            for i in range(len(imgs)):
                if masks[i] is None:
                    mask = np.ones_like(imgs[i])[...,0]
                else:
                    mask = masks[i]
                if self.kata_mask_mode and i != len(imgs)//2:
                    mask = np.ones_like(imgs[i])[...,0]
                if len(mask.shape)==2:
                    mask = mask[...,None]
                else:
                    mask = mask[...,-1:]
                imgs[i] = np.concatenate((imgs[i],mask),axis=2)

        #get sparse mv's valid
        if self.sparse: 
            for i in range(len(mv0s)):
                if mv0s[i] is not None: 
                    valid = read(self.data_root+self.mv0_list[index][i],type='mask')
                    mv0s[i] = np.concatenate((mv0s[i],valid[...,None]),axis=2)
            for i in range(len(mv1s)):
                if mv1s[i] is not None: 
                    valid = read(self.data_root+self.mv1_list[index][i],type='mask')
                    mv1s[i] = np.concatenate((mv1s[i],valid[...,None]),axis=2)

        flows = mv0s+mv1s
        # grayscale images
        # if len(imgs[0].shape) == 2:
        #     imgs = [np.tile(img[...,None], (1, 1, 3)) for img in imgs]
        # else:
        #     imgs = [img for img in imgs]
            
        # data augment
        if self.augmentor is not None:
            imgs, flows = self.augmentor(imgs, flows)
        
        # img bn
        tt = 1
        if self.img_type.lower() == 'image':
            tt = 255
        imgs_ = [2 * (img.astype('float32') / tt ) - 1.0 for img in imgs]
        if self.mask_type is not None:
            for i in range(len(imgs_)):
                imgs_[i] = np.concatenate((imgs_[i],imgs[i][...,3:]),axis=-1) 
        imgs = imgs_
        
        # film grain
        film_grain = int(self.film_grain[1])
        if film_grain>0:
            tmp = 1/255 if self.img_type == 'image' else 1
            imgs = [(image + tmp*np.random.uniform(*film_grain) * np.random.randn(*image.shape).astype('float32')).clip(*self.image_range) for image in imgs]
        # motion blur
        motion_blur_rate = float(self.motion_blur_rate)
        if motion_blur_rate > 0: 
            if np.random.rand() < motion_blur_rate:
                ksize = np.random.randint(5,20)
                imgs = [motion_blur(img,ksize) if img is not None else None for img in imgs]
        
        h,w,_ = imgs[0].shape
        valids = [np.ones((h,w))] * len(flows)
        
        # retrive valids from augmented datas
        if self.sparse:
            for i in range(len(flows)):
                if flows[i] is not None:
                    valids[i] = flows[i][...,-1] 
                    valids[i][np.where(valids[i]!=1)] = 0
                    flows[i] = flows[i][...,:2]
                else:
                    valids[i] = np.zeros((h,w))

        #retrive mask and valid from augmented datas
        if self.mask_type is not None: #go prune
            for i in range(len(imgs)):
                masks[i] = np.copy(imgs[i][...,-1])
                imgs[i] = imgs[i][...,:3]
        
        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        flows = [torch.from_numpy(flow).permute(2, 0, 1).float() if flow is not None else None for flow in flows]
        valids = [torch.from_numpy(valid).float() for valid in valids]
        masks = [torch.from_numpy(mask).float() if mask is not None else None for mask in masks]
        
        # prune exceed mvs
        for i,flow in enumerate(flows):
            
            if flows[i] is None:
                flows[i] = torch.zeros(2,h,w)
                valids[i] = torch.zeros(h,w)
            else:
                valids[i] *= ((flow[0].abs() < self.max_mv_x*w) & (flow[1].abs() < self.max_mv_y*h)).float()

        # prune masks from valids
        if self.mask_type is not None: #go prune
            for i in range(len(imgs)):
                mask = masks[i]
                if mask is not None:
                    if i > 0 and i < len(imgs)-1:
                        valids[i-1]*=mask
                        valids[i-1+self.input_frames-2]*=mask
                    if np.random.rand() < self.mask_enhance_rate:#mask enhance
                        mask = mask_adjust(mask,size=np.random.randint(self.mask_enhance_range[0],self.mask_enhance_range[1]+1))
                    # imgs[i]*=mask[None,...]
                    if mask.max().item() > 0: #no obj case
                        for k in range(3):
                            imgs[i][k,...][torch.where(mask==0)] = -100                          

        if np.random.rand() < self.reverse_rate:
            return torch.stack(imgs[::-1]), torch.stack(flows[::-1]), torch.stack(valids[::-1])
        else:
            return torch.stack(imgs), torch.stack(flows), torch.stack(valids)

    def __rmul__(self, v):
        self.image_list = v * self.image_list
        self.mv0_list = v * self.mv0_list
        self.mv1_list = v * self.mv1_list
        self.mask_list = v * self.mask_list
        return self
        
    def __len__(self):
        return len(self.image_list)

class FlyingThings3D(FlowDatasetKousei):
    def __init__(self,dataroot,pklroot,aug_params=None, input_frames=5, dstype='clean'):
        super(FlyingThings3D, self).__init__(osp.join(dataroot,'flyingthings'),aug_params=aug_params, input_frames=input_frames,sparse=False,norm=False)
        root = osp.join(pklroot,'flyingthings')
        dstype = 'frames_cleanpass' if dstype =='clean' else 'frames_finalpass'
        with open(f'{root}/{dstype}.pkl', "rb") as f:
            pkldatas = pickle.load(f)
        image_list,mv0_list,mv1_list,mask_list  = pkldatas['image_list'],pkldatas['mv0_list'],pkldatas['mv1_list'],pkldatas['mask_list']
        self.image_list,self.mv0_list,self.mv1_list,self.mask_list = image_list,mv0_list,mv1_list,mask_list
        
class MpiSintel(FlowDatasetKousei):
    def __init__(self,dataroot,pklroot,aug_params=None, input_frames=5, dstype='clean'):
        super(MpiSintel, self).__init__(osp.join(dataroot,'Sintel'),aug_params=aug_params, input_frames=input_frames,sparse=False,norm=False)
        root = osp.join(pklroot,'Sintel')
        with open(f'{root}/{dstype}.pkl', "rb") as f:
                pkldatas = pickle.load(f)
        image_list,mv0_list,mv1_list  = pkldatas['image_list'],pkldatas['mv0_list'],pkldatas['mv1_list']
        self.image_list,self.mv0_list,self.mv1_list = image_list,mv0_list,mv1_list
        self.mask_list = [[None]*input_frames for _ in range(len(self.image_list))]

class Spring(FlowDatasetKousei):
    def __init__(self,dataroot,pklroot,aug_params=None, input_frames=5, dstype='clean'):
        super(Spring, self).__init__(osp.join(dataroot,'Spring'),aug_params=aug_params, input_frames=input_frames,sparse=True,norm=True)
        root = osp.join(pklroot,'Spring')
        with open(f'{root}/train.pkl', "rb") as f:
            pkldatas = pickle.load(f)
        image_list,mv0_list,mv1_list = pkldatas['image_list'],pkldatas['mv0_list'],pkldatas['mv1_list']
        self.image_list,self.mv0_list,self.mv1_list = image_list,mv0_list,mv1_list
        self.mask_list = [[None]*input_frames for _ in range(len(self.image_list))]
        
class Unreal(FlowDatasetKousei):
     def __init__(self,dataroot,pklroot, aug_params=None, input_frames=5, split='train',dstype='clean',objtype='large_obj',fps=24):
        super(Unreal, self).__init__(osp.join(dataroot,'Unreal'),aug_params=aug_params, input_frames=input_frames,sparse=False,norm=True)
        root = osp.join(pklroot,'Unreal')
        pkl_file = f'{root}/{split}_{dstype}_{fps}fps.pkl'
        if os.path.isfile(pkl_file):
            with open(pkl_file, "rb") as f:
                    pkldatas = pickle.load(f)
            image_list,mv0_list,mv1_list,mask_list  = pkldatas['image_list'],pkldatas['mv0_list'],pkldatas['mv1_list'],pkldatas['mask_list']
        else:
            image_list,mv0_list,mv1_list,mask_list = [],[],[],[]
        self.image_list,self.mv0_list,self.mv1_list,self.mask_list = image_list,mv0_list,mv1_list,mask_list

class Unreal_MRQ(FlowDatasetKousei):
     def __init__(self,dataroot,pklroot, aug_params=None, input_frames=5, split='train',dstype='clean',objtype='large_obj',fps=24):
        super(Unreal_MRQ, self).__init__(osp.join(dataroot,'Unreal_MRQ'),aug_params=aug_params, input_frames=input_frames,sparse=False,norm=True)
        root = osp.join(pklroot,'Unreal_MRQ')
        pkl_file = f'{root}/{split}_{dstype}_{fps}fps.pkl'
        if os.path.isfile(pkl_file):
            with open(pkl_file, "rb") as f:
                    pkldatas = pickle.load(f)
            image_list,mv0_list,mv1_list,mask_list  = pkldatas['image_list'],pkldatas['mv0_list'],pkldatas['mv1_list'],pkldatas['mask_list']
        else:
            image_list,mv0_list,mv1_list,mask_list = [],[],[],[]
        self.image_list,self.mv0_list,self.mv1_list,self.mask_list = image_list,mv0_list,mv1_list,mask_list

def stage_add(stages,stage):
    if stages is None:
        return stage
    return stages + stage

train_stage = {
    'sintel_clean':MpiSintel,
    'sintel_final':MpiSintel,
    'unreal_clean':Unreal,
    'unreal_final':Unreal,
    'spring_clean':Spring,
    'things_clean':FlyingThings3D,
    'things_final':FlyingThings3D,
    'unreal_MRQ_clean':Unreal_MRQ,
    'unreal_MRQ_final':Unreal_MRQ,

}
def fetch_dataloader(args):
    train_dataset = None
    dataset_len = '[datas len: ]'
    for stage in args.kousei:
        if 'stage_' not in stage:
            continue
        cur_stage = []
        aug_params = dict(args.kousei)
        stage_name = stage.replace('stage_','')
        nums,aug_params['min_scale'],aug_params['max_scale'] = aug_params[stage] 
        nums = int(nums)
        if nums <=0:
            continue
        dstype = 'final' if '_final' in stage else 'clean'
        if 'unreal' in stage_name and '_MRQ' not in stage_name:
            for fps in args.fps:
                tmp = train_stage[stage_name](args.data_root,args.pkl_root,aug_params,dstype=dstype, input_frames=args.input_frames,fps=int(fps))
                if len(tmp) == 0:
                    continue
                cur_stage = nums * tmp
                if len(cur_stage)> 0:
                     train_dataset = stage_add(train_dataset,cur_stage)
        else:
            cur_stage = nums * train_stage[stage_name](args.data_root,args.pkl_root,aug_params,dstype=dstype, input_frames=args.input_frames)
            train_dataset = stage_add(train_dataset,cur_stage)
        dataset_len+=f'{stage_name}:{len(cur_stage)},'
    assert train_dataset is not None,'no training datas!'
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=args.batch_size*2, drop_last=True)
    print('Training with %d image pairs' % len(train_dataset))
    print(dataset_len)
    return train_loader

if __name__ == "__main__":
    from mytrain import get_cfg
    args = get_cfg()

