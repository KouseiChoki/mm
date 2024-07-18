'''
Author: Qing Hong
Date: 2023-08-10 15:13:47
LastEditors: Qing Hong
LastEditTime: 2024-04-28 13:20:42
Description: file content
'''
import pickle
import os.path as osp
from glob import glob
import os,shutil
import numpy as np

import sys

if sys.platform.startswith('win'):
    sl = '\\'
else:
    sl = '/'
def pickle_write(path,data):
    with open(path, 'wb') as fh:
        pickle.dump(data, fh)

def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))


def sliding_window(sequence, window_size,window_step,pad=0,pad_step=0):
    """Generate a sliding window over a sequence."""
    window_size -=2
    res = []
    for i in range(0, len(sequence), window_step):
        window = sequence[i:i+window_size]
        if len(window) < window_size:
            window = sequence[-window_size:]
        if pad>0:
            for _ in range(pad):
                window = np.hstack((window[0]-pad_step,window,window[-1]+pad_step))
        res.append(window)
    return res

def data_process(images,mv0s,mv1s,masks,input_frames=5,step=1,multi=True):
    res_images,res_mv0s,res_mv1s,res_masks = [],[],[],[]
    task_indexes = np.arange(0,len(images),step)
    # slide_indexes = sliding_window(task_indexes,frams,frams-2,1,step) if multi else sliding_window(task_indexes,3,1,frams//2,step)
    frams = np.clip(input_frames,3,50)
    slide_indexes = sliding_window(task_indexes,3,1,(frams-1)//2,step) 
    if input_frames==2:
        slide_indexes = slide_indexes[1:-1]
    assert len(slide_indexes)>=1,'data error!'
    for indexes in slide_indexes:
        res_images_,res_mv0s_,res_mv1s_,res_masks_ = [],[],[],[]
        for i in range(len(indexes)):
            position_valid = indexes[i] >= 0 and indexes[i]<= len(images)-1
            valid_mv1,valid_mv0 = True,True
            if not position_valid:
                valid_mv1,valid_mv0 = False,False
            index = np.clip(indexes[i],0,len(images)-1)
            res_images_.append(images[index])
            if masks is not None:
                res_masks_.append(masks[index])
            if i!=0 and i!=len(indexes)-1:
                if valid_mv0:
                    res_mv0s_.append(mv0s[index])
                else:
                    res_mv0s_.append(None)
                if valid_mv1:
                    res_mv1s_.append(mv1s[index])
                else:
                    res_mv1s_.append(None)
        res_images.append(res_images_),res_mv0s.append(res_mv0s_),res_mv1s.append(res_mv1s_),res_masks.append(res_masks_)
        if masks is None:res_masks=[None]*len(res_images)
    return res_images,res_mv0s,res_mv1s,res_masks

    
        
def prepare_things(root,sp,input_frames,multi_mv):
    if not os.path.isdir(root):
        return
    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        image_lists = []
        mv0_lists = []
        mv1_lists = []
        mask_lists = []
        for cam in ['left','right']:
            image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN{sl}*{sl}*')))
            image_dirs = sorted([osp.join(f, cam) for f in image_dirs])
            flow_dirs = sorted(glob(osp.join(root, 'optical_flow{sl}TRAIN{sl}*{sl}*')))
            mask_dirs  = sorted(glob(osp.join(root, 'object_index{sl}TRAIN{sl}*{sl}*')))
            mask_dirs = sorted([osp.join(f, cam) for f in mask_dirs])
            flow_future_dirs = sorted([osp.join(f, 'into_future', cam) for f in flow_dirs])
            flow_past_dirs = sorted([osp.join(f, 'into_past', cam) for f in flow_dirs])
            for idir, fdir, pdir,mdir in zip(image_dirs, flow_future_dirs, flow_past_dirs,mask_dirs):
                images = sorted(glob(osp.join(idir, '*.png')) )
                future_flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                past_flows = sorted(glob(osp.join(pdir, '*.pfm')) )
                masks = sorted(glob(osp.join(mdir, '*.pfm')) )
                image_list= []
                mv0_list = []
                mv1_list = []
                mask_list = []
                for i in range(len(images)):
                    image_list.append(images[i].replace(root,''))
                    mask_list.append(masks[i].replace(root,''))
                    if i == 0:
                        mv1_list.append(None)
                    else:
                        mv1_list.append(past_flows[i].replace(root,''))
                    if i != len(images)-1:
                        mv0_list.append(future_flows[i].replace(root,''))
                    else:
                        mv0_list.append(None)
                image_list,mv0_list,mv1_list,mask_list = data_process(image_list,mv0_list,mv1_list,mask_list,input_frames,multi_mv)
                image_lists+=image_list
                mv0_lists+=mv0_list
                mv1_lists+=mv1_list
                mask_lists+=mask_list
        data = {'image_list':image_lists,'mv0_list':mv0_lists,'mv1_list':mv1_lists,'mask_list':mask_lists}
        mkdir(sp)
        pickle_write(osp.join(sp,'{}.pkl'.format(dstype)),data)
    
        
def prepare_sintel(root,sp,input_frames,multi_mv):
    if not os.path.isdir(root):
        return
    for dstype in ['clean', 'final']:
        image_lists = []
        mv0_lists = []
        mv1_lists = []
        flow_root = osp.join(root, 'training', 'flow')
        image_root = osp.join(root, 'training', dstype)

        for scene in os.listdir(image_root):
            images = sorted(glob(osp.join(image_root, scene, '*.png')))
            mv0s = sorted(glob(osp.join(flow_root, scene, '*.flo')))
            image_list = []
            mv0_list = []
            mv1_list = []
            for i in range(len(images)):
                image_list.append(images[i].replace(root,''))
                if i != len(images)-1:
                    mv0_list.append(mv0s[i].replace(root,''))
                else:
                    mv0_list.append(None)
                mv1_list.append(None)
            image_list,mv0_list,mv1_list,_ = data_process(image_list,mv0_list,mv1_list,None,input_frames,multi_mv)
            image_lists+=image_list
            mv0_lists+=mv0_list
            mv1_lists+=mv1_list
        data = {'image_list':image_lists,'mv0_list':mv0_lists,'mv1_list':mv1_lists}
        mkdir(sp)
        pickle_write(osp.join(sp,'{}.pkl'.format(dstype)),data)

def clean_source(source):
    res = set()
    for s in source:
        num = s.split('{sl}')[-1]
        if num.lower() not in ['image','mask','mv0','mv1','obj','video']:
            res.add(s)
        else:
            # print(s[:s.find(num)-1])
            res.add(s[:s.find(num)-1])
    return list(res)

# def find_folders_with_subfolder(root_path, subfolder_name,subfolder_name2):
#     """
#     Find all folders in the root_path that contain a subfolder with the name subfolder_name.
#     """
#     folders_with_subfolder = []

#     # Walk through the directory
#     for dirpath, dirnames, filenames in os.walk(root_path):
#         # Check if the subfolder_name is in the list of directories
#         if subfolder_name in dirnames and subfolder_name2 in dirpath:
#             folders_with_subfolder.append(dirpath)

#     return folders_with_subfolder


def find_folders_with_subfolder(root_path, keys = [], path_keys = [] ,excs = [] ,path_excs =[]):
    """
    Find all folders in the root_path that contain a subfolder with the name subfolder_name.
    """
    folders_with_subfolder = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Check if the subfolder_name is in the list of directories
        flag = True
        for key in keys:
            if key not in dirnames:
                flag = False
        for path_key in path_keys:
            if path_key not in dirpath:
                flag = False
        for exc in excs:
            if exc in dirnames:
                flag = False
        for exc in path_excs:
            if exc in dirpath:
                flag = False
        if flag:
            folders_with_subfolder.append(dirpath)

    return folders_with_subfolder

def prepare_Unreal(root,sp,input_frames,multi_mv,ldr=True,skip_frames=0,ctype='image'):
    if not os.path.isdir(root):
        return
    mkdir(sp)
    # ctype = 'image' if ldr else 'video'
    ctype_ = '.png' if ctype =='image' else '.exr'
    for split in ['train','test']:
        for baja in ['clean','final']:
            for fps in [12,24,48]:
                image_lists = []
                mv0_lists = []
                mv1_lists = []
                mask_lists = []
                fpstype = f'{fps}fps'
                source_root = find_folders_with_subfolder(root,keys=[ctype],path_keys=[fpstype,f'{sl}{split}{sl}',f'{sl}{baja}{sl}'],path_excs=['{sl}bbox','{sl}obj'])
                if fps == 24:
                    source_root += find_folders_with_subfolder(root,keys=[ctype],path_keys=[f'{sl}{split}{sl}',f'{sl}{baja}{sl}'],path_excs=['{sl}bbox','fpstype','{sl}obj'])
                # source_root = find_folders_with_subfolder(f'{root}{sl}{split}{sl}{baja}',ctype,f'{sl}{fpstype}')
                # source_root = sorted(glob(osp.join(root, f'{split}{sl}{baja}{sl}*{sl}{fpstype}{sl}*'))+glob(osp.join(root, f'{split}{sl}{baja}{sl}*{sl}{fps}{sl}*')))
                # source_root = clean_source(source_root)
                hdr_dirs = [osp.join(f, ctype) for f in source_root]
                flow_dirs_mv0 = [osp.join(f, 'mv0') for f in source_root]
                flow_dirs_mv1 = [osp.join(f, 'mv1') for f in source_root]
                mdir_dirs = [osp.join(f, 'Mask') for f in source_root]
                if baja == 'final': #final data use clean mv0,mv1
                    flow_dirs_mv0 = [f.replace('{sl}final','{sl}clean') for f in flow_dirs_mv0]
                    flow_dirs_mv1 = [f.replace('{sl}final','{sl}clean')for f in flow_dirs_mv1]
                    mdir_dirs = [f.replace('{sl}final','{sl}clean') for f in mdir_dirs]
                for idir, fdir0,fdir1,mdir in zip(hdr_dirs, flow_dirs_mv0,flow_dirs_mv1,mdir_dirs):
                    images = sorted(glob(osp.join(idir, f'*{ctype_}')) )
                    if (len(images))<=input_frames:
                        continue
                    mv0s = sorted(glob(osp.join(fdir0, '*.exr')) )
                    mv1s = sorted(glob(osp.join(fdir1, '*.exr')) )
                    if len(mv1s) ==0 and len(mv0s) ==0:
                        continue
                    masks = sorted(glob(osp.join(mdir, '*.exr')) )
                    image_list = []
                    mv0_list = []
                    mv1_list = []
                    mask_list = [] if len(masks)>0 else None
                    for i in range(skip_frames,len(images)-skip_frames):#mask少一张 所以要剔除最后一张,  1208添加： 第一张图片质量有问题，也剔除
                        image_list.append(images[i].replace(root,''))
                        try:
                            if mask_list is not None:
                                mask_list.append(masks[i].replace(root,''))
                            if i == 1:
                                mv1_list.append(None)
                            else:
                                mv1_list.append(mv1s[i].replace(root,''))
                            if i != len(images)-2:
                                if len(mv0s) ==0:
                                    mv0_list.append(None)
                                else:
                                    mv0_list.append(mv0s[i].replace(root,''))
                            else:
                                mv0_list.append(None)
                        except:
                            print(len(images),len(masks),idir)
                    image_list,mv0_list,mv1_list,mask_list = data_process(image_list,mv0_list,mv1_list,mask_list,input_frames,multi_mv)
                    image_lists+=image_list
                    mv0_lists+=mv0_list
                    mv1_lists+=mv1_list
                    mask_lists+=mask_list
                data = {'image_list':image_lists,'mv0_list':mv0_lists,'mv1_list':mv1_lists,'mask_list':mask_lists}
                pickle_write(osp.join(sp,f'{split}_{baja}_{fpstype}.pkl'),data)


def prepare_UnrealMRQ(root,sp,input_frames,multi_mv,skip_frames,ldr=True):
    if not os.path.isdir(root):
        return
    mkdir(sp)
    ctype = 'image' if ldr else 'video'
    ctype_ = '.png' if ldr else '.exr'
    for split in ['train','test']:
        for baja in ['clean','final']:
            image_lists = []
            mv0_lists = []
            mv1_lists = []
            mask_lists = []
            fpstype = '24fps'
            source_root = find_folders_with_subfolder(f'{root}{sl}{split}{sl}{baja}',ctype,'')
            hdr_dirs = [osp.join(f, ctype) for f in source_root]
            flow_dirs_mv0 = [osp.join(f, 'mv0') for f in source_root]
            flow_dirs_mv1 = [osp.join(f, 'mv1') for f in source_root]
            mdir_dirs = [osp.join(f, 'Mask') for f in source_root]
            if baja == 'final': #final data use clean mv0,mv1
                flow_dirs_mv0 = [f.replace('{sl}final','{sl}clean') for f in flow_dirs_mv0]
                flow_dirs_mv1 = [f.replace('{sl}final','{sl}clean')for f in flow_dirs_mv1]
                mdir_dirs = [f.replace('{sl}final','{sl}clean') for f in mdir_dirs]
            for idir, fdir0,fdir1,mdir in zip(hdr_dirs, flow_dirs_mv0,flow_dirs_mv1,mdir_dirs):
                images = sorted(glob(osp.join(idir, f'*{ctype_}')) )
                if (len(images))<=input_frames:
                    continue
                mv0s = sorted(glob(osp.join(fdir0, '*.exr')) )
                mv1s = sorted(glob(osp.join(fdir1, '*.exr')) )
                masks = sorted(glob(osp.join(mdir, '*.exr')) )
                image_list = []
                mv0_list = []
                mv1_list = []
                mask_list = [] if len(masks)>0 else None
                for i in range(skip_frames,len(images)-skip_frames):
                    image_list.append(images[i].replace(root,''))
                    try:
                        if mask_list is not None:
                            mask_list.append(masks[i].replace(root,''))
                        if i == 1:
                            mv1_list.append(None)
                        else:
                            mv1_list.append(mv1s[i].replace(root,''))
                        if i != len(images)-2:
                            if len(mv0s) ==0:
                                mv0_list.append(None)
                            else:
                                mv0_list.append(mv0s[i].replace(root,''))
                        else:
                            mv0_list.append(None)
                    except:
                        print('error:{}'.format(images[0]),len(images),len(masks))
                image_list,mv0_list,mv1_list,mask_list = data_process(image_list,mv0_list,mv1_list,mask_list,input_frames,multi_mv)
                image_lists+=image_list
                mv0_lists+=mv0_list
                mv1_lists+=mv1_list
                mask_lists+=mask_list
            data = {'image_list':image_lists,'mv0_list':mv0_lists,'mv1_list':mv1_lists,'mask_list':mask_lists}
            pickle_write(osp.join(sp,f'{split}_{baja}_{fpstype}.pkl'),data)




def prepare_Spring(root,sp,input_frames,multi_mv):
    if not os.path.isdir(root):
        return
    for split in ['train']:
        image_lists = []
        mv0_lists = []
        mv1_lists = []
        for cam in ['left','right']:
            source_root = sorted(glob(osp.join(root, f'{split}{sl}*')))
            hdr_dirs = sorted([osp.join(f, f'{cam}{sl}image') for f in source_root])
            flow_dirs_mv0 = sorted([osp.join(f, f'{cam}{sl}mv0') for f in source_root])
            flow_dirs_mv1 = sorted([osp.join(f, f'{cam}{sl}mv1') for f in source_root])
            for idir, fdir0,fdir1 in zip(hdr_dirs, flow_dirs_mv0,flow_dirs_mv1):
                images = sorted(glob(osp.join(idir, '*.png')) )
                mv0s = sorted(glob(osp.join(fdir0, '*.exr')) )
                mv1s = sorted(glob(osp.join(fdir1, '*.exr')) )
                image_list = []
                mv0_list = []
                mv1_list = []
                for i in range(len(images)):
                    image_list.append(images[i].replace(root,''))
                    if i == 0:
                        mv1_list.append(None)
                    else:
                        mv1_list.append(mv1s[i-1].replace(root,''))
                    if i != len(images)-1:
                        mv0_list.append(mv0s[i].replace(root,''))
                    else:
                        mv0_list.append(None)
                image_list,mv0_list,mv1_list,_ = data_process(image_list,mv0_list,mv1_list,None,input_frames,multi_mv)
                image_lists+=image_list
                mv0_lists+=mv0_list
                mv1_lists+=mv1_list
        mkdir(sp)
        data = {'image_list':image_lists,'mv0_list':mv0_lists,'mv1_list':mv1_lists}
        pickle_write(osp.join(sp,f'{split}.pkl'),data)


def train_data_prepare(root,sp,datasets,input_frames=5,multi_mv=True,skip_frames=3,ctype='image'):
    if os.path.isdir(sp):
        shutil.rmtree(sp,ignore_errors=True)
    datasets = [dataset.lower() for dataset in datasets]
    #sintel data
    if 'sintel' in datasets:prepare_sintel(osp.join(root,'Sintel'),osp.join(sp,'Sintel'),input_frames,multi_mv)
    #things data
    if 'things' in datasets:prepare_things(osp.join(root,'flyingthings'),osp.join(sp,'flyingthings'),input_frames,multi_mv)
    #UE4 data
    if 'unreal' in datasets:prepare_Unreal(osp.join(root,'Unreal'),osp.join(sp,'Unreal'),input_frames,multi_mv,skip_frames=skip_frames,ctype=ctype)
    #Spring data
    if 'spring' in datasets:prepare_Spring(osp.join(root,'Spring'),osp.join(sp,'Spring'),input_frames,multi_mv)
    if 'unreal_mrq' in datasets:prepare_Unreal(osp.join(root,'Unreal_MRQ'),osp.join(sp,'Unreal_MRQ'),input_frames,multi_mv,skip_frames=skip_frames,ctype=ctype)

def pickle_read(path):
    import pickle
    pickled_dat = open (path, "rb")
    return pickle.load(pickled_dat)

if __name__ == '__main__':
    # print('for test')
    root = '/home/rg0775/QingHong/opt_datasets'
    # root = '/Users/tcstudio_mm/Documents/gxliu/BaseLayer/MM/motionmodel/train/data/Unreal_MRQ'
    # source_root = find_folders_with_subfolder(root,keys=['image'],path_keys=['24fps'],path_excs=['/bbox'])
                
    pkl_root = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+'/data_pkl')
    # datasets = ['Sintel','things','Unreal','Spring']
    datasets = ['Unreal']
    train_data_prepare(root,pkl_root,datasets,input_frames=2,skip_frames=3)
    # cmd = 12
    # pkl_datas = pickle_read(f'/Users/qhong/Documents/1117test/MM/motionmodel/3rd/VideoFlow/core/data_pkl/Unreal/train_clean_{cmd}fps.pkl')
    # pkl_datas = pickle_read('/tt/nas/qhong/motionmodel/3rd/VideoFlow/data_pkl/Sintel/clean.pkl')
    
    # images,mv0s = pkl_datas['image_list'],pkl_datas['mv0_list']
    # print(np.array(images).shape)

