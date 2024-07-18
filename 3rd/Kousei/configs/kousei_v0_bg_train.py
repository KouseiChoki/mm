'''
Author: Qing Hong
Date: 2023-08-08 13:15:11
LastEditors: QingHong
LastEditTime: 2023-09-01 16:25:46
Description: file content
'''
from yacs.config import CfgNode as CN
_CN = CN()


_CN.batch_size = 7
_CN.sum_freq = 100
#datasets setting
_CN.kousei = CN()
_CN.kousei.do_flip = True
_CN.kousei.film_grain = [0,0] #min size, max size 0-8
_CN.kousei.crop_size = [880, 1920]
_CN.kousei.max_mv_x = 0.5
_CN.kousei.max_mv_y = 0.5
_CN.kousei.motion_blur_rate = 0
_CN.kousei.reverse_rate=0.3
_CN.kousei.repeat_frame_rate = 0.02

#stage setting
_CN.kousei.stage_sintel_clean = [0,1,1] #nums,min_scale,max_scale
_CN.kousei.stage_sintel_final = [0,1,1]
_CN.kousei.stage_things_clean = [0,1,1]
_CN.kousei.stage_things_final = [0,1,1]
_CN.kousei.stage_unreal_clean = [1,0,0.1]
_CN.kousei.stage_unreal_final = [2,0,0.1]
_CN.kousei.stage_spring_clean = [0,0,0.1]

#image setting
_CN.kousei.img_type = 'image' #hdr or image(ldr)

#mask setting
_CN.kousei.mask_type = 'bg'  # mix bg fg none
_CN.kousei.mask_treshold=120
_CN.kousei.mask_enhance_rate = 0.5
_CN.kousei.mask_enhance_range = [1,4]











_CN.name = ''
_CN.suffix =''
_CN.gamma = 0.9
_CN.val_freq = 100000000
_CN.use_smoothl1 = False
_CN.critical_params = []
_CN.mixed_precision = False
_CN.filter_epe = False
_CN.max_flow = _CN.kousei.crop_size[1] * _CN.kousei.max_mv_x
_CN.network = 'MOFNetStack'
_CN.model = 'VideoFlow_ckpt/videoflow_last.pth'
_CN.input_frames = 5
_CN.restore_ckpt = '/tt/nas/qhong/motionmodel/3rd/Kousei/train_checkpoints/1114_slow_bg/340000_ff.pth'
################################################
################################################
_CN.MOFNetStack = CN()
_CN.MOFNetStack.pretrain = True
_CN.MOFNetStack.Tfusion = 'stack'
_CN.MOFNetStack.cnet = 'basicencoder'
_CN.MOFNetStack.fnet = 'basicencoder'
_CN.MOFNetStack.down_ratio = 16
_CN.MOFNetStack.feat_dim = 256
_CN.MOFNetStack.corr_fn = 'default'
_CN.MOFNetStack.corr_levels = 4
_CN.MOFNetStack.mixed_precision = False
_CN.MOFNetStack.context_3D = False

_CN.MOFNetStack.decoder_depth = 4
_CN.MOFNetStack.critical_params = ["cnet", "fnet", "pretrain", 'corr_fn', "Tfusion", "corr_levels", "decoder_depth", "mixed_precision"]

### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'
_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 12.5e-5
_CN.trainer.adamw_decay = 1e-4
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 9000000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'
def get_cfg():
    return _CN.clone()
