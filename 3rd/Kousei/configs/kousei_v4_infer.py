'''
Author: Qing Hong
Date: 2023-08-08 13:15:11
LastEditors: QingHong
LastEditTime: 2024-04-22 11:14:23
Description: file content
'''

from yacs.config import CfgNode as CN
def get_cfg():
    _CN = CN()
    _CN.BOFNet = CN()
    _CN.trainer = CN()
    _CN.kousei = CN()
    _CN.batch_size = 1
    _CN.sum_freq = 100
    #datasets setting
    _CN.kousei = CN()
    _CN.kousei.do_flip = True
    _CN.kousei.film_grain = [0,0] #min size, max size 0-8
    _CN.kousei.crop_size = [800, 1920]
    _CN.kousei.max_mv_x = 0.6
    _CN.kousei.max_mv_y = 0.6
    _CN.kousei.motion_blur_rate = 0
    _CN.kousei.reverse_rate=0.3
    _CN.kousei.repeat_frame_rate = 0.01
    #stage setting
    _CN.kousei.stage_sintel_clean = [0,1,1] #nums,min_scale,max_scale
    _CN.kousei.stage_sintel_final = [0,1,1]
    _CN.kousei.stage_things_clean = [0,1,1]
    _CN.kousei.stage_things_final = [0,1,1]
    _CN.kousei.stage_unreal_clean = [1,0,0]
    _CN.kousei.stage_unreal_final = [0,0,0]
    _CN.kousei.stage_spring_clean = [1,0,0]
    #image setting
    _CN.kousei.img_type = 'image' #hdr or image or ldr
    #mask setting
    _CN.kousei.mask_type = None  # mix bg fg none
    _CN.kousei.mask_treshold=120
    _CN.kousei.mask_enhance_rate = 0.3
    _CN.kousei.mask_enhance_range = [-1,-1]
    #mask setting
    _CN.kousei.mask_type = None
    _CN.kousei.kata_mask_mode = True
    _CN.kousei.mask_treshold=120
    _CN.kousei.mask_enhance_rate = 0
    _CN.kousei.mask_enhance_range = [-2,2]
    _CN.name = ''
    _CN.suffix =''
    _CN.gamma = 0.85
    _CN.val_freq = 100000000
    _CN.use_smoothl1 = False
    _CN.critical_params = []
    _CN.mixed_precision = False
    _CN.filter_epe = False
    _CN.max_flow = _CN.kousei.crop_size[1] * _CN.kousei.max_mv_x
    _CN.network = 'BOFNet'
    _CN.input_frames = 3
    _CN.restore_ckpt = None
    ################################################
    ################################################

    _CN.BOFNet.cnet = 'basicencoder'
    _CN.BOFNet.fnet = 'basicencoder'
    _CN.BOFNet.down_ratio = 16
    _CN.BOFNet.decoder_depth = 32
    _CN.BOFNet.gma = 'GMA-SK2'
    _CN.BOFNet.pretrain = True
    _CN.BOFNet.Tfusion = 'stack'
    _CN.BOFNet.feat_dim = 256
    _CN.BOFNet.corr_fn = 'default'
    _CN.BOFNet.corr_levels = 4
    _CN.BOFNet.mixed_precision = False
    _CN.BOFNet.context_3D = False
    _CN.BOFNet.critical_params = ["cnet", "fnet", "pretrain", "corr_fn", "mixed_precision"]

    ### TRAINER
    _CN.trainer.scheduler = 'OneCycleLR'
    _CN.trainer.optimizer = 'adamw'
    _CN.trainer.canonical_lr = 12.5e-5
    _CN.trainer.adamw_decay = 1e-4
    _CN.trainer.clip = 1.0
    _CN.trainer.num_steps = 9000000
    _CN.trainer.epsilon = 1e-8
    _CN.trainer.anneal_strategy = 'linear'
    return _CN
