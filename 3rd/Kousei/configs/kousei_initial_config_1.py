'''
Author: Qing Hong
Date: 2023-08-08 13:15:11
LastEditors: QingHong
LastEditTime: 2024-04-08 17:18:46
Description: file content
'''

from yacs.config import CfgNode as CN
def get_cfg(args):
    _CN = CN()
    _CN.BOFNet = CN()
    _CN.trainer = CN()
    _CN.kousei = CN()
    if args.version == 0:
        _CN.kousei.crop_size = [880, 1920]
        _CN.kousei.crop_size = [432, 960]
        _CN.BOFNet.cnet = 'basicencoder'
        _CN.BOFNet.fnet = 'basicencoder'
        _CN.BOFNet.down_ratio = 16
        _CN.BOFNet.decoder_depth = 4
        _CN.BOFNet.gma = 'GMA-SK2'
    elif  args.version == 1:
        _CN.kousei.crop_size = [432, 960]
        _CN.BOFNet.cnet = 'twins'
        _CN.BOFNet.fnet = 'twins'
        _CN.BOFNet.down_ratio = 8
        _CN.BOFNet.decoder_depth = 9
        _CN.BOFNet.gma = 'GMA-SK2'
    elif args.version == 3:
        _CN.kousei.crop_size = [268, 480]
        _CN.BOFNet.cnet = 'basicencoder'
        _CN.BOFNet.fnet = 'basicencoder'
        _CN.BOFNet.down_ratio = 4
        _CN.BOFNet.decoder_depth = 4
        _CN.BOFNet.gma = 'GMA'
    else:
        raise NotImplementedError('wrong version')
    _CN.batch_size = args.nums_gpu
    _CN.sum_freq = 100
    #datasets setting
    _CN.kousei.do_flip = args.do_flip
    _CN.kousei.film_grain = args.film_grain
    _CN.kousei.max_mv_x = args.max_mv_x
    _CN.kousei.max_mv_y = args.max_mv_y
    _CN.kousei.motion_blur_rate = args.motion_blur_rate
    _CN.kousei.reverse_rate=args.reverse_rate
    _CN.kousei.repeat_frame_rate = args.repeat_frame_rate
    #stage setting
    _CN.kousei.stage_sintel_clean = args.sintel_clean
    _CN.kousei.stage_sintel_final = args.sintel_final
    _CN.kousei.stage_things_clean = args.things_clean
    _CN.kousei.stage_things_final = args.things_final
    _CN.kousei.stage_unreal_clean = args.unreal_clean
    _CN.kousei.stage_unreal_final = args.unreal_final
    _CN.kousei.stage_spring_clean = args.spring_clean
    _CN.kousei.stage_unreal_MRQ_clean = args.unreal_mrq_clean
    _CN.kousei.stage_unreal_MRQ_final = args.unreal_mrq_final
    #image setting
    _CN.kousei.img_type = args.img_type
    #mask setting
    _CN.kousei.mask_type = args.mode
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
    _CN.restore_ckpt = args.restore_ckpt
    ################################################
    ################################################
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
