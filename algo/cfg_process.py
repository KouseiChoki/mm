'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-06-26 12:55:59
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
import argparse,configparser
def adjust_weight(args):
    if '.pth' in args.weight_file:
        args.weight_file = args.weight_file.replace('.pth','')
    args.algorithm_fullname = args.weight_file
    args.algorithm = args.weight_file.rstrip().split('_')[0].split(' ')[0]
    if '_resize' in args.weight_file:
        gw = args.weight_file
        args.algorithm += gw[gw.find('_resize'):]
        args.weight_file = gw[:gw.find('_resize')]
    args.weight_file += '.pth'
    args.multi_frame_algo = multi_frame_algo_check(args.algorithm)
    if not args.multi_frame_algo:
        args.num_frames = 3
        args.multi_output = False
    return args
    
def init_param(path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help="Enable Eval mode.")
    parser.add_argument('--test', action='store_true', help="Enable Test mode.")
    parser.add_argument('--viz', action='store_true', help="Enable Viz mode.")
    parser.add_argument('--fixed_point_reuse', action='store_true', help="Enable fixed point reuse.")
    parser.add_argument('--warm_start', action='store_true', help="Enable warm start.")

    parser.add_argument('--name', default='deq-flow', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 

    parser.add_argument('--total_run', type=int, default=1, help="total number of runs")
    parser.add_argument('--start_run', type=int, default=1, help="begin from the given number of runs")
    parser.add_argument('--restore_name', help="restore experiment name")
    parser.add_argument('--resume_iter', type=int, default=-1, help="resume from the given iterations")

    parser.add_argument('--tiny', action='store_true', help='use a tiny model for ablation study')
    parser.add_argument('--large', action='store_true', help='use a large model')
    parser.add_argument('--huge', action='store_true', help='use a huge model')
    parser.add_argument('--gigantic', action='store_true', help='use a gigantic model')
    parser.add_argument('--old_version', action='store_true', help='use the old design for flow head')

    parser.add_argument('--restore_ckpt', help="restore checkpoint for val/test/viz")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--test_set', type=str, nargs='+')
    parser.add_argument('--viz_set', type=str, nargs='+')
    parser.add_argument('--viz_split', type=str, nargs='+', default=['test'])
    parser.add_argument('--output_path', help="output path for evaluation")

    parser.add_argument('--eval_interval', type=int, default=5000, help="evaluation interval")
    parser.add_argument('--save_interval', type=int, default=5000, help="saving interval")
    parser.add_argument('--time_interval', type=int, default=500, help="timing interval")

    parser.add_argument('--gma', action='store_true', help='use gma')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--schedule', type=str, default="onecycle", help="learning rate schedule")

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--vdropout', type=float, default=0.0, help="variational dropout added to BasicMotionEncoder for DEQs")
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--active_bn', action='store_true')
    parser.add_argument('--all_grad', action='store_true', help="Remove the gradient mask within DEQ func.")
    parser.add_argument('--wnorm', action='store_true', help="use weight normalization")
    parser.add_argument('--f_solver', default='anderson', type=str, choices=['anderson', 'broyden', 'naive_solver'],
                        help='forward solver to use (only anderson and broyden supported now)')
    parser.add_argument('--b_solver', default='broyden', type=str, choices=['anderson', 'broyden', 'naive_solver'],
                        help='backward solver to use')
    parser.add_argument('--f_thres', type=int, default=40, help='forward pass solver threshold')
    parser.add_argument('--b_thres', type=int, default=40, help='backward pass solver threshold')
    parser.add_argument('--f_eps', type=float, default=1e-3, help='forward pass solver stopping criterion')
    parser.add_argument('--b_eps', type=float, default=1e-3, help='backward pass solver stopping criterion')
    parser.add_argument('--f_stop_mode', type=str, default="abs", help="forward pass fixed-point convergence stop mode")
    parser.add_argument('--b_stop_mode', type=str, default="abs", help="backward pass fixed-point convergence stop mode")
    parser.add_argument('--eval_factor', type=float, default=1.5, help="factor to scale up the f_thres at test for better convergence.")
    parser.add_argument('--eval_f_thres', type=int, default=0, help="directly set the f_thres at test.")

    parser.add_argument('--indexing_core', action='store_true', help="use the indexing core implementation.")
    parser.add_argument('--ift', action='store_true', help="use implicit differentiation.")
    parser.add_argument('--safe_ift', action='store_true', help="use a safer function for IFT to avoid potential segment fault in older pytorch versions.")
    parser.add_argument('--n_losses', type=int, default=1, help="number of loss terms (uniform spaced, 1 + fixed point correction).")
    parser.add_argument('--indexing', type=int, nargs='+', default=[], help="indexing for fixed point correction.")
    parser.add_argument('--phantom_grad', type=int, nargs='+', default=[1], help="steps of Phantom Grad")
    parser.add_argument('--tau', type=float, default=1.0, help="damping factor for unrolled Phantom Grad")
    parser.add_argument('--sup_all', action='store_true', help="supervise all the trajectories by Phantom Grad.")

    parser.add_argument('--sradius_mode', action='store_true', help="monitor the spectral radius during validation")






    parser.add_argument('--model', help="restore checkpoint",default='checkpoints/gma-sintel.pth')
    parser.add_argument('--model_name', help="define model name", default="GMA")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args([])
    config = configparser.ConfigParser()
    config.read(path, encoding="utf-8")

    args.image_file = config.get('opticalflow','image_file')
    args.front_mask_file = config.get('opticalflow','front_mask_file')
    # args.back_mask_file = config.get('opticalflow','back_mask_file')
    args.weight_file = config.get('opticalflow','algorithm')
    
    args.mt_backend = config.get('opticalflow','mt_backend')
    args.evaluate_epe = config.getboolean('opticalflow','evaluate_epe')
    args.ground_truth_file = config.get('opticalflow','ground_truth_file')
    args.refine = config.getboolean('opticalflow','refine')
    args.root = config.get('opticalflow','root')
    args.output = config.get('opticalflow','output')
    args.cal_depth = config.getboolean('opticalflow','cal_depth')
    args.threeD_mode = config.getboolean('opticalflow','3D_mode')
    args.extra_depth = config.get('opticalflow','extra_depth')
    args.cal_virtual_depth = not args.threeD_mode and args.cal_depth and args.extra_depth.lower()=='none'
    args.cal_disparity = args.threeD_mode and args.cal_depth and args.extra_depth.lower()=='none'
    
    # args.right_eye_file = config.get('opticalflow','right_eye_file')
    # args.right_eye_back_mask_file = config.get('opticalflow','right_eye_back_mask_file')
    #extra mode
    args.enable_extra_input_mode = config.getboolean('opticalflow','enable_extra_input_mode')
    args.left_root = config.get('opticalflow','left_root')
    args.right_root = config.get('opticalflow','right_root')
    args.enable_single_input_mode_2D = config.getboolean('opticalflow','enable_single_input_mode_2D')
    args.root_2D = config.get('opticalflow','root_2D')
    args.num_frames = config.getint('opticalflow','num_frames')
    if '-disparity-' in args.weight_file:
        args.num_frames = 3
    if 'frames-' in args.weight_file:
        import re
        args.num_frames = re.search(r'(\d+)(?=frames-)', args.weight_file).group(1)

    #exr
    args.savetype = config.get('opticalflow','savetype')
    args.depth_range = config.getint('opticalflow','depth_range')
    # args.mask_mode= config.getboolean('opticalflow','mask_mode')
    args.mask_mode= True if '-mask-' in args.weight_file else False
    if args.mask_mode:
        if '-fg-' in args.weight_file:
            args.mask_type = 'fg'
        elif '-bg-' in args.weight_file:
            args.mask_type = 'bg'
        elif '-mix-' in args.weight_file:
            args.mask_type = 'mix'
        else:
            args.mask_type = 'all'
    # args.bg= config.getboolean('opticalflow','bg')
    args.self_mask= config.getboolean('opticalflow','self_mask')
    #enforce mask to image
    args.front_mask_file = args.image_file if args.self_mask else args.front_mask_file
    # args.mask_type = config.get('opticalflow','mask_type')
    args.restrain = config.getboolean('opticalflow','restrain')
    args.color_space = config.get('opticalflow','color_space')
    args.resize_rate_x = config.getfloat('opticalflow','resize_rate_x')
    args.resize_rate_y = config.getfloat('opticalflow','resize_rate_y')

    args.enable_dump = config.getboolean('opticalflow','enable_dump')
    args.load_dump = config.getboolean('opticalflow','load_dump')
    # args.fusion_mode = config.get('opticalflow','fusion_mode')   
    args.matting = config.getboolean('opticalflow','matting')
    # args.rm_ori_res = config.getboolean('opticalflow','rm_ori_res')
    args.n_limit = config.getint('opticalflow','n_limit')
    args.n_start = config.getint('opticalflow','n_start')
    args.gpu = config.get('opticalflow','gpu')
    args.restrain_all = config.getboolean('opticalflow','restrain_all')
    args.time_cost = config.getboolean('opticalflow','time_cost')
    args.cf_use_full = config.getboolean('opticalflow','cf_use_full')
    # args.bounding_with_no_restrain = config.getboolean('opticalflow','bounding_with_no_restrain')
    args.enable_single_input_mode = config.getboolean('opticalflow','enable_single_input_mode')
    args.pass_mv = config.getboolean('opticalflow','pass_mv')
    args.disparity_only = config.getboolean('opticalflow','disparity_only')
    args.pass_depth = config.getboolean('opticalflow','pass_depth')
    args.dump_restored_image = config.getboolean('opticalflow','dump_restored_image')
    args.dump_low_mv = config.getboolean('opticalflow','dump_low_mv')
    args.pass_when_exist = config.getboolean('opticalflow','pass_when_exist')
    args.depth_to_disparity = config.getboolean('opticalflow','depth_to_disparity')
    args.time_cost = False if args.pass_mv else args.time_cost
    args.depth_name = config.get('opticalflow','depth_name')  
    args.film_border = config.get('opticalflow','film_border')
    args.edge_filter = config.getboolean('opticalflow','edge_filter')
    args.edge_filter_ksize = config.getint('opticalflow','edge_filter_ksize')
    args.edge_filter_distance = config.getint('opticalflow','edge_filter_distance')
    args.edge_filter_iters = config.getint('opticalflow','edge_filter_iters')
    args.edge_filter_reverse = config.getboolean('opticalflow','edge_filter_reverse')

    args.cal_depth = config.getboolean('opticalflow','cal_depth')
    args.dilatation = config.getint('opticalflow','dilatation')
    args.erodition = config.getint('opticalflow','erodition')
    args.clean_tmp_file = config.getboolean('opticalflow','clean_tmp_file')
    args.cal_best_mv = config.getboolean('opticalflow','cal_best_mv')
    args.cal_best_mask_left = config.get('opticalflow','cal_best_mask_left')
    args.cal_best_mask_right = config.get('opticalflow','cal_best_mask_right')
    args.cal_best_target_algorithm = config.get('opticalflow','cal_best_target_algorithm')
    args.usewarm = config.getboolean('opticalflow','usewarm')
    args.left_dir_name = config.get('opticalflow','left_dir_name')
    args.right_dir_name = config.get('opticalflow','right_dir_name')

    #refine threshold
    args.threshold = config.getint('opticalflow','threshold')
    args.IMAGEIO_USERDIR = config.get('opticalflow','IMAGEIO_USERDIR') 
    args.TORCH_HUB = config.get('opticalflow','TORCH_HUB') 

    #multi masks
    args.multi_mask_mode = config.getboolean('opticalflow','multi_mask_mode')
    
    # args.auto_generate_depth = config.getboolean('opticalflow','auto_generate_depth') 
    args.io_num_workers = config.getint('opticalflow','io_num_workers') 
    args.virtualdepth_core = config.getint('opticalflow','virtualdepth_core') 


    #last
    args.empty_cache = config.getboolean('opticalflow','empty_cache')
    args.bounding_box_mode = config.getint('opticalflow','bounding_box_mode')
    args.fps = config.getint('opticalflow','fps')
    args.all_algorithm = config.get('opticalflow','all_algorithm')
    args.MM = config.get('opticalflow','MM')
    args.prune_repeat_frame = config.getboolean('opticalflow','prune_repeat_frame')
    args.compatibility_mode = config.getboolean('opticalflow','compatibility_mode')
    
    #color setting
    args.lut_file = config.get('opticalflow','lut_file')

    #output merge
    args.MM9_format = config.getboolean('opticalflow','MM9_format')
    args.clean_source = config.getboolean('opticalflow','clean_source')
    args.scene_change = config.getboolean('opticalflow','scene_change')

    #multiframe setting
    args.multi_output = config.getboolean('opticalflow','multi_output')
    
    #Post processing
    args.multi_mask_file = config.get('opticalflow','multi_mask_file') if args.multi_mask_mode else None
    # args.fix_inexistent_mv = True if args.MM9_format else config.getboolean('opticalflow','fix_inexistent_mv')
    args.compress_method = config.get('opticalflow','compress_method')
    args.use_bounding_box = config.getboolean('opticalflow','use_bounding_box') if args.mask_mode else False
    # args.weight_file = config.get('opticalflow','mask_algorithm') if args.mask_mode else args.weight_file
    # args.film_border = '0,0,0,0' if args.mask_mode else args.film_border 
    frame_step = config.get('opticalflow','frame_step')
    args.frame_step = [int(i) for i in frame_step.split(',')]
    # adjust to auto mode
    args = adjust_weight(args)
    #test mode
    args.testmode = config.getboolean('opticalflow','testmode')
    args.server = config.get('opticalflow','file_server_ip')
    if args.testmode:
        args.MM9_format = False
        args.clean_source = False
        args.scene_change = False
        args.cal_depth = False
        args.compress_method = 'piz'
    return args


def multi_frame_algo_check(algorithm):
    algos = ['videoflow','kousei']
    for algo in algos:
        if algo in algorithm.lower():
            return True
    return False