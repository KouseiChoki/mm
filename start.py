'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-09-27 13:42:15
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


import os,sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/algo')
if getattr(sys, 'frozen', None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/algorithm')
    sys.path.insert(1, "/tt/nas/miniconda3/lib/python3.9")
    sys.path.insert(2, "/tt/nas/miniconda3/lib/python3.9/lib-dynload")
    sys.path.insert(3, "/tt/nas/miniconda3/lib/python3.9/site-packages")
from myutil import *
from cfg_process import init_param
from scene_change_detection import scene_change_detect
from pre_treatment import pre_treatment,pre_treatment_3D,get_frames_count
from virtual_depth.virtual_depth_processor import virtual_depth_core
from combine_MM_file import *
cur_path = os.getcwd()
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
if len(sys.argv)<=1:
    raise ValueError('[MM ERROR][config]please specify config, usage: python start.py your_config')
config_file = sys.argv[1]
if not os.path.isfile(config_file):
    raise FileExistsError('[MM ERROR][config]config file not exist!')
##载入config文件
def run(args,att=None):
    gpu = args.gpu
    gpu = gpu.rstrip().split(',')
    num_gpu = len(gpu)
    scene_change_result = None
    cmd = 'torchrun --nproc_per_node={} algo/main.py {}'.format(num_gpu,config_file)
    if getattr(sys, 'frozen', None):
        cmd = 'torchrun --nproc_per_node={1} {0}/algorithm/main.py {2}'
        cmd = cmd.format(sys._MEIPASS, num_gpu, config_file)
    print('======================================== optical flow start ======================================== ')
    print('info:')
    print('algorithm:               {}'.format(args.algorithm))
    print('numbers of core:         {}'.format(num_gpu))
    print('calculate virtual depth: {}'.format(args.cal_virtual_depth))
    print('calculate disparity:     {}'.format(args.cal_disparity))
    if att:
        cmd +=f' {att}'
    try:
        # os.system(cmd)
        pass
    except:
        sys.exit('[MM ERROR][main process]main process error')
    else:
        args.distributed_task = (0,1)
        args.use_tqdm = True

        ## virtual depth
        try:
            if args.cal_virtual_depth:
                print('virtual depth')
                virtual_depth_core(args)
                
            if args.scene_change:
                print('=================================== scene change detection start ===================================== ')
                ckpt_path = os.path.dirname(os.path.abspath(__file__))+'/checkpoints/'
                model_name = 'scene_change'
                if not os.path.isfile(ckpt_path+model_name + '.pth'):
                    from conversion_tools.algorithm_center import check_and_download_pth_file
                    download_url = ckpt_path+model_name+'.pth'
                    md = args.server + '/scene_change'
                    md += '/' + model_name + '.pth'
                    print(download_url,md)
                    flag = check_and_download_pth_file(download_url,md)
                    if not flag:
                        raise NotImplementedError(f'[MM ERROR][model]model file not exists:{model_name},please use mmalgo to check')
        
                if args.threeD_mode: ##左右眼
                    image,_= pre_treatment_3D(args,args.root,args.image_file)
                else:
                    image = pre_treatment(args,args.root,args.image_file,single_mode=args.enable_single_input_mode_2D,single_file=args.root_2D)
                scene_change_result = scene_change_detect(image)

                write_txts(args.output,scene_change_result)
                    
            if args.MM9_format:
                print('======================================== MM9 format merging ======================================== ')
                masks_,right_masks_ = None,None
                if args.mask_mode:
                    front_masks = args.front_mask_file.split(',')
                    masks_,right_masks_ = {},{}
                    for front_mask in front_masks:
                        if args.threeD_mode: ##左右眼:
                            masks,right_masks = pre_treatment_3D(args,args.root,front_mask)
                        else:
                            masks = pre_treatment(args,args.root,front_mask,single_mode=args.enable_single_input_mode_2D,single_file=args.root_2D,multi_dir=args.multi_mask_file)
                            right_masks = None
                        masks_[front_mask] = masks
                        right_masks_[front_mask] = right_masks
                combine_result(args,masks_,right_masks_,multipro_nums=args.io_num_workers)
                if scene_change_result is not None:
                    create_mdl(args.output,scene_change_result,get_frames_count(os.path.join(args.output,args.MM)),args.MM,fps=args.fps)
                    write_txts(os.path.join(args.output,args.MM),scene_change_result)
        except Exception as e:
            print(e)
        
        print('========================================  finished ======================================== ')

if __name__ == '__main__':
    args = init_param(os.path.join(cur_path,config_file))
    run(args)

    

