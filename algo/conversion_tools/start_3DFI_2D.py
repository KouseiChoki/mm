'''
Author: Qing Hong
Date: 2022-05-25 17:35:01
LastEditors: QingHong
LastEditTime: 2023-03-28 15:44:06
Description: file content
'''
test = False
clean = False
depthin = False
import os,sys
import configparser
from myutil import *
import numpy
# from myutil import mkdir
cur_path = sys.argv[0][:-sys.argv[0][::-1].find('/')]
if 'site-package' in cur_path:
    cur_path = ''
elif cur_path.lower() in ['m','s']:
    cur_path = ''
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
TMPFILE = 'tmpconfigfile'
# assert len(sys.argv)>1 ,'please specify config, usage: python start_3DFI_2D.py your_config'
if not test:
    config_file = sys.argv[1]
else:
    config_file = 'config_mask'
##载入config文件

def run(args,cf = None,att=None):
    gpu = args.gpu
    gpu = gpu.rstrip().split(',')
    num_gpu = len(gpu)
    cmd = 'torchrun --nproc_per_node={} main.py {}'.format(num_gpu,cf)
    mkdir(args.output)
    print('======================================== optical flow start ======================================== ')
    print('info:')
    print('algorithm:               {}'.format(args.algorithm))
    print('numbers of core:         {}'.format(num_gpu))
    print('calculate depth:         {}'.format(args.cal_depth))
    print('merge depth:             {}'.format(args.merge_depth))
    rm_ori_res = args.rm_ori_res
    print('remove original result:  {}'.format(args.rm_ori_res))
    if att:
        cmd +=f' {att}'
    os.system(cmd)
    # python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_gpu_using_launch.py
    if rm_ori_res:
        print('clear the original result')
        rmcmd = 'rm -r ' +args.output+'/*/*original*'
        os.system(rmcmd)
        # print(rmcmd)
    print('======================================== optical flow finished ======================================== ')

def updateFile(file,old_strs,new_strs):
    """
    替换文件中的字符串
    :param file:文件名
    :param old_str:就字符串
    :param new_str:新字符串
    :return:
    """
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            for index,old_str in enumerate(old_strs):
                if old_str in line and '#' not in line:
                    line = new_strs[index]
            file_data += line
    with open(file,"w",encoding="utf-8") as f:
        f.write(file_data)

def write_config_file(config_file,keywords,datas):
    for i in range(len(keywords)):
        old_str = keywords[i]
        new_str = keywords[i] + ' = ' + datas[i] + '\n' if '=' not in keywords[i] else keywords[i] + ' ' + datas[i] +'\n'
        keywords[i] = old_str
        datas[i] = new_str
    updateFile(config_file,keywords,datas)

args = init_param(cur_path + config_file)
if not test:
    run(args,cur_path + config_file)
if os.path.isfile(cur_path+TMPFILE):
    os.remove(cur_path+TMPFILE)
shutil.copy(cur_path + config_file,cur_path+TMPFILE)
# write_config_file(cur_path+TMPFILE,['algorithm =','char','threshold ='],['flowformer-32g-mask16.pth','1','255'])
## 用fulldepth替换
# write_config_file(cur_path+TMPFILE,['algorithm =','char','threshold =','cal_depth =','image_file ='],['flowformer-30g-UE4-mask14-v2.pth','1','255','0','left'])
write_config_file(cur_path+TMPFILE,['algorithm =','char =','threshold =','cal_depth =','image_file =','gpu ='],['flowformer-32g-mask16.pth','1','255','0','le','0'])
mask_args = init_param(cur_path + TMPFILE)
if not test:
    run(mask_args,cur_path + TMPFILE)

if os.path.isfile(cur_path+TMPFILE):
    os.remove(cur_path+TMPFILE)


# 2d 3DFI Processing
# args.root = cur_path +args.root
# args.output = cur_path +args.output
scenes = jhelp(args.output)
scene_range = range(len(scenes))
for i in scene_range:
    scene = scenes[i]
    scene_name = os.path.basename(scene)
    masks = jhelp(os.path.join(args.root,scene_name,args.front_mask_file))
    files = jhelp(scene)
    sp0 = os.path.join(scene,args.algorithm+'_Object_Mixed_mv0')
    sp1 = os.path.join(scene,args.algorithm+'_Object_Mixed_mv1')
    mkdir(sp0),mkdir(sp1)
    weight_file_name = args.algorithm + '_' +args.gma_weight_file.replace('.pth','')
    weight_file_mask_name = mask_args.algorithm + '_' +mask_args.gma_weight_file.replace('.pth','')
    frame_mv0_file = os.path.join(scene,weight_file_name+'_left_original_mv0')
    assert frame_mv0_file in files,'frame_mv0_file error'+frame_mv0_file
    frame_mv0 = jhelp(frame_mv0_file)

    frame_mv1_file = os.path.join(scene,weight_file_name+'_left_original_mv1')
    assert frame_mv0_file in files,'frame_mv1_file error'
    frame_mv1 = jhelp(frame_mv1_file)

    obj_mv0_file = os.path.join(scene,weight_file_mask_name+'_Char_mv0')
    assert frame_mv0_file in files,'obj_mv0_file error'
    obj_mv0= jhelp(obj_mv0_file)

    obj_mv1_file = os.path.join(scene,weight_file_mask_name+'_Char_mv1')
    assert frame_mv0_file in files,'obj_mv1_file error'
    obj_mv1 = jhelp(obj_mv1_file)

    depth_left_file = os.path.join(scene,weight_file_name+'_left_right_depth_original')
    depth_left = None
    depth_right_file = os.path.join(scene,weight_file_name+'_right_left_depth_original')
    depth_right = None
    if depthin:
        assert depth_left_file in files and depth_right_file in files
        depth_left = jhelp(depth_left_file)
        depth_right = jhelp(depth_right_file)
        
    
        
    for i in tqdm(range(len((frame_mv0)))):
        fm = read(frame_mv0[i])
        om = read(obj_mv0[i])
        mask = read(masks[i])[...,0]
        # fm[np.where(mask!=0)] = om[np.where(mask!=0)]
        fm[np.where(mask>=args.threshold//2)] = om[np.where(mask>=args.threshold//2)]

        if depthin:
            dp = read(depth_left[i])[...,-1]
            om[...,-1] = dp
        om[np.where(mask<args.threshold)] = 0
        write(os.path.join(sp0,os.path.basename(frame_mv0[i])),fm)
        # write(os.path.join(sp0,os.path.basename(frame_mv)),om)
        fm = read(frame_mv1[i])
        om = read(obj_mv1[i])
        mask = read(masks[i+1])[...,0]
        # fm[np.where(mask!=0)] = om[np.where(mask!=0)]
        fm[np.where(mask>=args.threshold//2)] = om[np.where(mask>=args.threshold//2)]
        if depthin:
            dp = read(depth_right[i])[...,-1]
            om[...,-1] = dp
        om[np.where(mask<args.threshold)] = 0
        write(os.path.join(sp1,os.path.basename(frame_mv1[i])),fm)
        # write(os.path.join(sp0,os.path.basename(frame_mv)),om)
    # if clean:
    #     for file in files:
    #         if 'Mixed' not in file:
    #             shutil.rmtree(file,ignore_errors=True)

