'''
Author: Qing Hong
Date: 2022-10-27 13:43:03
LastEditors: QingHong
LastEditTime: 2022-11-17 16:50:35
Description: file content
'''
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
from tqdm import tqdm
import time
import imageio
from myutil import *
from algo import * 
from tqdm import tqdm
from algo import pre_treatment_caldepth
dir_mytest = os.path.dirname(os.path.abspath(__file__))+'/core'
sys.path.insert(0, dir_mytest)

##position
cur_time = str(time.gmtime().tm_mon) + '_' + str(time.gmtime().tm_mday)
cur_time_sec = cur_time+'/'+str(time.gmtime().tm_hour)+'/'+str(time.gmtime().tm_sec)

if __name__ == '__main__':
    args = init_param(sys.argv[1])
    args.distributed_task = [0,1]
    args.cur_rank = args.distributed_task[0] + 1
    args.use_tqdm = True if args.cur_rank ==1 else False
    
    if args.IMAGEIO_USERDIR.lower()!='none':
        os.environ['IMAGEIO_USERDIR'] = args.IMAGEIO_USERDIR

    if not args.cal_depth:
        #单目，未实现
        assert args.cal_depth ,'best mv calculation not support for 2d image!'
    else:
        left_eye_image,right_eye_image= pre_treatment_caldepth(args,args.image_file,args.right_eye_file)
        if not (args.cal_best_mask_left.lower() == 'none' and args.cal_best_mask_right.lower() =='none'):
            mask_left_dict,mask_right_dict = pre_treatment_caldepth(args,args.cal_best_mask_left,args.cal_best_mask_right)

        scenes =[os.path.join(args.output,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(args.output))))]
        for scene in scenes:
            if args.cal_best_mask_left.lower() == 'none' and args.cal_best_mask_right.lower() =='none':
                mask_left_files =  None
                mask_right_files =  None
            else:
                mask_left_files =  mask_left_dict[scene.split('/')[-1]]
                mask_right_files =  mask_right_dict[scene.split('/')[-1]]
            all_mv_files = list(filter(lambda x:x[0]!='.',sorted(os.listdir(scene))))
            assert len(all_mv_files)>1,'algorithm less than 2! run another algorithm first'
            total_algorithm = [i.replace('_left_original_mv0','') for i in list(filter(lambda x:'left_original_mv0' in x,all_mv_files))]
            image_left_files =  left_eye_image[scene.split('/')[-1]]
            image_right_files =  right_eye_image[scene.split('/')[-1]]
            base_algorithm = total_algorithm[0]
            lr0_ = os.path.join(scene,f'{base_algorithm}_left_right_mergedepth_mv0')
            lr1_ = os.path.join(scene,f'{base_algorithm}_left_right_mergedepth_mv1')
            rl0_ = os.path.join(scene,f'{base_algorithm}_right_left_mergedepth_mv0')
            rl1_ = os.path.join(scene,f'{base_algorithm}_right_left_mergedepth_mv1')

            lr0 = [os.path.join(lr0_,i) for i in sorted(os.listdir(lr0_))]
            lr1 = [os.path.join(lr1_,i) for i in sorted(os.listdir(lr1_))]
            rl0 = [os.path.join(rl0_,i) for i in sorted(os.listdir(rl0_))]
            rl1 = [os.path.join(rl1_,i) for i in sorted(os.listdir(rl1_))]

            assert len(image_left_files)-1 == len(image_right_files)-1 ==len(lr0)==len(lr1)==len(rl0)==len(rl1),f'image error, please check the number of images and mask files,{len(image_left_files)-1} == {len(image_right_files)-1} =={len(lr0)}=={len(lr1)}=={len(rl0)}=={len(rl1)}'
            #生成目标文件夹
            result_file = os.path.join(scene,f'best')
            mkdir(result_file)
            lr_mv0 = os.path.join(result_file,'left_right_mergedepth_mv0')
            lr_mv1 = os.path.join(result_file,'left_right_mergedepth_mv1')
            rl_mv0 = os.path.join(result_file,'right_left_mergedepth_mv0')
            rl_mv1 = os.path.join(result_file,'right_left_mergedepth_mv1')
            mkdir(lr_mv0),mkdir(lr_mv1),mkdir(rl_mv0),mkdir(rl_mv1)
            choice = {'c1':[],'c2':[],'c3':[],'c4':[]}
            for i in tqdm(range(len(lr0))):
                final_choice1,final_choice2,final_choice3,final_choice4 = [0,0],[0,0],[0,0],[0,0]
                best_lr_mv0 = read(lr0[i])
                best_lr_mv1 = read(lr1[i])
                best_rl_mv0 = read(rl0[i])
                best_rl_mv1 = read(rl1[i])
                mask_l = read(mask_left_files[i]) if mask_left_files else np.ones_like(image_l_pre)
                mask_r = read(mask_right_files[i]) if mask_right_files else np.ones_like(image_r_pre)
                if i ==0 or i ==len(lr0)-1:
                    #第一帧和最后一帧不作处理
                    choice['c1'].append(final_choice1)
                    choice['c2'].append(final_choice2)
                    choice['c3'].append(final_choice3)
                    choice['c4'].append(final_choice4)
                    best_lr_mv0 = reprocessing(best_lr_mv0,args,mask_l)
                    best_lr_mv1 = reprocessing(best_lr_mv1,args,mask_l)
                    best_rl_mv0 = reprocessing(best_rl_mv0,args,mask_r)
                    best_rl_mv1 = reprocessing(best_rl_mv1,args,mask_r)
                    imageio.imwrite(os.path.join(lr_mv0,'mvd_{:0>8}.exr'.format(int(re.findall(r'\d+', lr0[i])[-1]))),best_lr_mv0)
                    imageio.imwrite(os.path.join(lr_mv1,'mvd_{:0>8}.exr'.format(int(re.findall(r'\d+', lr1[i])[-1]))),best_lr_mv1)
                    imageio.imwrite(os.path.join(rl_mv0,'mvd_{:0>8}.exr'.format(int(re.findall(r'\d+', rl0[i])[-1]))),best_rl_mv0)
                    imageio.imwrite(os.path.join(rl_mv1,'mvd_{:0>8}.exr'.format(int(re.findall(r'\d+', rl1[i])[-1]))),best_rl_mv1)
                    continue
                image_l_pre = read(image_left_files[i-1])
                image_r_pre = read(image_right_files[i-1])
                image_l_next = read(image_left_files[i+1])
                image_r_next = read(image_right_files[i+1])
                

                if mask_l.shape != best_lr_mv0.shape:
                    mask_l = cv2.resize(mask_l,best_lr_mv0.shape[:2][::-1])
                    mask_r = cv2.resize(mask_r,best_lr_mv0.shape[:2][::-1])
                if len(scenes)>1:
                    for other_algorithm in total_algorithm[1:]:
                        lr0_other = [i.replace(base_algorithm,other_algorithm) for i in lr0]
                        lr1_other = [i.replace(base_algorithm,other_algorithm) for i in lr1]
                        rl0_other = [i.replace(base_algorithm,other_algorithm) for i in rl0]
                        rl1_other = [i.replace(base_algorithm,other_algorithm) for i in rl1]
                        best_lr_mv0,choice1 = compare_with(best_lr_mv0,read(lr0_other[i]),image_l_next,mask_l)
                        best_lr_mv1,choice2 = compare_with(best_lr_mv1,read(lr1_other[i]),image_l_pre,mask_l)
                        best_rl_mv0,choice3 = compare_with(best_rl_mv0,read(rl0_other[i]),image_r_next,mask_r)
                        best_rl_mv1,choice4 = compare_with(best_rl_mv1,read(rl1_other[i]),image_r_pre,mask_r)
                        final_choice1[0]+=choice1[0]
                        final_choice1[1]+=choice1[1]
                        final_choice2[0]+=choice2[0]
                        final_choice2[1]+=choice2[1]
                        final_choice3[0]+=choice3[0]
                        final_choice3[1]+=choice3[1]
                        final_choice4[0]+=choice4[0]
                        final_choice4[1]+=choice4[1]
                    choice['c1'].append(final_choice1)
                    choice['c2'].append(final_choice2)
                    choice['c3'].append(final_choice3)
                    choice['c4'].append(final_choice4)
                #图像后处理
                best_lr_mv0 = reprocessing(best_lr_mv0,args,mask_l)
                best_lr_mv1 = reprocessing(best_lr_mv1,args,mask_l)
                best_rl_mv0 = reprocessing(best_rl_mv0,args,mask_r)
                best_rl_mv1 = reprocessing(best_rl_mv1,args,mask_r)
                imageio.imwrite(os.path.join(lr_mv0,'mvd_{:0>8}.exr'.format(int(re.findall(r'\d+', lr0[i])[-1]))),best_lr_mv0)
                imageio.imwrite(os.path.join(lr_mv1,'mvd_{:0>8}.exr'.format(int(re.findall(r'\d+', lr1[i])[-1]))),best_lr_mv1)
                imageio.imwrite(os.path.join(rl_mv0,'mvd_{:0>8}.exr'.format(int(re.findall(r'\d+', rl0[i])[-1]))),best_rl_mv0)
                imageio.imwrite(os.path.join(rl_mv1,'mvd_{:0>8}.exr'.format(int(re.findall(r'\d+', rl1[i])[-1]))),best_rl_mv1)
            #写入结果txt
            if len(scenes)>1:
                Note=open(result_file+'/record.txt',mode='w')
                for c,datas in choice.items():
                    if c =='c1':
                        Note.write("left_right_mv0's algorithm is :\n")
                    if c =='c2':
                        Note.write("left_right_mv1's algorithm is :\n")
                    if c =='c3':
                        Note.write("right_left_mv0's algorithm is :\n")
                    if c =='c4':
                        Note.write("right_left_mv1's algorithm is :\n")
                    for i in range(len(datas)):
                        Note.write('mvd_{:0>8}.exr     Char:{}, Background:{}\n'.format(int(re.findall(r'\d+', lr0[i])[-1]),total_algorithm[datas[i][0]],total_algorithm[datas[i][1]]))
                Note.close()
            if args.clean_tmp_file:
                print('clean tmp file')
                rmcmd = 'rm -r ' +args.output+'/*/*mv*'
                os.system(rmcmd)
                rmcmd = 'rm -r ' +args.output+'/*/*right*'
                os.system(rmcmd)
                rmcmd = 'rm -r ' +args.output+'/*/*left*'
                os.system(rmcmd)

            


