'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-10-15 12:04:30
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

import re,os
import numpy as np
from myutil import appendzero,getname
from file_utils import mask_read

def repeat_frame_check(img1,img2):
    return np.abs(img1[...,:3] - img2[...,:3]).sum()<1

def get_pairs(indexes,images,append_name_,masks,args,step=1):
    # if len(images)-1 in indexes: #fix bg last frame bug
    #     indexes = [len(images)-1-i*step for i in range(len(indexes))]
    #     indexes = indexes[::-1]
    valid,p_valid,names,img_stack = [],[],[],[]
    # valid calculation
    for i in range(1,len(indexes)-1):
        position_valid = indexes[i] >= 0 and indexes[i]<= len(images)-1
        valid_mv1 = indexes[i-1] >= 0 and indexes[i-1]<= len(images)-1 and position_valid
        valid_mv0 = indexes[i+1] >= 0 and indexes[i+1]<= len(images)-1 and position_valid
        valid.append(valid_mv0)
        valid.append(valid_mv1)
        p_valid.append(position_valid)
        p_valid.append(position_valid)
        name = getname(images[np.clip(indexes[i],0,len(images)-1)])
        append_name = append_name_.format(appendzero(int(re.findall(r'\d+', name)[-1]),8))
        names.append(append_name.replace('$$',f'mv{step*2-2}'))
        names.append(append_name.replace('$$',f'mv{step*2-1}'))
    valid = valid[::2] + valid[1:][::2] ## mv0,mv0,mv0,mv1,mv1,mv1
    names = names[::2] + names[1:][::2]
    p_valid = p_valid[::2] + p_valid[1:][::2]
    for i in range(len(p_valid)):
        if not p_valid[i]:
            names[i] = None
    if not args.multi_output or step != 1 or not args.multi_frame_algo: #single output mode , only output middle pair mv
        tmp = [False] * len(names)
        tmp[args.num_frames//2 - 1] = True #p1 mv0
        tmp[args.num_frames//2 - 1 + args.num_frames -2] = True #cf mv1
        names = [z[0] if z[1] else None for z in zip(names,tmp)]
    if args.pass_when_exist and all(os.path.isfile(save_name) for save_name in names if save_name is not None):
        return [],[],[]
    for index in indexes:
        index = np.clip(index,0,len(images)-1)
        mask = None if masks is None else masks[index]
        img = mask_read(images[index],mask,args)
        img_stack.append(img)
    #repeat frame check
    if args.prune_repeat_frame:
        for i in range(1,len(img_stack)-1):
            if repeat_frame_check(img_stack[i-1],img_stack[i]):
                valid[i - 1 + (args.num_frames-2)] = False #cf mv1
            if repeat_frame_check(img_stack[i],img_stack[i+1]):
                valid[i - 1] = False #cf mv0
    #norm
    if args.multi_frame_algo:
        if args.color_space.lower() == 'image':
            for i in range(len(img_stack)):
                img_stack[i][...,:3] = 2 * (img_stack[i][...,:3] / 255.0) - 1.0
        else:
            for i in range(len(img_stack)):
                img_stack[i][...,:3] = 2 * (img_stack[i][...,:3]) - 1.0
    return img_stack,valid,names