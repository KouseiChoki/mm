'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2025-01-06 16:22:36
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/../../')
import shutil
from file_utils import jhelp_file,read,mkdir

root = '/Users/qhong/Desktop/0103/UE_XYPitchYaw'
all_odt = []
keyword = '_pw_fmt.fbx'

def check_fbx_valid(root,keyword):
    fbx_file = None
    for dir_name, _, file_list in os.walk(root):
        if fbx_file is None:
            for tmp in file_list:
                if keyword in tmp:
                    fbx_file = os.path.join(dir_name,tmp)
                    break
    return fbx_file

def generate_data_from_fbx(root,keyword='_pw_fmt.fbx'):
    fbx_file = check_fbx_valid(root,keyword)
    if fbx_file is None:
        return
    image_dir,mask_dir,ply_file = '','',''
    for dir_name, subdir_list, file_list in os.walk(root):
        for tmp in subdir_list:
            if tmp.lower() in ['image','images']:
                image_dir = os.path.join(dir_name,tmp)
            if tmp.lower() in ['mask','masks']:
                mask_dir = os.path.join(dir_name,tmp)
        for tmp in file_list:
            if '.ply' in tmp:
                ply_file = os.path.join(dir_name,tmp)
    return fbx_file,image_dir,mask_dir,ply_file


def generate_ply(images_folder):
    pass

images_folder = '/Users/qhong/Desktop/0103/UE_XYPitchYaw/raw/images'


images = jhelp_file(images_folder)[:10][::2]

for image_ in images:
    depth_ = image_.replace('images','depths')
    image = read(image_)
    depth_= read(depth_)[...,0]