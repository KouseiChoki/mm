'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-07-17 11:40:24
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
import numpy as np
import os,sys,shutil
from tqdm import tqdm
from cal_ply import ImageInfo,mkdir,CameraInfo,write_colmap_model,ja_ajust,jhelp_file,gofind,eulerAngles2rotationMat
from scipy.spatial.transform import Rotation as R
from striprtf.striprtf import rtf_to_text
from fileutil.read_write_model import Camera,write_model,Image
from file_utils import mvwrite,read
import argparse

def init_param():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--path',  help="your data path", required=True)
    parser.add_argument('--step',type=int, default=1,help="frame step")
    parser.add_argument('--start_frame',type=int, default=0,help="start frame")
    parser.add_argument('--max_frame',type=int, default=999,help="max frame")
    parser.add_argument('--f', action='store_true', help="force run")
    parser.add_argument('--mask', action='store_true', help="use mask")
    parser.add_argument('--judder_angle',type=int, default=-1,help="frame step")
    parser.add_argument('--test', action='store_true', help="use test")
    args = parser.parse_args()
    return args

# judder_angle = -1
def read_rtf(file_path):
    with open(file_path, 'r') as file:
        rtf_content = file.read()
        text_content = rtf_to_text(rtf_content)
    return text_content
# rtf = '/Users/qhong/Downloads/6DoF.rtf'
# path = '/Users/qhong/Downloads/3200_Vanilla/'
# data = read_rtf(rtf)
# lines = data.strip().split('\n')

# # Split each line into columns and convert to appropriate data types
# header = lines[0].split('\t')
# rows = [list(map(float, line.split())) for line in lines[1:]]
# index = 1
# image_infos = []
# for row in rows:
#     rx,ry,rz,tx,ty,tz = row
#     # extrinsic = eulerAngles2rotationMat([yaw,pitch,roll], loc = [tx,ty,tz], format='degree', order = 'XYZ',axis='right')
#     # angles = np.deg2rad([-yaw, -pitch, roll])
#     rotation_matrix = R.from_euler('ZYX', [-rx,-ry,rz],degrees=True).as_matrix()
#     extrinsic = np.eye(4,4)
#     extrinsic[:3,:3] = rotation_matrix
#     extrinsic[:3,-1] = [-tx,-ty,tz]
#     w2c = np.linalg.inv(extrinsic)
#     qx, qy, qz ,qw = R.from_matrix(w2c[:3, :3]).as_quat()
#     tvec = w2c[:3, 3]
#     image_info = ImageInfo(uid=index,extrinsic=np.array([qw,qx,qy,qz,tvec[0],tvec[1],tvec[2]]))
#     image_infos.append(image_info)
#     index += 1

# cameras = {}
# images = {}
# tmp = ['3200.1019.png','3200.1020.png','3200.1021.png']
# i = 0
# for image_info in image_infos:
#     qw,qx,qy,qz,tx,ty,tz = image_info.extrinsic
#     cameras[image_info.uid] = Camera(image_info.uid, 'PINHOLE', 2180 ,1152, (2417.84648 ,2417.84648, 1090.0 ,576.0))
#     qvec = np.array((qw, qx, qy, qz))
#     tvec = np.array((tx, ty, tz))
#     images[image_info.uid] = Image(image_info.uid, qvec, tvec, image_info.uid, tmp[i], [], [])
#     mkdir(path)
#     write_model(cameras, images, None, path,ext='.txt')
#     i +=1

# print(image_infos)

def get_intrinsic_extrinsic(images,rows,save_path,name,args):
    index = 1
    nums = len(images)
    cam_infos,image_infos = [],[]
    for i in tqdm(range(args.start_frame,nums,args.step),desc=f'processing {name}'): 
        if index>args.max_frame:
            break
        image = images[i]
        rx,ry,rz,tx,ty,tz = rows[i]
        mkdir(os.path.join(save_path,'image'))
        image_path = os.path.join(save_path,'image',os.path.basename(image))
        if not os.path.isfile(image_path) or args.f:
            shutil.copy(image,image_path)
        w,h = 2180,1152
        rotation_matrix = R.from_euler('XYZ', [rx,ry,rz],degrees=True).as_matrix()
        c2w = np.eye(4,4)
        c2w[:3,:3] = rotation_matrix
        c2w[:,1:3] *= -1
        c2w[:3,-1] = [tx,ty,tz]
    #     M = np.array([
    #     [1, 0, 0, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    # ])
    #     c2w= M @ c2w @ M
        # c2w[:3,:3] = euler_angles_to_rotation_matrix(np.radians(rx),np.radians(-ry),np.radians(-rz))
        # c2w= eulerAngles2rotationMat([rx,ry,rz], loc = [tx,ty,tz], format='degree', order = 'ZYX',axis='right')
        # extrinsic[:3,-1] = [tx,ty,tz]

        # print(euler_angles_to_rotation_matrix(np.radians(rx),np.radians(-ry),np.radians(-rz)))
        # print(rotation_matrix)
        w2c = np.linalg.inv(c2w)
        qx, qy, qz ,qw = R.from_matrix(w2c[:3, :3]).as_quat()
        tvec = w2c[:3, 3]
        image_info = ImageInfo(uid=index,extrinsic=np.array([qw,qx,qy,qz,tvec[0],tvec[1],tvec[2]]))
        image_infos.append(image_info)
        # intrinsic
        o_cx = w/2.0 
        o_cy = h/2.0
        model="PINHOLE"
        focal_length_x = 2417.84648
        focal_length_y = 2417.84648
        cam_info = CameraInfo(uid=index, fx=focal_length_x,fy=focal_length_y,cx=o_cx,cy=o_cy,image_name=os.path.basename(image_path).replace('.png',''),image_path = image_path, width=w, height=h,model=model)
        cam_infos.append(cam_info)
        index += 1
    return image_infos,cam_infos
def euler_angles_to_rotation_matrix(theta_x, theta_y, theta_z):
    """
    将欧拉角转换为旋转矩阵（按 ZYX 顺序）。
    """
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    R_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    R_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    # R = R_z @ R_y @ R_x
    R = R_x @ R_y @ R_z
    return R

def ply_cal_core(images,rows,save_path,args):
    name = os.path.basename(save_path)
    sp = os.path.join(save_path,'pointcloud')
    sparse_path = os.path.join(sp,'sparse/0')
    ply_path = os.path.join(sparse_path , "points3D.ply")
    if os.path.isdir(sp):
        if not args.f and args.judder_angle==-1:
            return
        else:
            shutil.rmtree(sp,ignore_errors=True)
    image_infos,cam_infos = get_intrinsic_extrinsic(images,rows,save_path,name,args)
    mkdir(os.path.join(sp , "images"))
    for cam_info in cam_infos:
        shutil.copy(cam_info.image_path, os.path.join(sp , "images",os.path.basename(cam_info.image_path)))
    # Write out the camera parameters.
    write_colmap_model(sparse_path,cam_infos,image_infos)
    if args.judder_angle!= -1:
        print('writing ja file')
        image_infos,cam_infos = ja_ajust(image_infos,cam_infos,args.judder_angle)
        sp = os.path.join(save_path,f'pointcloud_ja_{args.judder_angle}')
        shutil.rmtree(sp,ignore_errors=True)
        sparse_path = os.path.join(sp,'sparse/0')
        mkdir(sparse_path)
        # Write out the images.
        mkdir(os.path.join(sp , "images"))
        for cam_info in cam_infos:
            shutil.copy(cam_info.image_path, os.path.join(sp , "images",os.path.basename(cam_info.image_path)))
        write_colmap_model(sparse_path,cam_infos,image_infos)


if __name__ == '__main__':
    args = init_param()
    rtf = '/Users/qhong/Desktop/0717_ply/s_movement_0716/alita/6DoF.rtf'
    path = '/Users/qhong/Desktop/0717_ply/s_movement_0716/alita'
    args.f = True
    data = read_rtf(rtf)
    lines = data.strip().split('\n')
    rows = [list(map(float, line.split())) for line in lines[1:]]
    file_datas = jhelp_file(path)
    #prune data
    file_datas = gofind(file_datas,'.png')
    save_path = os.path.abspath(os.path.join(path,'..'))
    ply_cal_core(file_datas,rows,save_path,args)