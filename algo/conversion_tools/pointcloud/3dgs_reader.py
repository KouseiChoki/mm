'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-08-12 14:35:03
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
from cal_ply import ImageInfo,mkdir,CameraInfo,write_colmap_model,ja_ajust,jhelp_file,jhelp,jhelp_folder
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
# from striprtf.striprtf import rtf_to_text
from fileutil.read_write_model import Camera,write_model,Image
from file_utils import mvwrite,read
import argparse
IMG_DATA = ['.png','.tiff','.tif','.exr','.jpg']
def prune(c,keyword,mode = 'basename'):
    if mode =='basename':
        res = list(filter(lambda x:keyword.lower() not in os.path.basename(x).lower(),c)) 
    else:
        res = list(filter(lambda x:keyword.lower() not in x.lower(),c))
    return res 
def gofind(c,keywords,mode = 'basename'):
    if isinstance(keywords, str):  # 如果传入的是字符串，转换为列表
        keywords = [keywords]
    if mode == 'basename':
        res = list(filter(lambda x: any(keyword.lower() in os.path.basename(x).lower() for keyword in keywords), c))
    else:
        res = list(filter(lambda x: any(keyword.lower() in x.lower() for keyword in keywords), c))
    return res  

def init_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',  help="your data path", required=True)
    parser.add_argument('--step',type=int, default=1,help="frame step")
    parser.add_argument('--start_frame',type=int, default=0,help="start frame")
    parser.add_argument('--max_frame',type=int, default=999,help="max generated frames")
    parser.add_argument('--baseline_distance', type=float, default=0,help="baseline_distance")
    parser.add_argument('--f', action='store_true', help="force run")
    parser.add_argument('--mask', action='store_true', help="use mask")
    parser.add_argument('--judder_angle',type=int, default=-1,help="frame step")
    parser.add_argument('--cur',type=int, default=-1,help="which frame do not use mask")
    parser.add_argument('--inverse_depth',action='store_true', help="depth= 1/depth")
    parser.add_argument('--test', action='store_true', help="use test")
    parser.add_argument('--downscale',type=int, default=1,help="downscale rate")
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


def generate_point_cloud_from_depth(depth_image, intrinsics, extrinsics,mask=None):
    h, w = depth_image.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    # 相机内参
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # 计算每个像素的三维坐标
    z = depth_image.astype(np.float32) # 假设深度以毫米为单位，转换为米
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    # 将点组合成[N, 3]的点云
    points_camera = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    # 去除非法点
    if mask is not None:
        points_camera = points_camera[mask]
    points_camera = points_camera[points_camera[:, 2] != 0]
    # 将点云从相机坐标系转换到世界坐标系
    points_world = (extrinsics[:3, :3] @ points_camera.T).T + extrinsics[:3, 3]
    return points_world

def get_intrinsic_extrinsic(images,depths,ins,ext,save_path,args,masks=None):
    index = 1
    nums = len(images)
    cam_infos,image_infos = [],[]
    points,rgbs = [],[]
    for i in range(args.start_frame,nums,args.step): 
        rx,ry,rz,tx,ty,tz = ext[i]
        # mkdir(os.path.join(save_path,'image'))
        # image_path = os.path.join(save_path,'image',os.path.basename(image))
        # if not os.path.isfile(image_path) or args.f:
        #     shutil.copy(image,image_path)
        # if masks is not None:
        #     mask = masks[i]
        #     mkdir(os.path.join(save_path,'masks'))
        #     mask_path = os.path.join(save_path,'masks',os.path.basename(mask))
        #     if not os.path.isfile(mask_path) or args.f:
        #         shutil.copy(mask,mask_path)
        w,h = int(ins['w']),int(ins['h'])
        rotation_matrix = R.from_euler('XYZ', [rx,ry,rz],degrees=True).as_matrix()
        c2w = np.eye(4,4)
        if args.baseline_distance!=0:
            tx += args.baseline_distance
        c2w[:3,:3] = rotation_matrix
        c2w[:,1:3] *= -1
        c2w[:3,-1] = [tx,ty,tz]
        # tt = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        # c2w*= tt
        # if i ==1:
        #     print(c2w)
        #     sys.exit(0)

        w2c = np.linalg.inv(c2w)
        qx, qy, qz ,qw = R.from_matrix(w2c[:3, :3]).as_quat()
        tvec0,tvec1,tvec2 = w2c[:3, 3]
        
        image_info = ImageInfo(uid=index,extrinsic=np.array([qw,qx,qy,qz,tvec0,tvec1,tvec2]))
        image_infos.append(image_info)
        cam_info = CameraInfo(uid=index, fx=ins['focal_length_x'],fy=ins['focal_length_y'],cx=w/2.0 ,cy=h/2.0,image_name=os.path.basename(images[i]),image_path = images[i], width=w, height=h,model="PINHOLE")
        cam_infos.append(cam_info)

        #downscale
        o_cx = w/2.0 
        o_cy = h/2.0
        o_cx = o_cx //args.downscale
        o_cy = o_cy //args.downscale
        focal_length_x = ins['focal_length_x']/args.downscale
        focal_length_y = ins['focal_length_y']/args.downscale
        target_w = w//args.downscale
        target_h = h//args.downscale

        #point
        intrinsics = np.array([[focal_length_x,0,o_cx],[0,focal_length_y,o_cy],[0,0,1]])
        depth = read(depths[i])[...,0]
        if args.inverse_depth:
            depth = 1/depth
        rgb = read(images[i],type='image')
        if args.downscale != 1:
            import cv2
            depth = cv2.resize(depth,(target_w,target_h),interpolation=cv2.INTER_NEAREST)
            rgb = cv2.resize(rgb,(target_w,target_h))
        # rgb = rgb[depth!= 0]
        rgb=rgb.reshape(-1,3)
        
        if masks is not None:
            if args.cur == i+1:
                point = generate_point_cloud_from_depth(depth,intrinsics,c2w)
            else:
                mask_path = masks[i]
                mask = read(mask_path,type='mask')
                if args.downscale != 1 :
                    mask = cv2.resize(mask,(target_w,target_h),interpolation=cv2.INTER_NEAREST).reshape(-1)
                else:
                    mask = mask.reshape(-1)
                rgb = rgb[mask == 0]
                # point = point[mask == 0]
                point = generate_point_cloud_from_depth(depth,intrinsics,c2w,mask == 0)
        else:
            point = generate_point_cloud_from_depth(depth,intrinsics,c2w)
        points.append(point.reshape(-1,3))
        rgbs.append(rgb.reshape(-1,3))
        index += 1
    #create pointcloud
    xyz = np.concatenate(points)
    rgbs = np.concatenate(rgbs)
    # print('writing plyfile........')
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgbs), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    return image_infos,cam_infos,ply_data
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

def ply_cal_core(images,depths,instrinsics,extrinsics,sp,args,masks=None):
    if args.baseline_distance!=0:
        sp+=f'_bd_{args.baseline_distance}'

    sparse_path = os.path.join(sp,'sparse/0')
    ply_path = os.path.join(sparse_path , "points3D.ply")
    if os.path.isdir(sp):
        if not args.f and args.judder_angle==-1:
            return
        else:
            shutil.rmtree(sp,ignore_errors=True)
    image_infos,cam_infos,ply_data = get_intrinsic_extrinsic(images,depths,instrinsics,extrinsics,save_path,args,masks)
    mkdir(os.path.join(sp , "images"))
    for image in images:
        shutil.copy(image, os.path.join(sp , "images",os.path.basename(image)))
    mkdir(os.path.join(sp , "masks"))
    for mask in masks:
        shutil.copy(mask, os.path.join(sp , "masks",os.path.basename(mask)))
    # if mask_folder is not None:
    #     shutil.copytree(mask_folder, os.path.join(sp ,os.path.basename(mask_folder)),dirs_exist_ok=True)
    # shutil.copytree(image_folder, os.path.join(sp , os.path.basename(image_folder)),dirs_exist_ok=True)
    # Write out the camera parameters.
    write_colmap_model(sparse_path,cam_infos,image_infos)
    # shutil.copy(raw_ply,os.path.join(sp,'sparse/0/points3D.ply'))
    mkdir(sparse_path)
    # if args.baseline_distance==0:
    ply_data.write(ply_path)
    if args.judder_angle!= -1:
        print('writing ja file')
        image_infos,cam_infos = ja_ajust(image_infos,cam_infos,args.judder_angle)
        sp += f'_ja_{args.judder_angle}'
        shutil.rmtree(sp,ignore_errors=True)
        sparse_path = os.path.join(sp,'sparse/0')
        mkdir(sparse_path)
        # Write out the images.
        mkdir(os.path.join(sp , "images"))
        for image in images:
            shutil.copy(image, os.path.join(sp , "images",os.path.basename(image)))
        mkdir(os.path.join(sp , "masks"))
        for mask in masks:
            shutil.copy(mask, os.path.join(sp , "masks",os.path.basename(mask)))
        # if mask_folder is not None:
        #     shutil.copytree(mask_folder, os.path.join(sp ,os.path.basename(mask_folder)),dirs_exist_ok=True)
        # shutil.copytree(image_folder, os.path.join(sp , os.path.basename(image_folder)),dirs_exist_ok=True)
        write_colmap_model(sparse_path,cam_infos,image_infos)
        # shutil.copy(raw_ply,os.path.join(sp,'sparse/0/points3D.ply'))
        # if args.baseline_distance==0:
        ply_data.write(ply_path)
    
def read_intrinsic(intrinsic_file):
    res = {}
    res['w'],res['h'],res['focal_length_x'],res['focal_length_y'] = read_txt(intrinsic_file)[0]
    return res

def read_extrinsics(extrinsic_file):
    return read_txt(extrinsic_file)

def read_txt(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Skip the header line
    for line in lines[1:]:
        # Split the line into components and convert them to floats
        components = list(map(float, line.strip().split()))
        data.append(components)
    return data

def sliding_window(arr, n):
    if arr is None:
        return [None for i in range(len(arr) - n + 1)]
    return [arr[i:i + n] for i in range(len(arr) - n + 1)]

if __name__ == '__main__':
    args = init_param()
    args.f = True
    # rtf = '/home/rg0775/QingHong/data/plytestdata/fg_avatar_0725/0729_3frames/raw/6DoF.rtf'
    # path = '/home/rg0775/QingHong/data/plytestdata/fg_avatar_0725/0729_3frames/raw'
    path = args.root
    if os.path.basename(args.root) != 'raw':
        tmps = prune(jhelp(args.root),'raw')
        if not os.path.isdir(os.path.join(args.root,'raw')):
            mkdir(os.path.join(args.root,'raw'))
            for tmp in tmps:
                shutil.move(tmp,os.path.join(args.root,'raw',os.path.basename(tmp)))
        path = os.path.join(args.root,'raw')
    intrinsic_file = gofind(jhelp_file(path),'intrinsic.txt')[0]
    extrinsic_file = gofind(jhelp_file(path),'6DoF.txt')[0]
    
    # data = read_rtf(rtf)
    # lines = data.strip().split('\n')
    # rows = [list(map(float, line.split())) for line in lines[1:]]
    #prune data
    try:
        image_folder = gofind(jhelp_folder(path),'images')[0]
        mask_folder = gofind(jhelp_folder(path),'masks')[0]
        depth_folder = gofind(jhelp_folder(path),'depths')[0]
        images = jhelp_file(image_folder)
        masks = jhelp_file(mask_folder)
        depths  = jhelp_file(depth_folder)
    except:
        raise ImportError('error input folder, need IMAGES and DEPTHS (MASKS) folder!')
    assert len(images)==len(masks) and len(images)==len(depths),f'error input number of image/mask/depth,{len(images)},{len(masks)},{len(depths)}'
    if len(images) <= args.max_frame:
        images_prepare = [images]
        masks_prepare = [masks]
        depths_prepare = [depths]
    else:
        images_prepare = sliding_window(images,args.max_frame)
        masks_prepare = sliding_window(masks,args.max_frame)
        depths_prepare = sliding_window(depths,args.max_frame)


    instrinsics = read_intrinsic(intrinsic_file) # not finished
    extrinsics = sliding_window(read_extrinsics(extrinsic_file),args.max_frame)
    
    for i in tqdm(range(len(images_prepare)),desc=os.path.basename(os.path.abspath(os.path.join(path,'..')))):
        name0 = os.path.splitext(os.path.basename(images_prepare[i][0]))[0]
        name1 = os.path.splitext(os.path.basename(images_prepare[i][-1]))[0]
        name = f'{name0}_to_{name1}'
        save_path = os.path.join(path,'..','pointcloud',name)
        ply_cal_core(images_prepare[i],depths_prepare[i],instrinsics,extrinsics[i],save_path,args,masks_prepare[i])
    print('finished')
    