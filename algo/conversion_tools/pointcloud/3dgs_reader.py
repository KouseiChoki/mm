'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2025-08-27 17:53:47
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
# import open3d as o3d
import numpy as np
import os,sys,shutil
from tqdm import tqdm
from cal_ply import ImageInfo,mkdir,CameraInfo,write_colmap_model,ja_ajust,jhelp_file,jhelp,jhelp_folder
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
# from striprtf.striprtf import rtf_to_text
from fileutil.read_write_model import Camera,write_model,Image
from file_utils import write,read
from myutil import mask_adjust,write_txt
import argparse
MAX_DEPTH = 1e6
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
    parser.add_argument('--root','--path',  help="your data path", required=True)
    parser.add_argument('--step',type=int, default=1,help="frame step")
    parser.add_argument('--start_frame',type=int, default=0,help="start frame")
    parser.add_argument('--max_frame',type=int, default=5,help="max generated frames")
    parser.add_argument('--baseline_distance', type=float, default=0,help="baseline_distance")
    parser.add_argument('--f', action='store_true', help="force run")
    parser.add_argument('--mask_type', type=str,default='nomask', help="bg or mix",choices=['nomask','bg','mix'])
    parser.add_argument('--mask_threshold', type=float, default=0,help="prune mask threshold")
    parser.add_argument('--fg_mask_adjust', type=int, default=0,help="prune mask threshold")
    parser.add_argument('--bg_mask_adjust', type=int, default=0,help="prune mask threshold")
    parser.add_argument('--judder_angle',type=int, default=-1,help="frame step")
    parser.add_argument('--inverse_depth',action='store_true', help="depth= 1/depth")
    parser.add_argument('--inverse_mask',action='store_true', help="invere mask value")
    parser.add_argument('--inverse_mask_original_data',action='store_true', help="invere mask value")
    parser.add_argument('--rub', action='store_true', help="dump rub viewmatrix")
    parser.add_argument('--test', action='store_true', help="use test")
    parser.add_argument('--down_scale',type=int, default=1,help="downscale rate")
    parser.add_argument('--custom',nargs="+",type=int, default=[0],help="custom input mode, like 2,4,5(key frame is first value:2)")
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
def get_ply(xyz,rgbs):
    #create pointcloud
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
    # print('writing plyfile........')
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    assert xyz.shape[0] == rgbs.shape[0],'error input, please check your depth data(contains 0) or add --inverse_depth'
    attributes = np.concatenate((xyz, normals, rgbs), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    return ply_data


# def generate_point_cloud(rgb_,depth_,int,T_WC):
    
#     """
#     Converts depth maps to point clouds and merges them all into one global point cloud.
#     flags: command line arguments
#     data: dict with keys ['intrinsics', 'poses']
#     returns: [open3d.geometry.PointCloud]
#     """
#     intrinsics = o3d.camera.PinholeCameraIntrinsic(width=depth_.shape[1], height=depth_.shape[0], fx=int[0, 0],
#         fy=int[1, 1], cx=int[0, 2], cy=int[1, 2])
#     T_WC = np.linalg.inv(T_WC)

#     rgb = o3d.geometry.Image(rgb_)
#     depth = o3d.geometry.Image(depth_)
#     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#     rgb, depth,depth_scale=1.0, depth_trunc=MAX_DEPTH, convert_rgb_to_intensity=False)
#     return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics, extrinsic=T_WC)

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
    # points_camera = points_camera[points_camera[:, 2] != 0]
    # 将点云从相机坐标系转换到世界坐标系
    points_world = (extrinsics[:3, :3] @ points_camera.T).T + extrinsics[:3, 3]
    # points_camera += extrinsics[:3, 3]
    # points_world = (extrinsics[:3, :3] @ points_camera.T).T
    
    return points_world

# def generate_point_cloud_from_depth(depth_image, intrinsics, extrinsics, mask=None):
#     h, w = depth_image.shape
#     i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
#     # 相机内参
#     fx, fy = intrinsics[0, 0], intrinsics[1, 1]
#     cx, cy = intrinsics[0, 2], intrinsics[1, 2]
#     # 计算每个像素的三维坐标
#     z = depth_image.astype(np.float32)  # 假设深度以毫米为单位，转换为米
#     x = (i - cx) * z / fx
#     y = (j - cy) * z / fy
#     # 将点组合成 [N, 3] 的点云
#     points_camera = np.stack((x, y, z), axis=-1).reshape(-1, 3)
#     # 去除非法点
#     if mask is not None:
#         points_camera = points_camera[mask]
#     points_camera = points_camera[points_camera[:, 2] != 0]
#     # 如果平移是以原点计算，需要显式地将平移矢量应用到世界坐标原点
#     # 先从 extrinsics 中提取旋转和平移
#     rotation_matrix = extrinsics[:3, :3]
#     translation_vector = extrinsics[:3, 3]
#     # 转换到世界坐标系，平移由原点参考
#     points_world = (rotation_matrix @ points_camera.T).T + translation_vector
#     print("Rotation Matrix:\n", rotation_matrix)
#     print("Translation Vector:\n", translation_vector)
#     print("Points Camera (sample):\n", points_camera[:5])
#     print("Points World (sample):\n", points_world[:5])
#     return points_world

# def cal_qvec(data):
#     rx,ry,rz,tx,ty,tz = data
#     rotation_matrix = R.from_euler('zyx', [rx,ry,rz],degrees=True).as_matrix()
#     c2w = np.eye(4,4)
#     if args.baseline_distance!=0:
#         tx += args.baseline_distance
#     c2w[:3,:3] = rotation_matrix
#     translation_vector = [tx,ty,tz]

#     c2w[:3,-1] = translation_vector
#     c2w = np.linalg.inv(c2w)
#     # c2w[:3,-1] = -rotation_matrix @ translation_vector
#     # print(tx,ty,tz)
#     rub = c2w.copy() if args.rub else None
#     w2c = np.linalg.inv(c2w)
#     qx, qy, qz ,qw = R.from_matrix(w2c[:3, :3]).as_quat()
#     tvec0,tvec1,tvec2 = w2c[:3, 3]
#     return np.array([qw,qx,qy,qz,tvec0,tvec1,tvec2]),c2w,rub

# def cal_qvec(data):
#     from scipy.spatial.transform import Rotation as R
#     rx,ry,rz,tx,ty,tz = data
#     rotation_matrix = R.from_euler('YXZ', [rx,ry,rz],degrees=True).as_matrix()
#     c2w = np.eye(4,4)
#     c2w[:3,:3] = rotation_matrix
#     translation_vector = [tx,ty,tz]
#     c2w[:3,-1] = translation_vector
#      # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
#     c2w[:3, 1:3] *= -1
#     # get the world-to-camera transform and set R, T
#     w2c = np.linalg.inv(c2w)
#     R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
#     T = w2c[:3, 3]
#     return R,T

from enum import Enum
class EulerOrder(Enum):
    XYZ = 0
    XZY = 1
    YXZ = 2
    YZX = 3
    ZYX = 4
    ZXY = 5

def euler_to_rotation(euler, euler_order, is_right_handed=True):
    """
    Converts Euler angles to a rotation matrix.
    
    Parameters:
        euler (list or np.ndarray): A list or array of Euler angles [ex, ey, ez] in radians.
        euler_order (EulerOrder): The order of rotations.
        is_right_handed (bool): Whether the rotation is in a right-handed coordinate system.
    
    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    ex, ey, ez = euler
    if is_right_handed:
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(ex), -np.sin(ex)],
            [0, np.sin(ex), np.cos(ex)]
        ])
        Ry = np.array([
            [np.cos(ey), 0, np.sin(ey)],
            [0, 1, 0],
            [-np.sin(ey), 0, np.cos(ey)]
        ])
        Rz = np.array([
            [np.cos(ez), -np.sin(ez), 0],
            [np.sin(ez), np.cos(ez), 0],
            [0, 0, 1]
        ])
    else:  # Left-handed system
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(ex), np.sin(ex)],
            [0, -np.sin(ex), np.cos(ex)]
        ])
        Ry = np.array([
            [np.cos(ey), 0, -np.sin(ey)],
            [0, 1, 0],
            [np.sin(ey), 0, np.cos(ey)]
        ])
        Rz = np.array([
            [np.cos(ez), np.sin(ez), 0],
            [-np.sin(ez), np.cos(ez), 0],
            [0, 0, 1]
        ])

    # Combine rotations based on the order
    if euler_order == EulerOrder.XYZ:
        R = Rx @ Ry @ Rz
    elif euler_order == EulerOrder.XZY:
        R = Rx @ Rz @ Ry
    elif euler_order == EulerOrder.YXZ:
        R = Ry @ Rx @ Rz
    elif euler_order == EulerOrder.YZX:
        R = Ry @ Rz @ Rx
    elif euler_order == EulerOrder.ZYX:
        R = Rz @ Ry @ Rx
    elif euler_order == EulerOrder.ZXY:
        R = Rz @ Rx @ Ry
    else:
        raise ValueError("Invalid Euler order.")

    return R




def cal_qvec_rub_to_rdf(data):
    rx,ry,rz,tx,ty,tz = data
    # 示例输入
    # euler_angles_r = [np.radians(rx), np.radians(ry), np.radians(rz)]  # 以弧度为单位
    # euler_angles = [rx,ry,rz]  # 以弧度为单位
    # order = EulerOrder.XYZ
    # is_right_handed = True
    # 生成旋转矩阵
    # rotation_matrix = euler_to_rotation(euler_angles, order, False)
    
    # rotation_matrix = R.from_euler('XYZ', [rx,ry,rz],degrees=True).as_matrix()
    rotation_matrix = R.from_euler('xyz', [rx,ry,rz],degrees=True).as_matrix()
    # 输出结果
    # print("Rotation Matrix:")
    # print(Rotation)
    # print("rotation_matrix")
    # print(rotation_matrix)
    # sys.exit(0)
    
    c2w = np.eye(4,4)
    if args.baseline_distance!=0:
        tx += args.baseline_distance
    c2w[:3,:3] = rotation_matrix
    translation_vector = [tx,ty,tz]
    # print(tx,ty,tz)
    c2w[:3,-1] = translation_vector
     # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1
    # get the world-to-camera transform and set R, T
    rub = c2w.copy() if args.rub else None
    w2c = np.linalg.inv(c2w)
    # c2w[:3,-1] = w2c[:3,-1]
    qx, qy, qz ,qw = R.from_matrix(w2c[:3, :3]).as_quat()
    tvec0,tvec1,tvec2 = w2c[:3, 3]
    return np.array([qw,qx,qy,qz,tvec0,tvec1,tvec2]),c2w,rub

def get_intrinsic_extrinsic(images,depths,ins,ext,save_path,args,masks=None):
    index = 1
    nums = len(images)
    cam_infos,image_infos = [],[]
    points,rgbs = [],[]
    tmp_points,tmp_rgbs = [],[]
    fg_ply_data = None
    for i in range(nums): 
        w,h = int(ins['w']),int(ins['h'])
        # etmp,c2w,rub = cal_qvec_unreal_to_rdf(ext[i])
        etmp,c2w,rub = cal_qvec_rub_to_rdf(ext[i])
        image_info = ImageInfo(uid=index,extrinsic=etmp,rub=rub)
        image_infos.append(image_info)
        cam_info = CameraInfo(uid=index, fx=ins['fx'],fy=ins['fy'],cx=w/2.0 ,cy=h/2.0,image_name=os.path.basename(images[i]),image_path = images[i], width=w, height=h,model="PINHOLE")
        cam_infos.append(cam_info)

        #downscale
        o_cx = w/2.0 
        o_cy = h/2.0
        o_cx = o_cx //args.down_scale
        o_cy = o_cy //args.down_scale
        focal_length_x = ins['fx']/args.down_scale
        focal_length_y = ins['fy']/args.down_scale
        target_w = w//args.down_scale
        target_h = h//args.down_scale

        #point
        intrinsics = np.array([[focal_length_x,0,o_cx],[0,focal_length_y,o_cy],[0,0,1]])
        depth = read(depths[i])[...,0]
        if args.inverse_depth:
            depth = 1/depth
        rgb = read(images[i],type='image')
        if args.down_scale != 1:
            import cv2
            depth = cv2.resize(depth,(target_w,target_h),interpolation=cv2.INTER_NEAREST)
            rgb = cv2.resize(rgb,(target_w,target_h))
        # rgb = rgb[depth!= 0]
        rgb_=rgb.reshape(-1,3)
        mask = None
        if args.mask_type != 'nomask':
            mask = read(masks[i],type='mask')
            if args.inverse_mask:
                mask = np.abs(mask-1)
            if args.bg_mask_adjust != 0:
                mask = mask_adjust(mask,size=args.bg_mask_adjust)
            if args.down_scale != 1 :
                mask = cv2.resize(mask,(target_w,target_h),interpolation=cv2.INTER_NEAREST).reshape(-1)
            else:
                mask = mask.reshape(-1)
            condition = mask <= args.mask_threshold
            rgb = rgb_[condition]
            # c2w = np.linalg.inv(c2w)
            point = generate_point_cloud_from_depth(depth,intrinsics,c2w,condition)
            
            if args.mask_type =='mix' and args.cur == i:
                cur_mask = read(masks[i],type='mask')
                if args.fg_mask_adjust != 0:
                    cur_mask = mask_adjust(cur_mask,size=args.fg_mask_adjust)
                if args.down_scale != 1 :
                    cur_mask = cv2.resize(cur_mask,(target_w,target_h),interpolation=cv2.INTER_NEAREST).reshape(-1)
                else:
                    cur_mask = cur_mask.reshape(-1)
                tmp_condition = (cur_mask > args.mask_threshold) & (depth.reshape(-1)!=0)
                tmp_rgbs = rgb_[tmp_condition]
                tmp_points = generate_point_cloud_from_depth(depth,intrinsics,c2w,tmp_condition)
        else:
            point = generate_point_cloud_from_depth(depth,intrinsics,c2w)
            # tmp = generate_point_cloud_from_depth(rgb,depth,intrinsics,c2w)
            # point = np.asarray(tmp.points)
            # rgb = np.asarray(tmp.colors)
        if point is not None:

            if mask is not None:
                # points_camera = points_camera[mask]
                point = point.reshape(-1,3)[depth.reshape(-1)[condition]<MAX_DEPTH]
            else:
                point = point.reshape(-1,3)[depth.reshape(-1)<MAX_DEPTH]

            points.append(point.reshape(-1,3))
        if rgb is not None:
            if mask is not None:
                # points_camera = points_camera[mask]
                rgb = rgb.reshape(-1,3)[depth.reshape(-1)[condition]<MAX_DEPTH]
            else:
                rgb = rgb.reshape(-1,3)[depth.reshape(-1)<MAX_DEPTH]
            rgbs.append(rgb.reshape(-1,3))
        index += 1
    xyz = np.concatenate(points)
    rgbs = np.concatenate(rgbs)
    ply_data = get_ply(xyz,rgbs)
    if len(tmp_points)>0:
        # xyz = np.array(tmp_points)
        # rgbs = np.array(tmp_rgbs)
        fg_ply_data = get_ply(tmp_points,tmp_rgbs)
    return image_infos,cam_infos,ply_data,fg_ply_data
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
    image_infos,cam_infos,ply_data,fg_ply_data = get_intrinsic_extrinsic(images,depths,instrinsics,extrinsics,save_path,args,masks)
    mkdir(os.path.join(sp , "images"))
    for image in images:
        shutil.copy(image, os.path.join(sp , "images",os.path.basename(image)))
    if masks is not None:
        mkdir(os.path.join(sp , "masks"))
        for mask in masks:
            if not args.inverse_mask_original_data:
                shutil.copy(mask, os.path.join(sp , "masks",os.path.basename(mask)))
            else:
                mask_ = read(mask,type='mask')
                mask_ = np.abs(mask_-1)
                write(os.path.join(sp, "masks",os.path.basename(mask)),mask_*255)

    # if mask_folder is not None:
    #     shutil.copytree(mask_folder, os.path.join(sp ,os.path.basename(mask_folder)),dirs_exist_ok=True)
    # shutil.copytree(image_folder, os.path.join(sp , os.path.basename(image_folder)),dirs_exist_ok=True)
    # Write out the camera parameters.
    mkdir(sparse_path)
    write_colmap_model(sparse_path,cam_infos,image_infos)
    if fg_ply_data is not None:
        fg_ply_data.write(os.path.join(sparse_path , f"fg_points3D.ply"))
    # shutil.copy(raw_ply,os.path.join(sp,'sparse/0/points3D.ply'))
    # if args.baseline_distance==0:
    if ply_data is not None:
        ply_data.write(ply_path)
    
    # if args.judder_angle is not None and args.judder_angle!= -1:
    #     print('writing ja file')
    #     image_infos,cam_infos = ja_ajust(image_infos,cam_infos,args.judder_angle)
    #     sp += f'_ja_{args.judder_angle}'
    #     shutil.rmtree(sp,ignore_errors=True)
    #     sparse_path = os.path.join(sp,'sparse/0')
    #     mkdir(sparse_path)
    #     # Write out the images.
    #     mkdir(os.path.join(sp , "images"))
    #     for image in images:
    #         shutil.copy(image, os.path.join(sp , "images",os.path.basename(image)))
    #     mkdir(os.path.join(sp , "masks"))
    #     for mask in masks:
    #         shutil.copy(mask, os.path.join(sp , "masks",os.path.basename(mask)))
    #     # if mask_folder is not None:
    #     #     shutil.copytree(mask_folder, os.path.join(sp ,os.path.basename(mask_folder)),dirs_exist_ok=True)
    #     # shutil.copytree(image_folder, os.path.join(sp , os.path.basename(image_folder)),dirs_exist_ok=True)
    #     write_colmap_model(sparse_path,cam_infos,image_infos,step=args.step)
    #     # shutil.copy(raw_ply,os.path.join(sp,'sparse/0/points3D.ply'))
    #     # if args.baseline_distance==0:
    #     if ply_data is not None:
    #         ply_data.write(ply_path)
    
def read_intrinsic(intrinsic_file):
    res = {}
    res['w'],res['h'],res['fx'],res['fy'] = read_txt(intrinsic_file)[0]
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

def sliding_window(sequence, window_size=3,window_step=1,step=1,pad=0,pad_step=0):
    """Generate a sliding window over a sequence."""
    res = []
    index = 0
    while True:
        choose = [index + step * i for i in range(window_size)]
        if choose[-1]>=len(sequence):
            break
        tmp = [sequence[c] for c in choose]
        res.append(tmp)
        index += window_step
    return res

def load_data(path):
    try:
        image_folder = os.path.join(path,'image')
        if not os.path.isdir(image_folder):
            image_folder = os.path.join(path,'images')

        mask_folder = os.path.join(path,'masks')
        if not os.path.isdir(mask_folder):
            mask_folder = os.path.join(path,'Mask')

        depth_folder =  os.path.join(path,'depth')
        if not os.path.isdir(depth_folder):
            depth_folder = os.path.join(path,'depths')
            if not os.path.isdir(depth_folder):
                depth_folder = os.path.join(path,'world_depth')
        images = jhelp_file(image_folder)
        masks = jhelp_file(mask_folder) if os.path.isdir(mask_folder) else None
        depths  = jhelp_file(depth_folder)
    except:
        raise ImportError('error input folder, need IMAGES and DEPTHS (MASKS) folder!')
    return images,masks,depths

def check_stero_mode(root):
    """
    在 root 下递归查找所有同时包含 'left' 和 'right' 子文件夹的目录，
    返回列表，每个元素是 (left_path, right_path) 的元组
    """
    result = []
    for dirpath, dirnames, _ in os.walk(root):
        left_dirs = [d for d in dirnames if "left" in d.lower()]
        right_dirs = [d for d in dirnames if "right" in d.lower()]
        if left_dirs and right_dirs:
            for l in left_dirs:
                for r in right_dirs:
                    result.append((
                        os.path.join(dirpath, l),
                        os.path.join(dirpath, r)
                    ))
    if len(result) == 1:
        txt_files = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(".txt"):
                    txt_files.append(os.path.join(dirpath, f))
        return result[0],sorted(txt_files)
    else:
        return None


def find_all_txt_files(root):
    """
    在 root 下递归查找所有 .txt 文件，返回完整路径列表
    """
    txt_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".txt"):
                txt_files.append(os.path.join(dirpath, f))
    return txt_files

def stereo_renaming(l_images, keyword="left"):
    """
    在路径包含 `keyword` 的目录下，将文件名改为 name_keyword.ext
    但若路径中包含 'pointcloud' 则跳过；
    若文件名已带 '_keyword' 后缀（不区分大小写）也跳过。
    返回重命名后的路径列表。
    """
    renamed = []
    k_lower = keyword.lower()
    activate = False

    for path in l_images:
        parts_lower = [p.lower() for p in path.split(os.sep)]

        # 1) 跳过 pointcloud
        if "pointcloud" in parts_lower:
            renamed.append(path)
            # print(f"[SKIP pointcloud] {path}")
            continue

        # 2) 仅当路径中包含 keyword 的目录时才考虑改名
        if k_lower in parts_lower:
            folder, filename = os.path.split(path)
            name, ext = os.path.splitext(filename)

            # 2a) 已经命名过：xxx_left.ext → 跳过
            if name.lower().endswith(f"_{k_lower}"):
                renamed.append(path)
                # print(f"[SKIP already tagged] {path}")
                continue

            new_name = f"{name}_{keyword}{ext}"
            new_path = os.path.join(folder, new_name)

            # 2b) 目标已存在，避免覆盖（选择跳过并提示）
            if os.path.exists(new_path):
                renamed.append(path)
                # print(f"[SKIP exists] {new_path} already exists. Source kept: {path}")
                continue

            # 真正重命名
            os.rename(path, new_path)
            renamed.append(new_path)
            activate=True
            # print(f"[RENAME] {path}  ->  {new_path}")
        else:
            # 路径不含 keyword 的目录，保持不变
            renamed.append(path)

    return activate


if __name__ == '__main__':
    args = init_param()
    args.f = True
    # rtf = '/home/rg0775/QingHong/data/plytestdata/fg_avatar_0725/0729_3frames/raw/6DoF.rtf'
    # path = '/home/rg0775/QingHong/data/plytestdata/fg_avatar_0725/0729_3frames/raw'
    path = args.root
    task_indexes = []
    curs = []
    stereo_mode = False
    # 0827新增stereo方式
    stereo_data = check_stero_mode(path)
    if stereo_data is not None:
        stereo_mode=True
        print('using stereo mode')
        img_path,txt_files = stereo_data
        args.fbx = False
        if True: #render one mode
            l_images,l_masks,l_depths = load_data(img_path[0])
            r_images,r_masks,r_depths = load_data(img_path[1])
            activate = False
            activate |=stereo_renaming(l_images,'left')
            activate |=stereo_renaming(l_masks,'left')
            activate |=stereo_renaming(l_depths,'left')
            activate |=stereo_renaming(r_images,'right')
            activate |=stereo_renaming(r_masks,'right')
            activate |=stereo_renaming(r_depths,'right')
            if activate:
                stereo_data = check_stero_mode(path)#reload
                img_path,txt_files = stereo_data 
                l_images,l_masks,l_depths = load_data(img_path[0])
                r_images,r_masks,r_depths = load_data(img_path[1])

            intrinsic_file = [f for f in txt_files if 'intrinsic' in f.lower()]
            instrinsics = read_intrinsic(intrinsic_file[0])
            txt_files = [f for f in txt_files if 'intrinsic' not in f.lower()]
            images_prepare,masks_prepare,depths_prepare,extrinsics = [],[],[],[]
            for index in range(len(txt_files)):
                txt = txt_files[index]
                name = os.path.basename(txt)
                max_frame = len(read_txt(txt))//2
                tmp_ext = read_extrinsics(txt)
                #最后k的数据倒叙排列
                ext_ = tmp_ext[:max_frame]
                ext_r = tmp_ext[-max_frame:][::-1]
                images_prepare.append(l_images[index:index+max_frame] + r_images[index:index+max_frame])
                masks_prepare.append(l_masks[index:index+max_frame] + r_masks[index:index+max_frame])
                depths_prepare.append(l_depths[index:index+max_frame] + r_depths[index:index+max_frame])
                extrinsics.append(ext_+ext_r)
                curs.append((max_frame+1)//2)

            args.step = 1
            args.max_frame = max_frame
            for i in tqdm(range(len(images_prepare)),desc=os.path.basename(os.path.abspath(os.path.join(path,'..')))):
                args.cur = curs[i]
                # curname = '{:0>4}'.format(i)
                name0 = os.path.splitext(os.path.basename(images_prepare[i][0]))[0]
                name1 = os.path.splitext(os.path.basename(images_prepare[i][args.max_frame-1]))[0]
                name = f'stereo_from_{name0}_to_{name1}_{args.mask_type}'
                if args.step!=1:
                    name += f'_step_{args.step}'
                name += f'_cur_{args.cur}'
                save_path = os.path.join(path,'..','pointcloud',name)
                m = masks_prepare[i] if l_masks is not None else None
                ply_cal_core(images_prepare[i],depths_prepare[i],instrinsics,extrinsics[i],save_path,args,m)
                if args.test:
                    break
        else: #render all mode
            pass 
        
    else:
        if os.path.basename(args.root) != 'raw': #防呆码
            tmps = prune(jhelp(args.root),'raw')
            if not os.path.isdir(os.path.join(args.root,'raw')):
                mkdir(os.path.join(args.root,'raw'))
                for tmp in tmps:
                    shutil.move(tmp,os.path.join(args.root,'raw',os.path.basename(tmp)))
            path = os.path.join(args.root,'raw')

        images,masks,depths = load_data(path)
        if args.mask_type != 'nomask':
            assert len(masks)>0,'can not find mask file!'
        
        if len(gofind(jhelp_file(path),'.fbx'))>0: #fbx mode
            from fbx2json import fbx_reader
            instrinsics = {}
            instrinsics['h'],instrinsics['w'] = read(images[0],type='image').shape[:2]
            fbx_file = gofind(jhelp_file(path),'.fbx')[0]
            ext_,[fw,fh] = fbx_reader(fbx_file)
            focal_length_x = instrinsics['w']  * fw
            focal_length_y = instrinsics['h']  * fh
            instrinsics['fx'],instrinsics['fy'] = focal_length_x,focal_length_y
            args.fbx=True
        else: #txt mode
            intrinsic_file = gofind(jhelp_file(path),'intrinsic.txt')[0]
            extrinsic_file = gofind(jhelp_file(path),'6DoF.txt')[0]
            instrinsics = read_intrinsic(intrinsic_file)
            ext_ = read_extrinsics(extrinsic_file)
            args.fbx=False

        for i in range(len(images)): #prepare data
            cur = args.max_frame//2
            tmp = i+(np.arange(args.max_frame)-args.max_frame//2)*args.step
            while(tmp.min()<0):
                tmp +=args.step
                cur -= 1
                if cur < 0 or cur >= args.max_frame:
                    raise ValueError('error max frames')
            while(tmp.max()>len(images)-1):
                tmp -=args.step
                cur += 1
                if cur < 0 or cur >= args.max_frame:
                    raise ValueError('error max frames')
            
            tmp = [np.clip(k,0,len(images)-1) for k in tmp]
            task_indexes.append(tmp)
            curs.append(cur)

        if len(args.custom)>1:
            custom = args.custom
            sorted_arr_with_indices = sorted(enumerate(custom), key=lambda x: x[1])
            sorted_arr = [x[1] for x in sorted_arr_with_indices]
            task_indexes = [sorted_arr]
            original_positions = {original_idx: sorted_idx for sorted_idx, (original_idx, _) in enumerate(sorted_arr_with_indices)}
            curs = [original_positions[0]]
        
        if len(images) <= args.max_frame:
            args.step = 1
            images_prepare = [[images[i] for i in range(0,len(images))]]
            masks_prepare = [[masks[i] for i in range(0,len(masks))]] if masks is not None else None
            depths_prepare = [[depths[i] for i in range(0,len(depths))]]
            extrinsics = [ext_]
        else:
            images_prepare = [[images[ff] for ff in task_indexes[f]] for f in range(len(task_indexes))]
            masks_prepare = [[masks[ff] for ff in task_indexes[f]] for f in range(len(task_indexes))] if masks is not None else None
            depths_prepare = [[depths[ff] for ff in task_indexes[f]] for f in range(len(task_indexes))]
            extrinsics = [[ext_[ff] for ff in task_indexes[f]] for f in range(len(task_indexes))]
       
    
        for i in tqdm(range(len(images_prepare)),desc=os.path.basename(os.path.abspath(os.path.join(path,'..')))):
            args.cur = curs[i]
            curname = os.path.splitext(os.path.basename(images[i]))[0]
            # curname = '{:0>4}'.format(i)
            name0 = os.path.splitext(os.path.basename(images_prepare[i][0]))[0]
            name1 = os.path.splitext(os.path.basename(images_prepare[i][-1]))[0]
            name = f'{curname}_from_{name0}_to_{name1}_{args.mask_type}'
            if args.step!=1:
                name += f'_step_{args.step}'
            name += f'_cur_{args.cur}'
            save_path = os.path.join(path,'..','pointcloud',name)
            m = masks_prepare[i] if masks is not None else None
            ply_cal_core(images_prepare[i],depths_prepare[i],instrinsics,extrinsics[i],save_path,args,m)
            if args.test:
                break
    if args.test:
        plypath = os.path.join(save_path,'sparse/0/points3D.ply')
        from plytest import show_ply
        show_ply(plypath)

        # if i < len(images_prepare):
        #     if i == len(images_prepare)-1:
        #         image_infos = [ImageInfo(uid=i,extrinsic=source_ext[i],rub=None)]
        #         cam_infos = [CameraInfo(uid=i, fx=float(source_ins[i]['fx']),fy=float(source_ins[i]['fy']),cx=int(source_ins[i]['w'])/2.0 ,cy=int(source_ins[i]['h'])/2.0,image_name=os.path.basename(images[i]),image_path = images[i], width=int(source_ins[i]['w']), height=int(source_ins[i]['h']),model="PINHOLE")]
        #     else:
        #         image_infos = [ImageInfo(uid=i,extrinsic=source_ext[i],rub=None),ImageInfo(uid=i+1,extrinsic=source_ext[i+1],rub=None)]
        #         cam_infos = [CameraInfo(uid=i, fx=float(source_ins[i]['fx']),fy=float(source_ins[i]['fy']),cx=int(source_ins[i]['w'])/2.0 ,cy=int(source_ins[i]['h'])/2.0,image_name=os.path.basename(images[i]),image_path = images[i], width=int(source_ins[i]['w']), height=int(source_ins[i]['h']),model="PINHOLE"),CameraInfo(uid=i+1, fx=float(source_ins[i+1]['fx']),fy=float(source_ins[i+1]['fy']),cx=int(source_ins[i+1]['w'])/2.0 ,cy=int(source_ins[i+1]['h'])/2.0,image_name=os.path.basename(images[i]),image_path = images[i], width=int(source_ins[i+1]['w']), height=int(source_ins[i+1]['h']),model="PINHOLE")]
        #     write_colmap_model(os.path.join(save_path,'sparse/0'),cam_infos,image_infos,'.jatxt')

        # HEADER = (f'extra information for judder_angle renders,cf={str(i)}')
        # np.savetxt(os.path.join(save_path,'sparse/0/ja_images.txt'),source_ext[i:i+2],header=HEADER)
        # np.savetxt(os.path.join(save_path,'sparse/0/ja_cameras.txt'),source_ins[i:i+2])
    print('finished')
    