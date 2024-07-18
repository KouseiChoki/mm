'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-07-04 14:57:49
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
import numpy as np
import math
from typing import NamedTuple
import cv2
import shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/../../')
from file_utils import read,mvwrite
import torch
from einops import einsum,rearrange
from plyfile import PlyData, PlyElement
from fileutil.read_write_model import Camera,write_model,Image,rotmat2qvec
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from dataclasses import dataclass
@dataclass
class CameraInfo():
    uid: int
    fx:float
    fy:float
    cx:float
    cy:float
    image_name: str
    image_path: str
    width: int
    height: int
    model:str
@dataclass
class ImageInfo():
    uid:int
    extrinsic:np.array


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))

def check_chardet(file):
    import chardet
     # 读取文件的前1024字节来检测编码
    with open(file, 'rb') as f:
        raw_data = f.read(1024)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    return encoding


def generate_point_cloud_from_depth(depth_image, intrinsics, extrinsics):
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
    points_camera = points_camera[points_camera[:, 2] != 0]
    # 将点云从相机坐标系转换到世界坐标系
    points_world = (extrinsics[:3, :3] @ points_camera.T).T + extrinsics[:3, 3]
    return points_world

def sample_image_grid(shape,device):
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""
    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)
    return coordinates, stacked_indices

def homogenize_points(points):
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def unproject(coordinates,z,intrinsics,):
    """Unproject 2D camera coordinates with the given Z values."""
    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.inverse(), coordinates, "... i j, ... j -> ... i"
    )

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


def write_ply(path,xyz,rgb):
    # Adapted from https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/dataset_readers.py#L115
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
    print('writing plyfile........')
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    
def extract_euler_angles_from_view_matrix(view_matrix):
    R = view_matrix[0:3, 0:3]
    if R[2, 0] != 1 and R[2, 0] != -1:
        yaw = np.arcsin(-R[2, 0])
        pitch = np.arctan2(R[2, 1] / np.cos(yaw), R[2, 2] / np.cos(yaw))
        roll = np.arctan2(R[1, 0] / np.cos(yaw), R[0, 0] / np.cos(yaw))
    else:
        roll = 0
        if R[2, 0] == -1:
            yaw = np.pi / 2
            pitch = np.arctan2(R[0, 1], R[0, 2])
        else:
            yaw = -np.pi / 2
            pitch = np.arctan2(-R[0, 1], -R[0, 2])
    return pitch, yaw, roll

def get_camdata_from_tcdump(root,w=1920,h=1080,down_scale=6,step=1,max_step=99999,model='SIMPLE_PINHOLE'):
    # root = '/Users/qhong/Desktop/0624/output'
    depths = jhelp_file(os.path.join(root,'depth'))
    metas = jhelp_file(os.path.join(root,'meta'))
    videos = jhelp_file(os.path.join(root,'video'))
    assert len(depths)==len(metas)==len(videos),'data error,please check your image or depth,metas'
    nums = len(metas)
    # nums = 5 #for test
    index = 1
    down_scale_x = down_scale
    down_scale_y = down_scale
    cam_infos,image_infos,points,rgbs = [],[],[],[]
    for i in tqdm(range(0,nums,step),desc=f'reading {os.path.basename(root)}'):
        #image
        if index>=max_step:
            break
        image_file = videos[i]
        image = read_rgba(image_file,w,h,np.half) if '.rgba' in os.path.basename(image_file) else read(image_file,type='image')
        rgb = image.astype('float32')/255
        meta_file = metas[i]
        intrinsic,extrinsic = read_metas(meta_file)
        tx,ty,tz = extrinsic[:3,-1]/10
        #  x x -tx
        # print(extrinsic)
        extrinsic[:3,-1] = [0,0,-tx] #down 2
        print(extrinsic[:3,-1])
        #extrinsic #np.set_printoptions(precision=3, suppress=True)
        w2c = np.linalg.inv(extrinsic)
        # w2c,extrinsic = extrinsic,w2c
        qx, qy, qz ,qw = R.from_matrix(w2c[:3, :3]).as_quat()
        tvec = w2c[:3, 3]
        image_info = ImageInfo(uid=index,extrinsic=np.array([qw,qx,qy,qz,tvec[0],tvec[1],tvec[2]]))
        image_infos.append(image_info)
        focal_length_x = intrinsic[0,0] #fx
        focal_length_y = intrinsic[1,1] #fy
        o_cx = intrinsic[0,2]
        o_cy = intrinsic[1,2]
        cam_info = CameraInfo(uid=index, fx=focal_length_x,fy=focal_length_y,cx=o_cx,cy=o_cy,image_name=os.path.basename(image_file).replace('.png',''),image_path = image_file, width=w, height=h,model=model)
        cam_infos.append(cam_info)
        #downscale
        focal_length_x = focal_length_x/down_scale_x
        focal_length_y = focal_length_y/down_scale_y
        o_cx = o_cx/down_scale_x
        o_cy = o_cy/down_scale_y
        target_w = w//down_scale_x
        target_h = h//down_scale_y
        depth_file = depths[i]
        depth = read_depths(depth_file,target_w,target_h)
        # depth = cv2.resize(depth,(target_w,target_h),interpolation=cv2.INTER_NEAREST) 苹果相机不用6倍下采样 本身就是
        #内参做了归一化
        intrinsics = np.array([[focal_length_x,0,o_cx],[0,focal_length_y,o_cy],[0,0,1]])
        point = generate_point_cloud_from_depth(depth,intrinsics,extrinsic)
        #prune unvailid point
        rgb = cv2.resize(image,(target_w,target_h))
        #删除非法depth
        rgb = rgb[depth!= 0]
        points.append(point.reshape(-1,3))
        rgbs.append(rgb.reshape(-1,3))
        index += 1
        # if True: #for test
        #     depth = np.repeat(depth[...,None],4,axis=2)
        #     d = depth[...,0]
        #     d = (d - d.min()) / (d.max() - d.min())
        #     depth[...,-1] = d
        #     mvwrite(f'/Users/qhong/Desktop/0704/depthtest/{i}.exr',depth,precision='half')
    points = np.concatenate(points)
    rgbs = np.concatenate(rgbs)
    return image_infos,cam_infos,points,rgbs



def tc_reader(root,save_path,step,max_step=99999,save_as_rgb=True):
    sp = os.path.join(save_path,'pointcloud')
    sparse_path = os.path.join(sp,'sparse/0')
    ply_path = os.path.join(sparse_path , "points3D.ply")
    image_infos,cam_infos,xyz,rgbs = get_camdata_from_tcdump(root,w=1920,h=1080,down_scale=6,step=step,max_step=max_step)
    mkdir(sparse_path)
    write_ply(ply_path, xyz,rgbs)
    # Write out the images.
    mkdir(os.path.join(sp , "images"))
    for cam_info in cam_infos:
        if save_as_rgb:
            image = read_rgba(cam_info.image_path,cam_info.width,cam_info.height,np.half)
            mvwrite(os.path.join(cam_info.image_path , sp , "images",cam_info.image_name)+'.png',image)
        else:
            shutil.copy(cam_info.image_path, os.path.join(cam_info.image_path , sp , "images",cam_info.image_full_name))
    # Write out the camera parameters.
    write_colmap_model(sparse_path,cam_infos,image_infos)
    
         
def read_depths(file,w,h):
    # /Users/qhong/Desktop/0624/output/depth/3632FF65-0F14-4A05-8ABF-DD6226F934AB_00000001.dat
    with open(file, 'rb') as f:
        depth_data = np.frombuffer(f.read(), dtype='float32').reshape((int(h),int(w)))
    return depth_data

def read_rgba(file,w,h,dtype):
    with open(file, 'rb') as f:
        raw_data = f.read()
    
    # 将二进制数据转换为 NumPy 数组
    img_data = np.frombuffer(raw_data, dtype=dtype)
    
    # 确保数据长度正确
    assert len(img_data) == w * h * 4, f"文件大小与图像分辨率不匹配。{len(img_data)} !={w * h * 4} "
    
    # 将数据转换为指定分辨率的图像
    img_data = img_data.reshape((h, w, 4))
    
    # 缩放到 8 位 (uint8)
    img_data_8bit = (img_data *255).astype(np.uint8)[...,:3]

    # from PIL import Image
    # Image.fromarray(img_data_8bit)
    return img_data_8bit
        
def read_metas(file):
    # file = '/Users/qhong/Desktop/0624/output/meta/3632FF65-0F14-4A05-8ABF-DD6226F934AB_00000004.txt'
    with open(file,'r',encoding='utf-8') as f:
            lines = f.readlines()
    intrinsic_matrix = None
    extrinsic_matrix = None
    # Read through the file to find the matrices
    for i, line in enumerate(lines):
        if '[intrinsic]' in line:
            intrinsic_matrix = np.array([float(x) for x in lines[i+1].split()]).reshape((3, 3))
        elif '[view matrix]' in line:
            extrinsic_matrix = np.array([float(x) for x in (lines[i+1]).split()]).reshape((4, 4)).transpose()
    return intrinsic_matrix,extrinsic_matrix

def write_colmap_model(path,cam_infos,image_infos):
    # Define the cameras (intrinsics).
    cameras = {}
    images = {}
    for cam_info,image_info in zip(cam_infos,image_infos):
        assert cam_info.uid == image_info.uid
        qw,qx,qy,qz,tx,ty,tz = image_info.extrinsic
        cameras[cam_info.uid] = Camera(cam_info.uid, 'PINHOLE', cam_info.width, cam_info.height, (cam_info.fx, cam_info.fy, cam_info.cx, cam_info.cy))
        qvec = np.array((qw, qx, qy, qz))
        tvec = np.array((tx, ty, tz))
        images[cam_info.uid] = Image(cam_info.uid, qvec, tvec, cam_info.uid, cam_info.image_name+'.png', [], [])
    mkdir(path)
    write_model(cameras, images, None, path,ext='.txt')

if __name__ == "__main__":
    import open3d as o3d
    root = '/Users/qhong/Desktop/0704/down2'
    sp = root+'_result'
    tc_reader(root,sp,step=20,max_step=999,save_as_rgb=True)
    pcd = o3d.io.read_point_cloud(f'{sp}/pointcloud/sparse/0/points3D.ply')
    # 创建坐标系
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    # pcd = o3d.io.read_point_cloud('/Users/qhong/Downloads/points3D(4).ply')
    colors = pcd.colors
    # print("前几个点的颜色信息:")
    # for i in range(min(10, len(colors))):  # 打印前10个点的颜色信息
    #     print(f"点 {i} 颜色: {colors[i]}")
    o3d.visualization.draw_geometries([pcd,axis])