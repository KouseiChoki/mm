'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-08-08 15:11:07
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
from tqdm import tqdm
import numpy as np 
import cv2
import shutil
import OpenEXR, Imath, array
import os,sys
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/../..')
from tqdm.contrib.concurrent import process_map
import warnings
import re
import argparse
import Imath,math
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from fileutil.read_write_model import Camera,write_model,Image
from plyfile import PlyData, PlyElement
from file_utils import mvwrite,read
from conversion_tools.exr_processing.color_convertion.colorutil import Color_transform
from copy import deepcopy
import re
acescg_to_rec709 = Color_transform('acescg','lin_rec709')
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

type_dict = {"PW_PRM_BT601"              : [  0.640, 0.330, 0.290, 0.600, 0.150, 0.060  ],
    "rec709"              : [  0.640, 0.330, 0.300, 0.600, 0.150, 0.060  ],
    "PW_PRM_DCI_P3"             : [  0.680, 0.320, 0.265, 0.690, 0.150, 0.060  ],
    "PW_PRM_BT2020"             : [  0.708, 0.292, 0.170, 0.797, 0.131, 0.046  ],
    "PW_PRM_ARRI_WG"            : [  0.684, 0.313, 0.211, 0.848, 0.0861, -0.102],
    "PW_PRM_ACES_AP0"           : [  0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.0770 ],
    "PW_PRM_ACES_AP1"           : [  0.713, 0.293, 0.165, 0.83, 0.128, 0.0440  ],
    "PW_PRM_CINITY"             : [  0.705, 0.2872,0.1205,0.8029,0.1557,0.0288 ],
    "PW_PRM_GAMUT3"             : [  0.730, 0.280, 0.140, 0.855, 0.100, -0.050 ],
    "PW_PRM_GAMUT3_CINE"        : [  0.766, 0.275, 0.225, 0.800, 0.089, -0.087 ],
    "PW_PRM_UNSPECIFIED"        : [  0.708, 0.292, 0.170, 0.797, 0.131, 0.046  ]}

def init_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',  help="your data path", required=True)
    parser.add_argument('--down_scale',type=int, default=6,help="down scale rate")
    parser.add_argument('--step',type=int, default=1,help="frame step")
    parser.add_argument('--start_frame',type=int, default=0,help="start frame")
    parser.add_argument('--max_frame',type=int, default=999,help="max frame")
    parser.add_argument('--max_depth',type=int, default=1000,help="depth range (meters)")
    parser.add_argument('--f', action='store_true', help="force run")
    parser.add_argument('--rp', action='store_true', help="reverse pitch")
    parser.add_argument('--ry', action='store_true', help="reverse yaw")
    parser.add_argument('--rr', action='store_true', help="reverse roll")
    parser.add_argument('--mask', action='store_true', help="use mask")
    parser.add_argument('--judder_angle',type=int, default=-1,help="frame step")
    parser.add_argument('--final_image', action='store_true', help="use final image")
    parser.add_argument('--extra_depth', action='store_true', help="use extra_depth")
    parser.add_argument('--test', action='store_true', help="use test")
    args = parser.parse_args()
    return args

def GetViewMatrixFromEularAngle(pitch, yaw, roll):
    viewMat_Yaw = GetViewMatrixFromEular(0,yaw,0)
    viewMat_Pitch = GetViewMatrixFromEular(pitch,0,0)
    viewMat_Roll = GetViewMatrixFromEular(0,0,roll)
    return np.dot(np.dot(viewMat_Yaw,viewMat_Roll),viewMat_Pitch)


def GetViewMatrixFromEular(pitch, yaw, roll):
    half_yaw = D2R * yaw
    half_pitch = D2R * pitch
    half_roll = D2R * roll
    SP = np.sin(half_pitch)
    SY = np.sin(half_yaw)
    SR = np.sin(half_roll)
    CP = np.cos(half_pitch)
    CY = np.cos(half_yaw)
    CR = np.cos(half_roll)
    # Column first
    viewMat_T = np.zeros(16, dtype=np.float32)
    viewMat_T[0] = CP * CY
    viewMat_T[1] = CP * SY
    viewMat_T[2] = SP
    viewMat_T[3] = 0
    viewMat_T[4] = SR * SP * CY - CR * SY
    viewMat_T[5] = SR * SP * SY + CR * CY
    viewMat_T[6] = -SR * CP
    viewMat_T[7] = 0
    viewMat_T[8] = -(CR * SP * CY + SR * SY)
    viewMat_T[9] = CY * SR - CR * SP * SY
    viewMat_T[10] = CR * CP
    viewMat_T[11] = 0
    viewMat_T[12] = 0  # was pos_x, but commented out in the original code
    viewMat_T[13] = 0  # was pos_y
    viewMat_T[14] = 0  # was pos_z
    viewMat_T[15] = 1
    # Transpose to make it row first
    return viewMat_T.reshape((4,4)).T

warnings.filterwarnings("ignore")
pt = Imath.PixelType(Imath.PixelType.FLOAT)
D2R  =  np.pi/180
'''
description: 读取exr文件
param {*} filePath 文件地址
return {*} 返回rgb图像，mv1，mask
'''
def get_channel_data(img_exr,keyword,type='f'):
    data = np.array(array.array(type, img_exr.channels(keyword,pt)))
    return data
def mkdir(path):
        if  not os.path.exists(path):
            os.makedirs(path,exist_ok=True)

def find_folders_with_subfolder(root_path, keys = [], path_keys = [] ,excs = [] ,path_excs =[]):
    """
    Find all folders in the root_path that contain a subfolder with the name subfolder_name.
    """
    folders_with_subfolder = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Check if the subfolder_name is in the list of directories
        flag = True
        for key in keys:
            if key not in dirnames:
                flag = False
        for path_key in path_keys:
            if path_key not in dirpath:
                flag = False
        for exc in excs:
            if exc in dirnames:
                flag = False
        for exc in path_excs:
            if exc in dirpath:
                flag = False
        if flag:
            folders_with_subfolder.append(dirpath)

    return folders_with_subfolder
def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))
def prune(c,keyword,mode = 'basename'):
    if mode =='basename':
        res = list(filter(lambda x:keyword.lower() not in os.path.basename(x).lower(),c)) 
    else:
        res = list(filter(lambda x:keyword.lower() not in x.lower(),c))
    return res 

def gofind(c,keyword,mode = 'basename'):
    if mode =='basename':
        res = list(filter(lambda x:keyword.lower() in os.path.basename(x).lower(),c)) 
    else:
        res = list(filter(lambda x:keyword.lower() in x.lower(),c)) 
    return res 

def custom_refine(flow,zero_to_one=True):
    height,width = flow[...,0].shape
    #average value
    if zero_to_one:
        flow[...,0]/=width
        flow[...,1]/=-height
    else:
        flow[...,0]/=-width
        flow[...,1]/=height
    # if flow.shape[2] >= 3:
    #     flow[...,2] /= 65535 #65535*255
    return flow


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def RightLeftAxisChange(mat):
    M = np.eye(mat.shape[0])
    M[1,1] = -1
    return M.dot(mat).dot(M)

def eulerAngles2rotationMat(theta, loc = [], format='degree', order = 'ZYX',axis='left'):
    """
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return:
    RPY角，是ZYX欧拉角，依次 绕定轴XYZ转动[rx, ry, rz]
    """
    if format == 'degree':
        theta = [i * math.pi / 180.0 for i in theta]
    if axis == 'right':
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])
    else:
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), math.sin(theta[0])],
                        [0, -math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, -math.sin(theta[1])],
                        [0, 1, 0],
                        [math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), math.sin(theta[2]), 0],
                        [-math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])
    # R = np.dot(R_z, np.dot(R_y, R_x))
    if order == 'ZYX':
        R = np.dot(R_x, np.dot(R_y, R_z))
    elif order == 'ZXY':
        R = np.dot(R_y, np.dot(R_x, R_z))
    elif order == 'XYZ':
        R = np.dot(R_z, np.dot(R_y, R_x))
    elif order == 'XZY':
        R = np.dot(R_y, np.dot(R_z, R_x))
    elif order == 'YXZ':
        R = np.dot(R_z, np.dot(R_x, R_y))
    elif order == 'YZX':
        R = np.dot(R_x, np.dot(R_z, R_y))
    if loc.__len__() > 0:
        ans = np.eye(4)
        ans[0:3,0:3] = R
        ans[0:3, -1] = loc
    else:
        ans = R
    if axis == 'left':
        ans = RightLeftAxisChange(ans)
    return ans

         
def check_type(exr): #not finished, only check acescg and rec709 now 2024/04/29
    dtype = 'rec709'
    if 'chromaticities' in exr.header():
        chro = exr.header()['chromaticities']
        check = [chro.red.x,chro.red.y,chro.green.x,chro.green.y,chro.blue.x,chro.blue.y]
        # ,chro.white.x,chro.white.y]
        check_two = [round(i,3) for i in check]
        loss = 1e9
        for item,key in type_dict.items():
            loss_ = sum([abs(a-b) for a,b in zip(check_two,key)])
            if loss_<loss:
                loss = loss_
                dtype = item
    elif 'unreal/colorSpace/destination' in exr.header():
        if 'acescg' in str(exr.header()['unreal/colorSpace/destination'].lower()):
            dtype = 'acescg'
    assert dtype.lower() in ['rec709','acescg'],f'not supported algorithm {dtype}'
    return dtype
   
def read_exr(filePath):
    img_exr = OpenEXR.InputFile(filePath)
    dw = img_exr.header()['dataWindow']
    size = (dw.max.y - dw.min.y + 1,dw.max.x - dw.min.x + 1)
    for key in img_exr.header()['channels']:
        if 'FinalImagePWWorldDepth.R' in key:
            depth_R = key
        elif 'FinalImageMovieRenderQueue_WorldDepth.R' in key:
            depth_R = key
        if 'FinalImagePWMask.R' in key:
            fnmask_R = key
    if depth_R is None:
        worldpos = None
    else:
        worldpos = np.array(array.array('f', img_exr.channel(depth_R,pt))).reshape(size)
    data = {}
    camera_type = 1
    for ic in img_exr.header().keys():
        if '/focalLength' in ic:
            camera_type = 0
            data['focal_length'] = float(img_exr.header()['unreal/camera/FinalImage/focalLength'])
            break
    data['sensor_w'] = float(img_exr.header()['unreal/camera/FinalImage/sensorWidth'])
    data['sensor_h'] = float(img_exr.header()['unreal/camera/FinalImage/sensorHeight'])
    data['fov'] = float(img_exr.header()['unreal/camera/FinalImage/fov'])
    data['camera_type'] = camera_type
    data['h'],data['w'] = size
    data_cur = data.copy()
    data_cur['x'] = float(img_exr.header()['unreal/camera/curPos/x'])
    data_cur['y'] = float(img_exr.header()['unreal/camera/curPos/y'])
    data_cur['z'] = float(img_exr.header()['unreal/camera/curPos/z'])
    data_cur['pitch'] = float(img_exr.header()['unreal/camera/curRot/pitch'])
    data_cur['roll'] = float(img_exr.header()['unreal/camera/curRot/roll'])
    data_cur['yaw'] = float(img_exr.header()['unreal/camera/curRot/yaw'])
    # mask = np.array(array.array('f', img_exr.channel(fnmask_R,pt))).reshape(size) if fnmask_R is not None else None
    r_str, g_str, b_str = img_exr.channels('RGB',pt)
    red = np.array(array.array('f', r_str)).reshape(size)
    green = np.array(array.array('f', g_str)).reshape(size)
    blue = np.array(array.array('f', b_str)).reshape(size)
    image = np.stack([red,green,blue],axis=2).astype('float32')
    if check_type(img_exr)=='acescg':
        image = acescg_to_rec709.apply(image)
    image = hdr_to_rgb(image)[...,:3]
    return image,data_cur,worldpos


def adjust(res):
    # res[...,-1] *= 0
    res[...,0] /= res.shape[1]
    res[...,1] /= res.shape[0]
    # prune 3rd channel mv_z
    res[...,2]*=0
    return res

def hdr_to_rgb(hdr_image):
    # 对HDR图像进行色调映射
    # tonemap = cv2.createTonemapReinhard(1.0, 0, 0, 0)
    # ldr_image = tonemap.process(np.ascontiguousarray(hdr_image.copy()[...,:3]))
    # 将[0, 1]范围的图像转换为[0, 255]
    ldr_image_8bit = np.clip(hdr_image * 255, 0, 255).astype('uint8')
    ldr_image = adjust_gamma(ldr_image_8bit)
    # 保存转换后的图像
    return ldr_image[...,:3]

def adjust_gamma(image, gamma=2.4):
    # 建立一个映射表
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    
    # 应用gamma校正使用查找表
    return cv2.LUT(image, table)

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
        images[cam_info.uid] = Image(cam_info.uid, qvec, tvec, cam_info.uid, cam_info.image_name, [], [])
    mkdir(path)
    write_model(cameras, images, None, path,ext='.txt')

def slerp(q1, q2, t):
    """Perform spherical linear interpolation (slerp) between two quaternions."""
    # Convert to numpy arrays
    q1 = np.array(q1)
    q2 = np.array(q2)
    
    # Compute the cosine of the angle between the two vectors.
    dot_product = np.dot(q1, q2)
    
    # If the dot product is negative, slerp won't take the shorter path.
    # Note that q and -q represent the same rotation, but may produce different slerp.
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product
    
    # Clamp dot_product to be in the range of [0.0, 1.0]
    dot_product = np.clip(dot_product, 0.0, 1.0)
    
    # Calculate the angle between the quaternions
    theta_0 = np.arccos(dot_product)  # theta_0 = angle between input vectors
    sin_theta_0 = np.sin(theta_0)     # compute this value only once
    
    # If the angle is small, use linear interpolation
    if sin_theta_0 < 1e-6:
        return (1.0 - t) * q1 + t * q2
    
    # Compute the actual interpolation factor
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot_product * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return (s0 * q1) + (s1 * q2)

def interpolate_translation(t1, t2, t=0.5):
    """Perform linear interpolation between two translations."""
    return (1 - t) * np.array(t1) + t * np.array(t2)
# --path /Users/qhong/Desktop/0607/FtGothicCastle_04 --judder_angle=360 --max_frame=10
def plus_name(name):
    num = re.findall(r'\d+\.png', name)[-1].replace('.png','')
    # int_num = int(num) + 1
    # new_str_num = f'{int_num:len(num)}'
def ja_ajust(image_infos,cam_infos,judder_angle):
    ja =  judder_angle/720
    num = len(image_infos)
    new_image_infos,new_cam_infos = [image_infos[0]],[cam_infos[0]]
    basename = cam_infos[0].image_name
    basename = basename.replace(re.findall(r'\d+', basename)[-1],'')
    for i in range(1,num):
        image_info = image_infos[i]
        cam_info = cam_infos[i]
        prev_image_info = image_infos[i-1]
        prev_extrinsic = prev_image_info.extrinsic
        extrinsic = image_info.extrinsic
        new_q = slerp(prev_extrinsic[:4],extrinsic[:4],t=ja)
        new_t = interpolate_translation(prev_extrinsic[4:],extrinsic[4:],t=ja)
        #change uid and name
        new_image_info = ImageInfo(uid=image_info.uid*2-2,extrinsic=np.concatenate([new_q,new_t]))
        image_info.uid = image_info.uid*2-1
        new_cam_info = deepcopy(cam_info)
        new_cam_info.uid = new_cam_info.uid*2-2
        cam_info.uid = cam_info.uid*2-1
        new_cam_info.image_name = cam_info.image_name 
        # cam_info.image_name = f'{basename}{cam_info.uid-1:04}'
        # new_cam_info.image_name = f'{basename}{new_cam_info.uid-1:04}'

        new_image_infos.append(new_image_info)
        new_image_infos.append(image_info)
        new_cam_infos.append(new_cam_info)
        new_cam_infos.append(cam_info)
    return new_image_infos,new_cam_infos


def ply_cal_core(oris,save_path,args):
    name = os.path.basename(save_path)
    sp = os.path.join(save_path,'pointcloud')
    sparse_path = os.path.join(sp,'sparse/0')
    ply_path = os.path.join(sparse_path , "points3D.ply")
    if os.path.isdir(sp):
        if not args.f and args.judder_angle==-1:
            return
        else:
            shutil.rmtree(sp,ignore_errors=True)
    image_infos,cam_infos,xyz,rgbs = get_intrinsic_extrinsic(oris,save_path,name,args)
    mkdir(sparse_path)
    write_ply(ply_path, xyz,rgbs)
    # Write out the images.
    mkdir(os.path.join(sp , "images"))
    for cam_info in cam_infos:
        if args.final_image:
            shutil.copy(cam_info.image_path.replace('/image/','/final_image/'), os.path.join(sp , "images",os.path.basename(cam_info.image_path)))
        else:
            shutil.copy(cam_info.image_path, os.path.join(sp , "images",os.path.basename(cam_info.image_path)))
    # Write out the camera parameters.
    write_colmap_model(sparse_path,cam_infos,image_infos)
    if args.judder_angle!= -1:
        print('writing ja file')
        old_ply_path = ply_path
        image_infos,cam_infos = ja_ajust(image_infos,cam_infos,args.judder_angle)
        sp = os.path.join(save_path,f'pointcloud_ja_{args.judder_angle}')
        shutil.rmtree(sp,ignore_errors=True)
        sparse_path = os.path.join(sp,'sparse/0')
        mkdir(sparse_path)
        ply_path = os.path.join(sparse_path , "points3D.ply")
        shutil.copy(old_ply_path,ply_path)
        # Write out the images.
        mkdir(os.path.join(sp , "images"))
        for cam_info in cam_infos:
            if args.final_image:
                shutil.copy(cam_info.image_path.replace('/image/','/final_image/'), os.path.join(sp , "images",os.path.basename(cam_info.image_path)))
            else:
                shutil.copy(cam_info.image_path, os.path.join(sp , "images",os.path.basename(cam_info.image_path)))
        write_colmap_model(sparse_path,cam_infos,image_infos)

def get_intrinsic_extrinsic(oris,save_path,name,args):
    index = 1
    nums = len(oris)
    cam_infos,image_infos,points,rgbs = [],[],[],[]
    if args.extra_depth:
        extra_depths = jhelp_file(os.path.join(os.path.dirname(os.path.dirname(oris[0])),'extra_depth'))
    for i in tqdm(range(args.start_frame,nums,args.step),desc=f'processing {name}'): 
        if index>args.max_frame:
            break
        image,o_data,depth = read_exr(oris[i])
        if args.extra_depth:
            depth = read(extra_depths[i])[...,0]
        image_path = os.path.join(save_path,'image',os.path.basename(oris[i]).replace('.exr','.png'))
        if not os.path.isfile(image_path) or args.f:
            print(image.shape)
            mvwrite(image_path,image)
        w,h = o_data['w'],o_data['h']
        down_scale_x = args.down_scale
        down_scale_y = args.down_scale
        depth /= 100
        depth[np.where(depth>=args.max_depth)] = args.max_depth
        # Provided Euler angles
        pitch = o_data['pitch']
        roll  = o_data['roll']
        yaw   = o_data['yaw']
        tx,ty,tz = o_data['x']/100,o_data['y']/100,o_data['z']/100
        if args.rp:
            pitch *= -1
        if args.ry:
            yaw *= -1
        if args.rr:
            roll *= -1
        extrinsic = eulerAngles2rotationMat([-pitch,-yaw,roll], loc = [ty,tz,tx], format='degree', order = 'ZYX',axis='left')
        # extrinsic_ = GetViewMatrixFromEularAngle(pitch,-yaw,-pitch)
        # extrinsic_[0:3, -1] = [ty,tz,tx]
        # extrinsic_ = RightLeftAxisChange(extrinsic_)
        # print(pitch,yaw,roll)
        # if index == 1:
        #     print(extrinsic)
        #     print(extrinsic_)
        # extrinsic = extrinsic_
        w2c = np.linalg.inv(extrinsic)
        qx, qy, qz ,qw = R.from_matrix(w2c[:3, :3]).as_quat()
        tvec = w2c[:3, 3]
        image_info = ImageInfo(uid=index,extrinsic=np.array([qw,qx,qy,qz,tvec[0],tvec[1],tvec[2]]))
        image_infos.append(image_info)
        # intrinsic
        o_cx = w/2.0 
        o_cy = h/2.0
        model="PINHOLE"
        focal_length_x = w  * o_data['focal_length'] * 1 / o_data['sensor_w']
        focal_length_y = h  * o_data['focal_length'] * 1 / o_data['sensor_h']
        cam_info = CameraInfo(uid=index, fx=focal_length_x,fy=focal_length_y,cx=o_cx,cy=o_cy,image_name=os.path.basename(image_path),image_path = image_path, width=w, height=h,model=model)
        cam_infos.append(cam_info)
        #downscale
        focal_length_x = focal_length_x/down_scale_x
        focal_length_y = focal_length_y/down_scale_y
        o_cx = o_cx/down_scale_x
        o_cy = o_cy/down_scale_y
        target_w = w//down_scale_x
        target_h = h//down_scale_y
        depth = cv2.resize(depth,(target_w,target_h),interpolation=cv2.INTER_NEAREST)
        #prune unvalid depth
        # depth[np.where(depth>depth.mean()*10)] = 0
        # cloud point
        #内参做了归一化
        intrinsics = np.array([[focal_length_x,0,o_cx],[0,focal_length_y,o_cy],[0,0,1]])
        point = generate_point_cloud_from_depth(depth,intrinsics,extrinsic)
        #prune unvailid point
        rgb = cv2.resize(image,(target_w,target_h))
        #删除非法depth
        rgb = rgb[depth!= 0]
        #Mask
        if args.mask:
            mask_path = os.path.join(save_path,'Mask',os.path.basename(oris[i]))
            mask = read(mask_path,type='mask')
            mask = cv2.resize(mask,(target_w,target_h),interpolation=cv2.INTER_NEAREST).reshape(-1)
            rgb = rgb[mask == 0]
            point = point[mask == 0]


        points.append(point.reshape(-1,3))
        rgbs.append(rgb.reshape(-1,3))
        index += 1
    points = np.concatenate(points)
    rgbs = np.concatenate(rgbs)
    return image_infos,cam_infos,points,rgbs


        
def loop_helper(files,key='ori'):
    if len(jhelp_folder(files)) == 0:
        return [files]
    res = []
    for file in jhelp_folder(files):
        if os.path.basename(file) ==key:
            return [file]
        if 'fps' in os.path.basename(file).lower() or os.path.basename(file) in ['12','24','48']:
            res += [file]
        else:
            res += loop_helper(file)
    return res

def mkdir_helper(files,root,name):
    if len(files)>0:
        mkdir(os.path.join(root,name))
        for file in files:
            shutil.move(file,os.path.join(root,name))

def refine_float(lst):
    return sorted(lst, key=lambda x: int(re.findall(r"0\.(\d+)",x)[-1]))

if __name__ == '__main__':
    # assert len(sys.argv)==3 ,'usage: python exr_get_mv.py root save_path'
    
    args = init_param()
    root = args.path
    file_names = loop_helper(root)
    assert len(file_names)>0,'error root'
    for id,file_name in enumerate(file_names):
        print('starting ply calculation({}/{}) {}'.format(id+1,len(file_names),file_name))
        if len(jhelp_file(file_name)) != 0 and 'ori' != os.path.basename(file_name):
            ori_files = jhelp_file(file_name)
            if len(ori_files)>0:
                mkdir(os.path.join(file_name,'ori'))
                for ori_file in ori_files:
                    shutil.move(ori_file,os.path.join(file_name,'ori'))
        if 'ori' != os.path.basename(file_name):
            file_name = os.path.join(file_name,'ori')
        save_path = os.path.abspath(os.path.join(file_name,'..'))
        if not os.path.isdir(file_name):
            continue
        data = []
        file_datas = jhelp_file(file_name)
        #prune data
        file_datas = prune(file_datas,'finalimage')
        ply_cal_core(file_datas,save_path,args)
    if args.test:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(f'{save_path}/pointcloud/sparse/0/points3D.ply')
        # 创建坐标系
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd,axis])
    
    