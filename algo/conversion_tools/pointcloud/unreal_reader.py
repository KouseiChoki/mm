'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-09-05 16:37:38
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
from conversion_tools.pointcloud.fileutil.read_write_model import Camera,write_model,Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import OpenEXR
import array
import Imath
pt = Imath.PixelType(Imath.PixelType.FLOAT)
D2R  =  np.pi/180

class CameraInfo(NamedTuple):
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

class ImageInfo(NamedTuple):
    uid:int
    extrinsic:np.array

def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))
def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

def GetViewMatrixFromEularAngle(pitch, yaw, roll):
    viewMat_Yaw = GetViewMatrixFromEular(0,yaw,0)
    viewMat_Pitch = GetViewMatrixFromEular(pitch,0,0)
    viewMat_Roll = GetViewMatrixFromEular(0,0,roll)
    return np.dot(np.dot(viewMat_Yaw,viewMat_Roll),viewMat_Pitch)

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


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec
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
    # 将点云从相机坐标系转换到世界坐标系
    points_world = (extrinsics[:3, :3] @ points_camera.T).T + extrinsics[:3, 3]
    # points_world = np.dot(points_camera, extrinsics[:3, 3].T)    # 过滤掉深度为零的点
    # points_world = points_world[depth_image.flatten() > 0]

    return points_world

def GetViewMatrixFromEular(pitch, yaw, roll,):
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

def inv_4x4_matrix(src):
    return np.linalg.inv(np.array(src).reshape(4, 4))

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
        # worldpos_G = np.array(array.array('f', img_exr.channel(depth_G,pt))).reshape(size)
        # worldpos_B = np.array(array.array('f', img_exr.channel(depth_B,pt))).reshape(size)
        # worldpos_A = np.array(array.array('f', img_exr.channel('FinalImagePWWorldDepth.A',pt))).reshape(size)
        # worldpos = np.stack([worldpos_R,worldpos_G,worldpos_B],axis=2)
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
    return data_cur,worldpos

# filePath = '/Users/qhong/Desktop/0607/FtGothicCastle_05/ori/FtGothicCastle_05.0002.exr'
# o_data,depth = read_exr(filePath)
# o_CamMeta_curRot_pitch = -o_data['yaw']
# o_CamMeta_curRot_roll  = o_data['pitch']
# o_CamMeta_curRot_yaw   = o_data['roll']
# o_view = GetViewMatrixFromEularAngle(o_CamMeta_curRot_pitch,o_CamMeta_curRot_yaw,o_CamMeta_curRot_roll)
# inv_o_view = inv_4x4_matrix(o_view)
# w,h = o_data['w'],o_data['h']
# sss = 1
# if o_data['camera_type'] == 0:
#     # if True:
#     o_fx = w  * o_data['focal_length'] * sss / o_data['sensor_w']
#     half_fov = D2R * o_data['fov'] / 2
#     focal_len =  o_data['sensor_w'] /( np.tan(half_fov) * 2)
#     cur_focalLength = focal_len
# else:
#     half_fov = D2R * o_data['fov'] / 2
#     focal_len =  o_data['sensor_w'] /( np.tan(half_fov) * 2)
#     cur_focalLength = focal_len
#     o_fx = w  * cur_focalLength * sss / o_data['sensor_w']
# o_fy = o_fx
# offset = 0
# o_cx = w/2.0  + offset
# o_cy = h/2.0 + offset
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.
    Roll, pitch, and yaw should be provided in degrees.
    """
    # Convert degrees to radians
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qw, qx, qy, qz

def left_to_right_hand_roll_pitch_yaw(roll, pitch, yaw):
    # 反转pitch和yaw
    pitch = -pitch
    yaw = -yaw

    # 创建左手坐标系的旋转对象
    left_hand_rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    
    # 提取旋转矩阵
    left_hand_matrix = left_hand_rotation.as_matrix()

    # 反转z轴
    right_hand_matrix = np.array([
        [left_hand_matrix[0, 0], left_hand_matrix[0, 1], -left_hand_matrix[0, 2]],
        [left_hand_matrix[1, 0], left_hand_matrix[1, 1], -left_hand_matrix[1, 2]],
        [-left_hand_matrix[2, 0], -left_hand_matrix[2, 1], left_hand_matrix[2, 2]]
    ])

    # 创建右手坐标系的旋转对象
    right_hand_rotation = R.from_matrix(right_hand_matrix)

    # 转换为四元数
    quaternion = right_hand_rotation.as_quat()
    
    return quaternion


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
    else:
        R = np.dot(R_z, np.dot(R_y, R_x))
    if loc.__len__() > 0:
        ans = np.eye(4)
        ans[0:3,0:3] = R
        ans[0:3, -1] = loc
    else:
        ans = R
    if axis == 'left':
        ans = RightLeftAxisChange(ans)
    return ans

def get_intrinsic_extrinsic(root,down_scale,step=1,max_step=9999):
    # root = '/Users/qhong/Desktop/0607/FtGothicCastle_05'
    imgs = jhelp_file(os.path.join(root,'image'))
    oris = jhelp_file(os.path.join(root,'ori'))
    assert len(imgs)>0 and len(oris)>0 and len(oris)==len(imgs),'error input!'
    index = 1
    nums = len(oris)
    cam_infos,image_infos,points,rgbs = [],[],[],[]
    for i in tqdm(range(0,nums,step),desc=f'reading {os.path.basename(root)}'): 
        if index>=max_step:
            break
        filePath = oris[i]
        o_data,depth = read_exr(filePath)
        w,h = o_data['w'],o_data['h']
        if down_scale is None:
            down_scale_x = 1
            down_scale_y = 1
        else:
            down_scale_x = down_scale
            down_scale_y = down_scale
        depth /= 100
        depth[np.where(depth>1e5)] = 0
        # Provided Euler angles
        pitch_ = o_data['pitch']
        roll_  = o_data['roll']
        yaw_   = o_data['yaw']

        pitch = o_data['pitch']
        roll  = o_data['roll']
        yaw   = o_data['yaw']
        tx,ty,tz = o_data['x']/100,o_data['y']/100,o_data['z']/100,
        # extrinsic
        # Calculate quaternion
        # qw, qx, qy, qz = euler_to_quaternion(roll, pitch, yaw)
        # qw, qx, qy, qz = left_to_right_hand_roll_pitch_yaw(roll,pitch,yaw)

        # o_CamMeta_curRot_pitch = pitch
        # o_CamMeta_curRot_roll  = roll
        # o_CamMeta_curRot_yaw   = yaw 
        # o_CamMeta_curRot_pitch = -o_data['yaw']
        # o_CamMeta_curRot_roll  = o_data['pitch']
        # o_CamMeta_curRot_yaw   = o_data['roll']
        # np.set_printoptions(precision=3, suppress=True)
        # print(yaw,roll, pitch)
        # extrinsic = GetViewMatrixFromEularAngle(roll,pitch,yaw)
        # rotation = R.from_euler('xyz', [roll,pitch,yaw])
        # print(rotation.as_matrix())
        # rotation = eulerAngles2rotationMat_old([pitch,yaw,roll])
        # print(rotation)
        # extrinsic = eulerAngles2rotationMat([pitch,yaw,roll], loc = [ty,tz,-tx], format='degree', order = 'XYZ',axis='left')
        extrinsic = eulerAngles2rotationMat([-pitch,-yaw,roll], loc = [ty,tz,tx], format='degree', order = 'ZYX',axis='left')
        # print(rotation)
        #     # Get rotation matrix
        # extrinsic = np.eye(4)
        # extrinsic[:3, :3] = rotation
        # print(rotation.as_matrix())
        # print(tmp)
    # 通过矩阵乘法应用z轴反转
        # extrinsic = np.dot(extrinsic, flip_z)
        # print(extrinsic)
        # o_view = convert_left_to_right_view_matrix(GetViewMatrixFromEularAngle(pitch,yaw,roll))
        # qw,qx,qy,qz = rotmat2qvec(np.linalg.inv(o_view[:3,:3]))
        # qw,qx,qy,qz = rotmat2qvec(o_view[:3,:3])
        
        w2c = np.linalg.inv(extrinsic)
        # extrinsic[:3, 3] = [0,-tz,0]
        # extrinsic[:3, 3] = [-ty,-tz,tx]
        qx, qy, qz ,qw = R.from_matrix(w2c[:3, :3]).as_quat()
        tvec = w2c[:3, 3]

        # cR = np.transpose(qvec2rotmat([qw,qx, qy, qz]))
        # np.set_printoptions(precision=3, suppress=True)
        # print(extrinsic[:3,:3])
        # print(cR)
        # print(np.transpose(cR))

        image_info = ImageInfo(uid=index,extrinsic=np.array([qw,qx,qy,qz,tvec[0],tvec[1],tvec[2]]))
        image_infos.append(image_info)

        # tvec = np.array((tx, ty, tz))
        # w2c = torch.eye(4, dtype=torch.float32)
        # rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
        # w2c[:3, :3] = torch.tensor(rotation, dtype=torch.float32)
        # w2c[:3, 3] = torch.tensor(tvec, dtype=torch.float32)
        # print(w2c.inverse())
        
        # intrinsic
        offset=0
        o_cx = w/2.0  + offset
        o_cy = h/2.0 + offset
        model="PINHOLE"
        focal_length_x = w  * o_data['focal_length'] * 1 / o_data['sensor_w']
        focal_length_y = h  * o_data['focal_length'] * 1 / o_data['sensor_h']
        # fov = D2R * o_data['fov']
        # focal_length_x = fov2focal(fov,w)
        # focal_length_y = fov2focal(fov,h)
        # focal_length_x = o_data['focal_length']
        # focal_length_y = focal_length_x
        cam_info = CameraInfo(uid=index, fx=focal_length_x,fy=focal_length_y,cx=o_cx,cy=o_cy,image_name=os.path.basename(imgs[i]).replace('.png',''),image_path = imgs[i], width=w, height=h,model=model)
        cam_infos.append(cam_info)


        #downscale
        focal_length_x = focal_length_x/down_scale_x
        focal_length_y = focal_length_y/down_scale_y
        o_cx = o_cx/down_scale_x
        o_cy = o_cy/down_scale_y
        if down_scale_x != 1 or down_scale_y != 1:
            depth = cv2.resize(depth,None,fx=1/down_scale_x,fy=1/down_scale_y,interpolation=cv2.INTER_NEAREST)
        #prune unvalid depth
        # print(depth.max(),depth.mean())
        # depth[np.where(depth>depth.mean()*2)] = 0
        # cloud point
        #unproject
        # from tc_reader import sample_image_grid,unproject,homogenize_points
        # xy, _ = sample_image_grid((h//down_scale_y, w//down_scale_x),torch.device('cpu'))
        #内参做了归一化
        intrinsics = np.array([[focal_length_x,0,o_cx],[0,focal_length_y,o_cy],[0,0,1]])
        # intrinsics_nm = torch.FloatTensor([[focal_length_x/target_w,0,o_cx/target_w],[0,focal_length_y/target_h,o_cy/target_h],[0,0,1]])
        # xyz = unproject(xy, depth, intrinsics_nm)
        # xyz = homogenize_points(xyz)
        # # qvec = np.array([qw,qx,qy,qz])
        # # extrinsic = np.eye(4)
        # # extrinsic[:3, :3] = np.transpose(qvec2rotmat(qvec))
        # # extrinsic[:3, 3] = [tx,ty,tz]
        # # print(extrinsic)
        # xyz = einsum(extrinsic, xyz, "i j, ... j -> ... i")[..., :3]
        # point = rearrange(xyz, "h w xyz -> (h w) xyz")

        point = generate_point_cloud_from_depth(depth,intrinsics,extrinsic)
        # def depth_to_point_cloud(depth_map, intrinsics):

        

        # w2c = extrinsic.inverse().detach().cpu().numpy()
        # qx, qy, qz, qw = R.from_matrix(w2c[:3, :3]).as_quat()
        # qvec = np.array((qw, qx, qy, qz))
        # tvec = w2c[:3, 3]

        # v, u = np.indices(depth.shape)
        # Xc = (u - cx) * depth / fx
        # Yc = (v - cy) * depth / fy
        # Zc = depth

        # # 相机坐标系转换为世界坐标系
        # points_cam = np.stack((Xc, Yc, Zc), axis=-1)  # 相机坐标系中的点云

        # points_world = np.dot(R, points_cam.T).T + t  # 转换为世界坐标系中的点云

        # # 点云数据处理，如存储、可视化等
        # cloud_points = points_world.reshape(-1, 3)  # 转换为点云数据格式

        # depth = cv2.resize(depth,(target_w,target_h))
        # fx, fy = focal_length_x, focal_length_y
        # cx, cy = o_cx,o_cy
        # x, y = np.meshgrid(np.arange(target_w), np.arange(target_h))
        # x = (x - cx) / fx
        # y = (y - cy) / fy
        # z = depth
        # x *= z
        # y *= z
        # point = np.stack((x, y, z), axis=-1).reshape(-1, 3)

        rgb = read(imgs[i],type='image')
        if down_scale_x != 1 or down_scale_y != 1:
            rgb = cv2.resize(rgb,None,fx=1/down_scale_x,fy=1/down_scale_y,interpolation=cv2.INTER_NEAREST)
        
        points.append(point.reshape(-1,3))
        rgbs.append(rgb.reshape(-1,3))
        index += 1
        # FINAL_WIDTH = 1920
        # FINAL_HEIGHT = 1080
        # x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
        # x = (x - FINAL_WIDTH / 2) / focal_length_x
        # y = (y - FINAL_HEIGHT / 2) / focal_length_y
        # z = depth
        # points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    points = np.concatenate(points)
    rgbs = np.concatenate(rgbs)
    return image_infos,cam_infos,points,rgbs

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
    print('finished')

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
    print('writing camera info')
    write_model(cameras, images, None, path,ext='.txt')

def unreal_ply(root,down_scale):
    sp = os.path.join(root,'pointcloud')
    # target_w = 1920
    # target_h = 1080
    # target_w = 1920
    # target_h = 1080
    image_infos,cam_infos,xyz,rgbs = get_intrinsic_extrinsic(root,down_scale,step=1,max_step=6666)
    sparse_path = os.path.join(sp,'sparse/0')
    mkdir(sparse_path)
    write_ply(os.path.join(sparse_path , "points3D.ply"), xyz,rgbs)
    # Write out the images.
    mkdir(os.path.join(sp , "images"))
    for cam_info in cam_infos:
        shutil.copy(cam_info.image_path, os.path.join(sp , "images",os.path.basename(cam_info.image_path)))
    # Write out the camera parameters.
    write_colmap_model(sparse_path,cam_infos,image_infos)

if __name__ == "__main__":
    # import open3d as o3d
    # pcd= o3d.geometry.PointCloud()
    # # # 将 NumPy 数组赋值给点云对象
    # pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])
    root = '/Users/qhong/Desktop/0906'
    sp = os.path.join(root,'colmap')
    target_w = 1920
    target_h = 1080
    # target_w = 1920
    # target_h = 1080
    image_infos,cam_infos,xyz,rgbs = get_intrinsic_extrinsic(root,target_w,target_h,step=1,max_step=66)
    sparse_path = os.path.join(sp,'sparse/0')
    mkdir(sparse_path)
    write_ply(os.path.join(sparse_path , "points3D.ply"), xyz,rgbs)
    # Write out the images.
    mkdir(os.path.join(sp , "images"))
    for cam_info in cam_infos:
        shutil.copy(cam_info.image_path, os.path.join(sp , "images",os.path.basename(cam_info.image_path)))
    # Write out the camera parameters.
    write_colmap_model(sparse_path,cam_infos,image_infos)