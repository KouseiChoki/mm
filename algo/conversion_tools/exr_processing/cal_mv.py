'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-09-05 16:32:50
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
from file_utils import mvwrite,read
from tqdm.contrib.concurrent import process_map
import warnings
import re
import argparse
from color_convertion.colorutil import Color_transform
from conversion_tools.pointcloud.unreal_reader import unreal_ply
rec709_to_acescg = Color_transform('lin_rec709','acescg')
acescg_to_rec709 = Color_transform('acescg','lin_rec709')
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
    parser.add_argument('--extra_depth',  help="using extra depth to calculate mv")
    parser.add_argument('--mvinmask', action='store_true', help="no_mask_mode mode.")
    parser.add_argument('--objmvonly', action='store_true', help="test object mv")
    parser.add_argument('--inverse_mv', action='store_true', help="inverse_mv")
    parser.add_argument('--onlymv', action='store_false', help="output hdr to ldr image")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
    parser.add_argument('--dump_depth', action='store_true', help="dump world depth.")
    parser.add_argument('--dump_ply', action='store_true', help="dump_ply")
    parser.add_argument('--down_scale', type=int, default=1)
    parser.add_argument('--depth_only', action='store_true', help="only dump world depth.")
    parser.add_argument('--colormap', action='store_true', help="dump world depth colormap.")
    parser.add_argument('--MRQ', action='store_true', help="movie render queue source")
    parser.add_argument('--ACESCG', action='store_true', help="imagetype as acescg")
    parser.add_argument('--final', action='store_true', help="final image source")
    parser.add_argument('--f', action='store_true', help="force run")
    parser.add_argument('--trans_mode',action='store_true', help="testmode")
    parser.add_argument('--bg_mode',action='store_true', help="bg_mode")
    parser.add_argument('--step',type=int, default=1,help="for bg mv only")
    parser.add_argument('--core', type=int, default=4)
    parser.add_argument('--check_mode',action='store_true', help="check invalid data")
    args = parser.parse_args()
    if args.depth_only or args.colormap:
        args.dump_depth = True
    return args

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

def batch_exr_read_worldpos(files_name,name=''):
    files = jhelp(files_name)
    data = []
    for index in tqdm(range(len(files)),desc='loading:{}'.format(name)):
        file = files[index]
        data.append(exr_read_worldpos(file))
    return data


def exr_read_rgb(filePath):
    img_exr = OpenEXR.InputFile(filePath)
    dw = img_exr.header()['dataWindow']
    size = (dw.max.y - dw.min.y + 1,dw.max.x - dw.min.x + 1)
    # 获取rgb
    r_str, g_str, b_str = img_exr.channels('RGB',pt)
    red = np.array(array.array('f', r_str))
    green = np.array(array.array('f', g_str))
    blue = np.array(array.array('f', b_str))
    red = red.reshape(size)
    green = green.reshape(size)
    blue = blue.reshape(size)
    image = np.stack([red,green,blue],axis=2)
    return image

def exr_read_worldpos(filePath):
    img_exr = OpenEXR.InputFile(filePath)
    dtype=check_type(img_exr)
    dw = img_exr.header()['dataWindow']
    size = (dw.max.y - dw.min.y + 1,dw.max.x - dw.min.x + 1)
    #adjust
    depth_R,fnmv1_R,fnmv0_R,fnmask_R = None,None,None,None
    for key in img_exr.header()['channels']:
        if 'FinalImagePWWorldDepth.R' in key:
            depth_R = key
        elif 'FinalImageMovieRenderQueue_WorldDepth.R' in key:
            depth_R = key
        # if 'FinalImagePWWorldDepth.G' in key:
        #     depth_G = key
        # if 'FinalImagePWWorldDepth.B' in key:
        #     depth_B = key
        if 'FinalImagePWMV1.R' in key:
            fnmv1_R = key
        elif 'MotionVectors.R' in key:
            fnmv1_R = key
        if 'FinalImagePWMV1.G' in key:
            fnmv1_G = key
        elif 'MotionVectors.G' in key:
            fnmv1_G = key
        if 'FinalImagePWMV1.B' in key:
            fnmv1_B = key
        elif 'MotionVectors.B' in key:
            fnmv1_B = key
        if 'FinalImagePWMV1.A' in key:
            fnmv1_A = key
        elif 'MotionVectors.A' in key:
            fnmv1_A = key
        if 'FinalImagePWMV0.R' in key:
            fnmv0_R = key
        if 'FinalImagePWMV0.G' in key:
            fnmv0_G = key
        if 'FinalImagePWMV0.B' in key:
            fnmv0_B = key
        if 'FinalImagePWMV0.A' in key:
            fnmv0_A = key
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
    data['fov'] = float(img_exr.header()['unreal/camera/FinalImage/fov'])
    data['camera_type'] = camera_type
    data['h'],data['w'] = size
    data_pre,data_cur = data.copy(),data.copy()
    data_cur['x'] = float(img_exr.header()['unreal/camera/curPos/x'])
    data_cur['y'] = float(img_exr.header()['unreal/camera/curPos/y'])
    data_cur['z'] = float(img_exr.header()['unreal/camera/curPos/z'])
    data_cur['pitch'] = float(img_exr.header()['unreal/camera/curRot/pitch'])
    data_cur['roll'] = float(img_exr.header()['unreal/camera/curRot/roll'])
    data_cur['yaw'] = float(img_exr.header()['unreal/camera/curRot/yaw'])
    data_pre['x'] = float(img_exr.header()['unreal/camera/prevPos/x'])
    data_pre['y'] = float(img_exr.header()['unreal/camera/prevPos/y'])
    data_pre['z'] = float(img_exr.header()['unreal/camera/prevPos/z'])
    data_pre['pitch'] = float(img_exr.header()['unreal/camera/prevRot/pitch'])
    data_pre['roll'] = float(img_exr.header()['unreal/camera/prevRot/roll'])
    data_pre['yaw'] = float(img_exr.header()['unreal/camera/prevRot/yaw'])

    if fnmv0_R is not None:
        mv0_x = np.array(array.array('f', img_exr.channel(fnmv0_R,pt))).reshape(size)
        mv0_y = np.array(array.array('f', img_exr.channel(fnmv0_G,pt))).reshape(size)
        mv0_z = np.array(array.array('f', img_exr.channel(fnmv0_B,pt))).reshape(size)
        mv0_a = np.array(array.array('f', img_exr.channel(fnmv0_A,pt))).reshape(size)
        #denormalized
        mv0_x = (mv0_x - 0.5) * 2 * size[1] * -1
        # mv0_x = (mv0_x - 0.5) * 2 * size[1] 
        mv0_y = (mv0_y - 0.5) * 2 * size[0]
        mv0 = np.stack([mv0_x,mv0_y,mv0_z,mv0_a],axis=2)
    else:
        mv0 = None
    if fnmv1_R is not None:
        mv1_x = np.array(array.array('f', img_exr.channel(fnmv1_R,pt))).reshape(size)
        mv1_y = np.array(array.array('f', img_exr.channel(fnmv1_G,pt))).reshape(size)
        mv1_z = np.array(array.array('f', img_exr.channel(fnmv1_B,pt))).reshape(size)
        mv1_a = np.array(array.array('f', img_exr.channel(fnmv1_A,pt))).reshape(size)
        mv1_x = (mv1_x - 0.5) * 2 * size[1] * -1
        # mv1_x = (mv1_x - 0.5) * 2 * size[1]
        mv1_y = (mv1_y - 0.5) * 2 * size[0]
        mv1 = np.stack([mv1_x,mv1_y,mv1_z,mv1_a],axis=2)
    else:
        mv1 = None
    mask = np.array(array.array('f', img_exr.channel(fnmask_R,pt))).reshape(size) if fnmask_R is not None else None
    
    r_str, g_str, b_str = img_exr.channels('RGB',pt)
    red = np.array(array.array('f', r_str)).reshape(size)
    green = np.array(array.array('f', g_str)).reshape(size)
    blue = np.array(array.array('f', b_str)).reshape(size)
    image = np.stack([red,green,blue],axis=2).astype('float32')
    return image,worldpos,data_pre,data_cur,mv0,mv1,mask,dtype



def exr_read_worldpos_next(filePath):
    if not os.path.isfile(filePath):
        return None,None
    img_exr = OpenEXR.InputFile(filePath)
    dw = img_exr.header()['dataWindow']
    size = (dw.max.y - dw.min.y + 1,dw.max.x - dw.min.x + 1)
    data = {}
    camera_type = 1
    for ic in img_exr.header().keys():
        if '/focalLength' in ic:
            camera_type = 0
            data['focal_length'] = float(img_exr.header()['unreal/camera/FinalImage/focalLength'])
            break
    data['sensor_w'] = float(img_exr.header()['unreal/camera/FinalImage/sensorWidth'])
    data['fov'] = float(img_exr.header()['unreal/camera/FinalImage/fov'])
    data['camera_type'] = camera_type
    # print(f'camera_type={camera_type}')
    data['h'],data['w'] = size
    data_cur = data.copy()
    data_cur['x'] = float(img_exr.header()['unreal/camera/curPos/x'])
    data_cur['y'] = float(img_exr.header()['unreal/camera/curPos/y'])
    data_cur['z'] = float(img_exr.header()['unreal/camera/curPos/z'])
    data_cur['pitch'] = float(img_exr.header()['unreal/camera/curRot/pitch'])
    data_cur['roll'] = float(img_exr.header()['unreal/camera/curRot/roll'])
    data_cur['yaw'] = float(img_exr.header()['unreal/camera/curRot/yaw'])
    if 'FinalImagePWMV0.R' in img_exr.header()['channels']:
        mv0_x = np.array(array.array('f', img_exr.channel('FinalImagePWMV0.R',pt))).reshape(size)
        mv0_y = np.array(array.array('f', img_exr.channel('FinalImagePWMV0.G',pt))).reshape(size)
        mv0_z = np.array(array.array('f', img_exr.channel('FinalImagePWMV0.B',pt))).reshape(size)
        mv0_a = np.array(array.array('f', img_exr.channel('FinalImagePWMV0.A',pt))).reshape(size)
        #denormalized
        mv0_x = (mv0_x - 0.5) * 2 * size[1] * -1
        # mv0_x = (mv0_x - 0.5) * 2 * size[1]
        mv0_y = (mv0_y - 0.5) * 2 * size[0]
        mv0 = np.stack([mv0_x,mv0_y,mv0_z,mv0_a],axis=2)
    else:
        mv0 = None
    return data_cur,mv0


def camera_tracking(depth,o_data,p_data,factor=1,reverse=1):
    w,h = o_data['w'],o_data['h']
    o_CamMeta_curRot_pitch = -o_data['yaw']
    o_CamMeta_curRot_roll  = o_data['pitch']
    o_CamMeta_curRot_yaw   = o_data['roll']

    p_CamMeta_curRot_pitch = -p_data['yaw']
    p_CamMeta_curRot_roll  = p_data['pitch'] 
    p_CamMeta_curRot_yaw   = p_data['roll']

    o_view = GetViewMatrixFromEularAngle(o_CamMeta_curRot_pitch,o_CamMeta_curRot_yaw,o_CamMeta_curRot_roll)
    p_view = GetViewMatrixFromEularAngle(p_CamMeta_curRot_pitch,p_CamMeta_curRot_yaw,p_CamMeta_curRot_roll)

    #inverse cur_view
    inv_o_view = inv_4x4_matrix(o_view)
    #camera type
    sss = 1
    if o_data['camera_type'] == 0:
    # if True:
        o_fx = w  * o_data['focal_length'] * sss / o_data['sensor_w']
        half_fov = D2R * o_data['fov'] / 2
        focal_len =  o_data['sensor_w'] /( np.tan(half_fov) * 2)
        cur_focalLength = focal_len
    else:
        half_fov = D2R * o_data['fov'] / 2
        focal_len =  o_data['sensor_w'] /( np.tan(half_fov) * 2)
        cur_focalLength = focal_len
        o_fx = w  * cur_focalLength * sss / o_data['sensor_w']

    if p_data['camera_type'] == 0:
    # if True:
        p_fx = w  * p_data['focal_length'] * sss / p_data['sensor_w']
        half_fov = D2R * p_data['fov'] / 2
        focal_len =  p_data['sensor_w'] /( np.tan(half_fov) * 2)
    else:
        half_fov = D2R * p_data['fov'] / 2
        focal_len =  p_data['sensor_w'] /( np.tan(half_fov) * 2)
        pre_focalLength = focal_len
        p_fx = w  * pre_focalLength * sss / p_data['sensor_w']

    o_fy = o_fx
    p_fy = p_fx

    offset = 0
    o_cx = w/2.0  + offset
    o_cy = h/2.0 + offset
    p_cx = w/2.0  + offset
    p_cy = h/2.0 + offset
    delta_y =  p_data['z'] -  o_data['z']
    delta_z =  -p_data['x'] +  o_data['x']
    delta_x =  p_data['y'] -  o_data['y']

    o_CS_Z = depth
    oxx = np.repeat(np.arange(w)[::-1][None,...],h,axis=0)
    oyy = np.repeat(np.arange(h)[...,None],w,axis=1)
    o_CS_X_p = (oxx - o_cx ) / o_fx
    o_CS_Y_p = (oyy - o_cy ) / o_fy

    o_CS_X = o_CS_X_p * o_CS_Z
    o_CS_Y = o_CS_Y_p * o_CS_Z
    
    o_WS_X = inv_o_view[0][0] * o_CS_X + inv_o_view[0][1] * o_CS_Y + inv_o_view[0][2] * o_CS_Z + inv_o_view[0][3]
    o_WS_Y = inv_o_view[1][0] * o_CS_X + inv_o_view[1][1] * o_CS_Y + inv_o_view[1][2] * o_CS_Z + inv_o_view[1][3]
    o_WS_Z = inv_o_view[2][0] * o_CS_X + inv_o_view[2][1] * o_CS_Y + inv_o_view[2][2] * o_CS_Z + inv_o_view[2][3]
    o_WS_W = inv_o_view[3][0] * o_CS_X + inv_o_view[3][1] * o_CS_Y + inv_o_view[3][2] * o_CS_Z + inv_o_view[3][3]
    o_WS_X = o_WS_X  + delta_x
    o_WS_Y = o_WS_Y  + delta_y
    o_WS_Z = o_WS_Z  + delta_z

    p_CS_X = p_view[0][0] * o_WS_X + p_view[0][1] * o_WS_Y + p_view[0][2] * o_WS_Z + p_view[0][3] * o_WS_W
    p_CS_Y = p_view[1][0] * o_WS_X + p_view[1][1] * o_WS_Y + p_view[1][2] * o_WS_Z + p_view[1][3] * o_WS_W
    p_CS_Z = p_view[2][0] * o_WS_X + p_view[2][1] * o_WS_Y + p_view[2][2] * o_WS_Z + p_view[2][3] * o_WS_W
    # p_CS_W = p_view[3][0] * o_WS_X + p_view[3][1] * o_WS_Y + p_view[3][2] * o_WS_Z + p_view[3][3] * o_WS_W

    p_SS_X = p_fx * p_CS_X / p_CS_Z + p_cx
    p_SS_Y = p_fy * p_CS_Y / p_CS_Z + p_cy
    
    mv_x =  (p_SS_X - oxx ) * reverse * -1
    mv_y =  (p_SS_Y - oyy ) * reverse
    mv_z = p_CS_Z - o_CS_Z
    mv_depth = o_CS_Z  * reverse
    result = np.stack([mv_x,mv_y,mv_z,mv_depth],axis=-1)
    return result

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

def inv_4x4_matrix(src):
    return np.linalg.inv(np.array(src).reshape(4, 4))

def adjust(res):
    # res[...,-1] *= 0
    res[...,0] /= res.shape[1]
    res[...,1] /= res.shape[0]
    # prune 3rd channel mv_z
    res[...,2]*=0
    return res

def depth_vis(depth):
    depth = depth[...,0]
    # max_value = max(depth.max(),6000)
    min_value = depth.min()
    max_value = depth.max()
    # 归一化深度值
    normalized_depth_values = (depth - min_value) / (max_value - min_value)
    # depth /= max_value
    return normalized_depth_values

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

def curv_read(img):
    hdr_image = read(img,type='image')
    # ldr_image = hdr_to_rgb(hdr_image)
    ldr_image = adjust_gamma(hdr_image)
    return ldr_image

def video_unzip_core(datas):
    i,file_name,save_path,img,args = datas
    sp = os.path.join(save_path,'image',os.path.basename(img).replace('.exr','.png'))
    if os.path.isfile(sp) and not args.f:
        return
    mvwrite(sp,curv_read(file_name[i]))

def trans_core(datas):
    i,file_name,save_path,img,args = datas
    if os.path.isfile(os.path.join(save_path,'ACESCG',os.path.basename(img))) and not args.f:
        return
    hdr_image = read(file_name[i],type='ldr')
    image = hdr_to_rgb(hdr_image)[...,:3]
    image_path = os.path.join(save_path,'image',os.path.basename(img)).replace('.exr','.png')
    if not os.path.isfile(image_path):
        mvwrite(image_path,image)
    mvwrite(os.path.join(save_path,'ACESCG',os.path.basename(img)),rec709_to_acescg.apply(hdr_image))
        
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



def mv_cal_core(datas):
    i,file_name,save_path,img,extra_depth,args = datas
    step = args.step
    if os.path.isfile(os.path.join(save_path,'mv1',os.path.basename(img))) and not args.f and not args.depth_only and step==1:
        return
    if os.path.isfile(os.path.join(save_path,'world_depth',os.path.basename(img))) and not args.f and args.depth_only and step==1:
        return
    mv0_sp,mv1_sp = os.path.join(save_path,f'mv{(step-1)*2}'),os.path.join(save_path,f'mv{(step-1)*2+1}')
    hdr_image,depth,data_debug,data1,objmv0_,objmv1,mask,dtype = exr_read_worldpos(file_name[i])
    if extra_depth is not None:
        depth = read(extra_depth)[...,0]*100
    if depth is None:
        print('no depth inf')
        if args.dump_depth:
            return
    elif args.depth_only:
        # print('depth only mode')
        pass
    else:
        if args.debug:
            for sens in ['x','y','z','pitch','roll','yaw']:
                if data0 is not None:
                    assert data0[sens] == data_debug[sens],'data error!'
                    # tmp = '相同' if data0[sens] == data_debug[sens] else '不相同'
                    # print(f'{sens}:{tmp},{data0[sens]},{data_debug[sens]}')
        if i-step>=0:
            data0,_  =  exr_read_worldpos_next(file_name[i-step])
        else:
            data0,objmv1 = None,None
        if data0 is None: #mv1 not exist
            mv1 = np.zeros((hdr_image.shape[0],hdr_image.shape[1],4))
        else:
            mv1 = camera_tracking(depth,data1,data0) 
            if objmv1 is not None and not args.bg_mode:
                if not args.mvinmask:
                    if args.objmvonly:
                        mv1 = objmv1
                    else:
                        mv1[np.where(objmv1[...,:2]!=0)] = objmv1[np.where(objmv1[...,:2]!=0)]
                else:
                    tmp_mask = np.repeat(mask[...,None],2,axis=2)
                    mv1[np.where(tmp_mask!=0)] = objmv1[np.where(tmp_mask!=0)]
            # mv1[...,0] *= -1
        mvwrite(os.path.join(mv1_sp,os.path.basename(img)),adjust(mv1),precision='half')

        if objmv0_ is not None or args.bg_mode:
            if i+step<len(file_name):
                data1_,objmv0 =  exr_read_worldpos_next(file_name[i+step])
            else:
                data1_,objmv0 =  None,None
            if data1_ is None: #mv0 not exist:
                mv0 = np.zeros((hdr_image.shape[0],hdr_image.shape[1],4))
            else:
                mv0 = camera_tracking(depth,data1,data1_)
                if objmv0 is not None and not args.bg_mode:
                    if not args.mvinmask:
                        mv0[np.where(objmv0[...,:2]!=0)] = objmv0[np.where(objmv0[...,:2]!=0)]
                    else:
                        tmp_mask = np.repeat(mask[...,None],2,axis=2)
                        mv0[np.where(tmp_mask!=0)] = objmv0[np.where(tmp_mask!=0)]
                # mv0[...,1] *= -1
            mvwrite(os.path.join(mv0_sp,os.path.basename(img)),adjust(mv0),precision='half')
    
    if step !=1:
        return
    #hdr
   
    if dtype=='acescg':
        mvwrite(os.path.join(save_path,'ACESCG',os.path.basename(img)),hdr_image)
        hdr_image = acescg_to_rec709.apply(hdr_image)
    elif dtype=='rec709':
        mvwrite(os.path.join(save_path,'rec709',os.path.basename(img)),hdr_image)
        if args.ACESCG:
            mvwrite(os.path.join(save_path,'ACESCG',os.path.basename(img)),rec709_to_acescg.apply(hdr_image))
    else:
        raise ValueError(f'not supported color space:{dtype}')
    #ldr
    image = hdr_to_rgb(hdr_image)[...,:3]
    image_path = os.path.join(save_path,'image',os.path.basename(img)).replace('.exr','.png')
    if not os.path.isfile(image_path) and args.onlymv:
        mvwrite(image_path,image)
    #mask
    if mask is not None:
        mvwrite(os.path.join(save_path,'Mask',os.path.basename(img)),np.repeat(mask[...,None],4,axis=2))
    

    if args.dump_depth:
        depth = depth
        # depth = depth_vis(depth)
        depth /= 100
        depth[np.where(depth>1e5)] = 0 #invalid depth like sora
        dpname = os.path.join(save_path,'world_depth',os.path.basename(img))
        depth = np.repeat(depth[...,None],4,axis=2)
        d = depth[...,-1]
        d = (d - d.min()) / (d.max() - d.min())
        depth[...,-1] = d
        if args.colormap:
            d = depth[...,-1]
            d = (d - d.min()) / (d.max() - d.min()) * 255.0
            d = cv2.applyColorMap(d.astype('uint8'), cv2.COLORMAP_INFERNO)
            depth = d
            dpname = dpname.replace('.exr','.png')
        mvwrite(dpname,depth)
    
def inverse_core(datas):
    i,file_name,save_path,img,args = datas
    mv1_ = file_name[i]
    mv0_ = file_name[i].replace('/mv1/','/mv0/')
    mv1 = read(mv1_,type='flo')
    mv1[...,0] *= -1
    mvwrite(mv1_,mv1,precision='half')
    if os.path.isfile(mv0_):
        mv0 = read(mv0_,type='flo')
        mv0[...,1] *= -1
        mvwrite(mv0_,mv0,precision='half')
    
def mv_check_core(sp,check_file):
    mv0 = os.path.join(sp,'mv0')
    mv1 = os.path.join(sp,'mv1')
    depth = os.path.join(sp,'world_depth')
    mask = os.path.join(sp,'Mask')
    image = os.path.join(sp,'image')
    # check_single(mv0,'mv0',check_file)
    # check_single(mv1,'mv1',check_file)
    check_single(depth,'depth',check_file)
    # check_single(mask,'mask',check_file)
    check_single(image,'image',check_file)

def check_single(d,name,check_file):
    record_depth = True
    if not os.path.isdir(d):
        return
    datas = jhelp_file(d)
    result = ''
    tmp = 0
    if 'mv' in name:
        looper = tqdm(range(1,len(datas)-1),desc=name)
    else:
        looper = tqdm(range(0,len(datas)),desc=name)
    for i in looper:
        data_ = datas[i]
        data = read(data_,type='flo')[...,:2] if name!='image' else read(data_,type='image').astype('float32')
        if data.mean()<=1e-6:
            result += f'invalid_value:{data_}\n'
        if record_depth:
            dpdata = data[...,0]
            valid = (dpdata != 0) & (dpdata < 1000)
            dpdata = dpdata.view()[valid.view()]
            tmp += dpdata.mean()
    if len(result)>1:
        print('error data generated')
        with open(check_file, 'a') as f:
                    f.write(result)
    if record_depth and name == 'depth': #record average depth
        with open(check_file, 'a') as f:
            tpath = os.path.basename(os.path.abspath(os.path.join(d,'..')))
            f.write(f'average_depth:{tpath}:{tmp/len(datas)}\n')

def mq_core(datas):
    save_path,image_,mask_,mv1_,depth,name,args = datas
    if os.path.isfile(os.path.join(save_path,'mv1',name)) and not args.f:
        return
    # image = hdr_to_rgb(read(image_,type='hdr'))
    image = curv_read(image_)
    mask =read(mask_,type='mask',Unrealmode=True)
    mv1 = read(mv1_,type='flo')
    #mask filter

    #denormalized
    size = image.shape[:2]
    mv1[...,0] = (mv1[...,0] - 0.5) * 2 * size[1] * -1
    # mv1[...,0] = (mv1[...,0] - 0.5) * 2 * size[1] *-1
    mv1[...,1] = (mv1[...,1] - 0.5) * 2 * size[0]
    mv1[np.where(mask==0)] = 0
    image_path = os.path.join(save_path,'image',name).replace('.exr','.png')
    mvwrite(image_path,image[...,:3])
    if depth is not None:
        depth = read(depth,type='flo')[...,:3]
        mvwrite(os.path.join(save_path,'world_depth',name),depth_vis(depth))
    mvwrite(os.path.join(save_path,'Mask',name),np.repeat(mask[...,None],4,axis=2))
    mvwrite(os.path.join(save_path,'mv1',name),adjust(mv1),precision='half')

def rename(source,target):
    try:
        os.rename(source,target)
        print(f"文件夹名已从 '{source}' 改为 '{target}'")
    except OSError as error:
        print(f"更改文件夹名时发生错误: {error}")

def restore_file_name(root):
    for file in jhelp_folder(root):
        if os.path.basename(file).lower() =='orinal':
            rename(file, os.path.join(os.path.dirname(file),'ori'))
        if os.path.basename(file).lower() =='12':
            rename(file, os.path.join(os.path.dirname(file),'12fps'))
        if os.path.basename(file).lower() =='24':
            rename(file, os.path.join(os.path.dirname(file),'24fps'))
        if os.path.basename(file).lower() =='48':
            rename(file, os.path.join(os.path.dirname(file),'48fps'))
        if os.path.basename(file) =='mask':
            rename(file, os.path.join(os.path.dirname(file),'Mask'))
            
            
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
    num_of_core = args.core
    root = args.path
    mode = 1
    args.dump_depth = True
    # if '/final' in root:
        # mode = 2
    if args.MRQ:
        mode = 1
    if args.final:
        mode = 2
    if args.trans_mode:
        mode = 4
    if args.inverse_mv:
        mode = 5
    if args.check_mode:
        mode = 6
    restore_file_name(root)
    
    
    if mode == 1:
        file_names = loop_helper(root)
        assert len(file_names)>0,'error root'
        for id,file_name in enumerate(file_names):
            print('starting camera mv calculation({}/{}) {}'.format(id+1,len(file_names),file_name))
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
            name = os.path.basename(save_path)
            data = []
            file_datas = jhelp_file(file_name)
            #prune data
            file_datas = prune(file_datas,'finalimage')
            #extra depth
            extra_depth = [None] * len(file_datas) 
            if args.extra_depth is not None:
                extra_depth_path = os.path.join(args.extra_depth,name)
                depth_name = 'mono_depth'
                for f in jhelp_folder(extra_depth_path): #change depth name 
                    if 'mono_depth' in f and 'mono_depth' != f:
                        depth_name = f
                extra_depth = jhelp_file(os.path.join(extra_depth_path,depth_name))
            for i in range(len(file_datas)):
                data.append([i,file_datas,save_path,file_datas[i],extra_depth[i],args])
            if num_of_core == 0:
                mv_cal_core(data[0])
            else:
                process_map(mv_cal_core, data, max_workers= num_of_core,desc='processing:{}'.format(name))
            if args.dump_ply:
                unreal_ply(save_path,args.down_scale)
    elif mode ==2:
        file_names = loop_helper(root)
        assert len(file_names)>0,'error root'
        for id,file_name in enumerate(file_names):
            if os.path.basename(file_name) =='image':
                continue
            if len(jhelp_file(file_name)) != 0 and 'video' not in os.path.basename(file_name).lower():
                ori_files = jhelp_file(file_name)
                if len(ori_files)>0:
                    mkdir(os.path.join(file_name,'video'))
                    for ori_file in ori_files:
                        shutil.move(ori_file,os.path.join(file_name,'video'))
            if 'video' not in os.path.basename(file_name).lower():
                file_name = os.path.join(file_name,'video')
            save_path = os.path.abspath(os.path.join(file_name,'..'))
            if not os.path.isdir(file_name):
                continue
            name = os.path.basename(save_path)
            data = []
            file_datas = jhelp_file(file_name)
            print('starting tonemapping process({}/{}) {}'.format(id+1,len(file_names),file_name))
            for i in range(len(file_datas)):
                data.append([i,file_datas,save_path,file_datas[i],args])
            process_map(video_unzip_core, data, max_workers= num_of_core,desc='{}'.format(name))
    elif mode ==3:
        file_names = loop_helper(root)
        assert len(file_names)>0,'error root'
        for id,file_name in enumerate(file_names):
            print('starting movie render queue mv calculation({}/{}) {}'.format(id+1,len(file_names),file_name))
            if len(jhelp_file(file_name)) != 0 and 'ori' not in os.path.basename(file_name).lower():
                all_files = jhelp_file(file_name)
                mkdir_helper(all_files,file_name,'ori')

            if 'ori' not in os.path.basename(file_name).lower():
                file_name = os.path.join(file_name,'ori')
            save_path = os.path.abspath(os.path.join(file_name,'..'))

            all_files = gofind(jhelp_file(file_name),keyword='.exr')
            imgs_files = refine_float(gofind(all_files,keyword='FinalImage_'))
            name_files = prune(all_files,keyword='FinalImage')
            depth_files = refine_float(gofind(all_files,keyword='FinalImagePWWorldDepth_'))
            mask_files = refine_float(gofind(all_files,keyword='FinalImagePWMask_'))
            mv1_files = refine_float(gofind(all_files,keyword='FinalImagePWMV1_'))
            if not os.path.isdir(file_name):
                continue
            name = os.path.basename(save_path)
            data = []
            #prune data
            num = min(len(imgs_files),len(mask_files),len(mv1_files),len(name_files))
            if args.dump_depth:
                num = min(num,len(depth_files))
            for i in range(num):
                if args.dump_depth :
                    depth = depth_files[i]
                else:
                    depth = None
                data.append([save_path,imgs_files[i],mask_files[i],mv1_files[i],depth,os.path.basename(name_files[i]),args])
            
            process_map(mq_core, data, max_workers= num_of_core,desc='processing:{}'.format(name))

    elif mode == 4:
        # file_names = loop_helper(root,key='video')
        file_names = find_folders_with_subfolder(root,keys=['video'],path_excs=['/bbox','fpstype','/obj'])
        assert len(file_names)>0,'error root'
        for id,file_name in enumerate(file_names):
            print('starting transform mode({}/{}) {}'.format(id+1,len(file_names),file_name))
            if 'video' not in os.path.basename(file_name).lower():
                file_name = os.path.join(file_name,'video')
            save_path = os.path.abspath(os.path.join(file_name,'..'))
            if not os.path.isdir(file_name):
                continue
            name = os.path.basename(save_path)
            data = []
            file_datas = jhelp_file(file_name)
            #prune data
            file_datas = prune(file_datas,'finalimage')
            for i in range(len(file_datas)):
                data.append([i,file_datas,save_path,file_datas[i],args])
            process_map(trans_core, data, max_workers= num_of_core,desc='processing:{}'.format(name))

    elif mode == 5:
        # file_names = loop_helper(root,key='video')
        file_names = find_folders_with_subfolder(root,keys=['mv1'],path_excs=['/bbox','fpstype','/obj'])
        assert len(file_names)>0,'error root'
        for id,file_name in enumerate(file_names):
            print('starting inverse mode({}/{}) {}'.format(id+1,len(file_names),file_name))
            if '/mv1' not in os.path.basename(file_name).lower():
                file_name = os.path.join(file_name,'mv1')
            save_path = os.path.abspath(os.path.join(file_name,'..'))
            if not os.path.isdir(file_name):
                continue
            name = os.path.basename(save_path)
            data = []
            file_datas = jhelp_file(file_name)
            #prune data
            file_datas = prune(file_datas,'finalimage')
            for i in range(len(file_datas)):
                data.append([i,file_datas,save_path,file_datas[i],args])
            process_map(inverse_core, data, max_workers= num_of_core,desc='processing:{}'.format(name))
    

    elif mode == 6:
        import datetime
        check_file = os.path.join(os.getcwd(),f'check_result_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        file_names = loop_helper(root)
        assert len(file_names)>0,'error root'
        for id,file_name in enumerate(file_names):
            print('starting mv check progress({}/{}) {}'.format(id+1,len(file_names),file_name))
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
            name = os.path.basename(save_path)
            mv_check_core(save_path,check_file)