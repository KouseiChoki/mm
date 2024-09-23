'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-09-23 13:42:37
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
from myutil import mask_adjust,write_txt
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
    parser.add_argument('--mask_type', type=str,default='nomask', help="bg or mix",choices=['nomask','bg','mix'])
    parser.add_argument('--mask_threshold', type=float, default=0,help="prune mask threshold")
    parser.add_argument('--fg_mask_adjust', type=int, default=0,help="prune mask threshold")
    parser.add_argument('--bg_mask_adjust', type=int, default=0,help="prune mask threshold")
    parser.add_argument('--judder_angle',type=int, default=-1,help="frame step")
    parser.add_argument('--inverse_depth',action='store_true', help="depth= 1/depth")
    parser.add_argument('--rub', action='store_true', help="dump rub viewmatrix")
    parser.add_argument('--test', action='store_true', help="use test")
    parser.add_argument('--down_scale',type=int, default=1,help="downscale rate")
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

def cal_qvec(data):
    rx,ry,rz,tx,ty,tz = data
    rotation_matrix = R.from_euler('XYZ', [rx,ry,rz],degrees=True).as_matrix()
    c2w = np.eye(4,4)
    if args.baseline_distance!=0:
        tx += args.baseline_distance
    c2w[:3,:3] = rotation_matrix
    c2w[:3,-1] = [tx,ty,tz]
    rub = c2w.copy() if args.rub else None
    c2w[:,1:3] *= -1
    w2c = np.linalg.inv(c2w)
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
        etmp,c2w,rub = cal_qvec(ext[i])
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
        
        if args.mask_type != 'nomask':
            mask = read(masks[i],type='mask')
            if args.bg_mask_adjust != 0:
                mask = mask_adjust(mask,size=args.bg_mask_adjust)
            if args.down_scale != 1 :
                mask = cv2.resize(mask,(target_w,target_h),interpolation=cv2.INTER_NEAREST).reshape(-1)
            else:
                mask = mask.reshape(-1)
            condition = mask <= args.mask_threshold
            rgb = rgb_[condition]
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
        if point is not None:
            points.append(point.reshape(-1,3))
        if rgb is not None:
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
    mkdir(os.path.join(sp , "masks"))
    for mask in masks:
        shutil.copy(mask, os.path.join(sp , "masks",os.path.basename(mask)))
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
        write_colmap_model(sparse_path,cam_infos,image_infos,step=args.step)
        # shutil.copy(raw_ply,os.path.join(sp,'sparse/0/points3D.ply'))
        # if args.baseline_distance==0:
        if ply_data is not None:
            ply_data.write(ply_path)
    
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

def sliding_window(sequence, window_size,window_step,pad=0,pad_step=0):
    """Generate a sliding window over a sequence."""
    window_size -=2
    res = []
    for i in range(0, len(sequence), window_step):
        window = sequence[i:i+window_size]
        if len(window) < window_size:
            window = sequence[-window_size:]
        if pad>0:
            for _ in range(pad):
                window = np.hstack((window[0]-pad_step,window,window[-1]+pad_step))
        res.append(window)
    return res

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
    if args.mask_type != 'nomask':
        assert len(masks)>0,'can not find mask file!'
    
    ext_ = read_extrinsics(extrinsic_file)
    if len(images) <= args.max_frame:
        images_prepare = [[images[i] for i in range(0,len(images),args.step)]]
        masks_prepare = [[masks[i] for i in range(0,len(masks),args.step)]]
        depths_prepare = [[depths[i] for i in range(0,len(depths),args.step)]]
        extrinsics = [ext_]
    else:
        images_prepare = sliding_window(images,args.max_frame,args.step)
        masks_prepare = sliding_window(masks,args.max_frame,args.step)
        depths_prepare = sliding_window(depths,args.max_frame,args.step)
        extrinsics = sliding_window(ext_,args.max_frame,args.step)
        # extrinsics = sliding_window(ext_,min(ext_,len(args.max_frame)*args.step-args.step+1))

    task_indexes = np.arange(args.start_frame,args.start_frame+args.max_frame)
    task_indexes = []
    curs = []
    for i in range(len(images)):
        cur = args.max_frame//2
        tmp = i+(np.arange(args.max_frame)-args.max_frame//2)*args.step
        while(tmp.min()<0):
            tmp +=args.step
            cur -= 1
            if cur < 0 or cur >= args.max_frame:
                raise ValueError('error max framse')
        while(tmp.max()>len(images)-1):
            tmp -=args.step
            cur += 1
            if cur < 0 or cur >= args.max_frame:
                raise ValueError('error max framse')
        
        tmp = [np.clip(k,0,len(images)-1) for k in tmp]

        task_indexes.append(tmp)
        curs.append(cur)
    # if not args.full_result:
    # tmp_curs = []
    # tmp_task_indexes = []
    # for i in range(len(curs)):
    #     if curs[i] == args.max_frame//2:
    #         tmp_curs.append(curs[i])
    #         tmp_task_indexes.append(task_indexes[i])
    # curs = tmp_curs
    # task_indexes = tmp_task_indexes

    images_prepare = [[images[ii] for ii in i]for i in task_indexes]
    masks_prepare = [[masks[ii] for ii in i]for i in task_indexes]
    depths_prepare = [[depths[ii] for ii in i]for i in task_indexes]
    extrinsics_ = read_extrinsics(extrinsic_file)
    extrinsics = [[extrinsics_[ii] for ii in i]for i in task_indexes]
    #需要判断重复元素 --root /Users/qhong/Desktop/avatar_data/2039 --max_frame 5 --step 2  --inverse_depth  --mask_type fg 
    instrinsics = read_intrinsic(intrinsic_file)
    source_ext = []
    source_ins = []
    for ext in ext_:
        tmp,_,_ = cal_qvec(ext)
        source_ext.append(tmp)
        source_ins.append(instrinsics)
    source_ext = np.stack(source_ext)
    
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
        ply_cal_core(images_prepare[i],depths_prepare[i],instrinsics,extrinsics[i],save_path,args,masks_prepare[i])

        if i < len(images_prepare):
            if i == len(images_prepare)-1:
                image_infos = [ImageInfo(uid=i,extrinsic=source_ext[i],rub=None)]
                cam_infos = [CameraInfo(uid=i, fx=float(source_ins[i]['fx']),fy=float(source_ins[i]['fy']),cx=int(source_ins[i]['w'])/2.0 ,cy=int(source_ins[i]['h'])/2.0,image_name=os.path.basename(images[i]),image_path = images[i], width=int(source_ins[i]['w']), height=int(source_ins[i]['h']),model="PINHOLE")]
            else:
                image_infos = [ImageInfo(uid=i,extrinsic=source_ext[i],rub=None),ImageInfo(uid=i+1,extrinsic=source_ext[i+1],rub=None)]
                cam_infos = [CameraInfo(uid=i, fx=float(source_ins[i]['fx']),fy=float(source_ins[i]['fy']),cx=int(source_ins[i]['w'])/2.0 ,cy=int(source_ins[i]['h'])/2.0,image_name=os.path.basename(images[i]),image_path = images[i], width=int(source_ins[i]['w']), height=int(source_ins[i]['h']),model="PINHOLE"),CameraInfo(uid=i+1, fx=float(source_ins[i+1]['fx']),fy=float(source_ins[i+1]['fy']),cx=int(source_ins[i+1]['w'])/2.0 ,cy=int(source_ins[i+1]['h'])/2.0,image_name=os.path.basename(images[i]),image_path = images[i], width=int(source_ins[i+1]['w']), height=int(source_ins[i+1]['h']),model="PINHOLE")]
            write_colmap_model(os.path.join(save_path,'sparse/0'),cam_infos,image_infos,'.jatxt')

        # HEADER = (f'extra information for judder_angle renders,cf={str(i)}')
        # np.savetxt(os.path.join(save_path,'sparse/0/ja_images.txt'),source_ext[i:i+2],header=HEADER)
        # np.savetxt(os.path.join(save_path,'sparse/0/ja_cameras.txt'),source_ins[i:i+2])
    print('finished')
    