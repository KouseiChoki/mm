from tqdm import tqdm
import sys,os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/../..')
from tqdm.contrib.concurrent import process_map
from dataclasses import dataclass
from fileutil.read_write_model import Camera,write_model,Image
from file_utils import mvwrite,read
import cv2
import numpy as np
from cal_ply import generate_point_cloud_from_depth
from plyfile import PlyData, PlyElement
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

def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))
def mkdir(path):
        if  not os.path.exists(path):
            os.makedirs(path,exist_ok=True)
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


root = '/Users/qhong/Desktop/avatar_data/2039/1120'
downscale = 1
mode3d = True
output = os.path.join(root,'newcloud')
mkdir(output)
output = os.path.join(output,'points3D.ply')

if os.path.isdir(os.path.join(root,'colmap')):
    root = os.path.join(root,'colmap')
elif os.path.isdir(os.path.join(root,'pointcloud')):
    root = os.path.join(root,'pointcloud')
sproot = os.path.join(root,'sparse','0')
intr = read_intrinsics_text(os.path.join(sproot,'cameras.txt'))
extr = read_extrinsics_text(os.path.join(sproot,'images.txt'))
dpfile = os.path.join(root,'depths')
imgs = jhelp_file(os.path.join(root,'images'))
tmp = os.path.join(root,'masks')
masks = jhelp_file(tmp) if os.path.isdir(tmp) else []
dps = jhelp_file(dpfile)
points,rgbs = [],[]

for i in tqdm(range(len(dps))):
    depth = 1/read(dps[i])[...,0]
    h,w = depth.shape
    focal_length_x = intr[i+1].params[0] /downscale
    focal_length_y = intr[i+1].params[1]/downscale
    o_cx = intr[i+1].params[2]//downscale
    o_cy = intr[i+1].params[3]//downscale
    target_w = w//downscale
    target_h = h//downscale
    depth = cv2.resize(depth,(target_w,target_h),interpolation=cv2.INTER_NEAREST)
    #prune unvalid depth
    # depth[np.where(depth>depth.mean()*10)] = 0
    # cloud point
    R = np.transpose(qvec2rotmat(extr[i+1].qvec))
    T = np.array(extr[i+1].tvec)
    W2C = getWorld2View2(R,T)
    C2W = np.linalg.inv(W2C)
    extrinsics = C2W
    # print(C2W)
    #内参做了归一化
    intrinsics = np.array([[focal_length_x,0,o_cx],[0,focal_length_y,o_cy],[0,0,1]])
    point = generate_point_cloud_from_depth(depth,intrinsics,extrinsics)
    #prune unvailid point
    # rgb = cv2.resize(image,(target_w,target_h))
    #删除非法depth
    rgb = read(imgs[i],type='image')
    rgb = cv2.resize(rgb,(target_w,target_h))
    rgb = rgb[depth!= 0]
    if len(masks)>0:
        if mode3d and i == 1:
            pass
        else:
            mask_path = masks[i]
            mask = read(mask_path,type='mask')
            if downscale != 1 :
                mask = cv2.resize(mask,(target_w,target_h),interpolation=cv2.INTER_NEAREST).reshape(-1)
            else:
                mask = mask.reshape(-1)
            rgb = rgb[mask == 0]
            point = point[mask == 0]
    #Mask
    # if args.mask:
    #     mask_path = os.path.join(save_path,'Mask',os.path.basename(oris[i]))
    #     mask = read(mask_path,type='mask')
    #     mask = cv2.resize(mask,(target_w,target_h),interpolation=cv2.INTER_NEAREST).reshape(-1)
    #     rgb = rgb[mask == 0]
    #     point = point[mask == 0]
    points.append(point.reshape(-1,3))
    rgbs.append(rgb.reshape(-1,3))
xyz = np.concatenate(points)
rgbs = np.concatenate(rgbs)

print('writing plyfile........')
normals = np.zeros_like(xyz)
elements = np.empty(xyz.shape[0], dtype=dtype)
attributes = np.concatenate((xyz, normals, rgbs), axis=1)
elements[:] = list(map(tuple, attributes))
vertex_element = PlyElement.describe(elements, "vertex")
ply_data = PlyData([vertex_element])
ply_data.write(output)