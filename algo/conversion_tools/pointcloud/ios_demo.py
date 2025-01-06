import os,sys
import imageio
import numpy as np
import cv2
import torch
from depth_utils import unproject_depth
import trimesh
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/../../')
import shutil
from file_utils import jhelp_file,read,mkdir
import open3d as o3d
import skvideo
from PIL import Image
from tqdm import tqdm
from cal_ply import ImageInfo,CameraInfo,write_colmap_model
from scipy.spatial.transform import Rotation as R
DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
MAX_DEPTH = 20.0

import collections
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
class CImage(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
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
                images[image_id] = CImage(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

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

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def check_is_stray_scanner_app_capture(input_dir):
    assert os.path.exists(os.path.join(input_dir, 'rgb.mp4')), 'rgb.mp4 not found'


def ensure_multiple_of(x, multiple_of=14):
    return int(x // multiple_of * multiple_of)


def to_tensor_func(arr):
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def to_numpy_func(tensor):
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr

def load_image(image_path, to_tensor=True, max_size=1008, multiple_of=14):
    '''
    Load image from path and convert to tensor
    max_size // 14 = 0
    '''
    image = np.asarray(imageio.imread(image_path)).astype(np.float32)
    image = image / 255.

    max_size = max_size // multiple_of * multiple_of
    if max(image.shape) > max_size:
        h, w = image.shape[:2]
        scale = max_size / max(h, w)
        tar_h = ensure_multiple_of(h * scale)
        tar_w = ensure_multiple_of(w * scale)
        image = cv2.resize(image, (tar_w, tar_h), interpolation=cv2.INTER_AREA)
    if to_tensor:
        return to_tensor_func(image)
    return image

def load_depth(path, confidence=None, filter_level=0):
    if path[-4:] == '.npy':
        depth_mm = np.load(path)
    elif path[-4:] == '.png':
        depth_mm = np.array(Image.open(path))
    depth_m = depth_mm.astype(np.float32) / 1000.0
    if confidence is not None:
        depth_m[confidence < filter_level] = 0.0
    return o3d.geometry.Image(depth_m)

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


def _resize_camera_matrix(camera_matrix, scale_x, scale_y):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    return np.array([[fx * scale_x, 0.0, cx * scale_x],
        [0., fy * scale_y, cy * scale_y],
        [0., 0., 1.0]])

def get_intrinsics(intrinsics):
    """
    Scales the intrinsics matrix to be of the appropriate scale for the depth maps.
    """
    intrinsics_scaled = _resize_camera_matrix(intrinsics, DEPTH_WIDTH / 1920, DEPTH_HEIGHT / 1440)
    return o3d.camera.PinholeCameraIntrinsic(width=DEPTH_WIDTH, height=DEPTH_HEIGHT, fx=intrinsics_scaled[0, 0],
        fy=intrinsics_scaled[1, 1], cx=intrinsics_scaled[0, 2], cy=intrinsics_scaled[1, 2])



def load_confidence(path):
    return np.array(Image.open(path))


# def point_clouds(rgbs,depths,confidences,ext,intrinsics,step=1):
#     """
#     Converts depth maps to point clouds and merges them all into one global point cloud.
#     flags: command line arguments
#     data: dict with keys ['intrinsics', 'poses']
#     returns: [open3d.geometry.PointCloud]
#     """
#     pc = o3d.geometry.PointCloud()
#     # for i, (T_WC, rgb) in enumerate(ext,rgbs):
#     for i in range(0,len(rgbs),step):
#         print(f"Point cloud {i}", end="\r")
#           
#         confidence = load_confidence(confidences[i])
#         depth_path = depths[i]
#         depth = load_depth(depth_path, confidence, filter_level=1)
#         rgb = read(rgbs[i],type='image')
#         rgb = cv2.resize(rgb,(DEPTH_WIDTH, DEPTH_HEIGHT),interpolation=cv2.INTER_NEAREST)
#         # depth = cv2.resize(depth,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)
#         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#             o3d.geometry.Image(rgb), depth,
#             depth_scale=1.0, depth_trunc=MAX_DEPTH, convert_rgb_to_intensity=False)
#         pc += o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics, extrinsic=T_CW)
#     return pc

def point_clouds(rgbs,depths,confidences,ext,intrinsics,step=1):
    """
    Converts depth maps to point clouds and merges them all into one global point cloud.
    flags: command line arguments
    data: dict with keys ['intrinsics', 'poses']
    returns: [open3d.geometry.PointCloud]
    """
    pc = o3d.geometry.PointCloud()
    tmp = []
    colors = []
    # for i, (T_WC, rgb) in enumerate(ext,rgbs):
    for i in range(0,len(rgbs),step):
        print(f"Point cloud {i}", end="\r")
        # T_CW = np.linalg.inv(ext[i])
        confidence = load_confidence(confidences[i])
        depth_path = depths[i]
        depth = np.asarray(load_depth(depth_path, confidence, filter_level=1))
        rgb = read(rgbs[i],type='image')
        rgb = cv2.resize(rgb,(DEPTH_WIDTH, DEPTH_HEIGHT),interpolation=cv2.INTER_NEAREST)
        # depth = cv2.resize(depth,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)
        t = generate_point_cloud_from_depth(depth,intrinsics.intrinsic_matrix,ext[i])
        tmp.append(t.reshape(-1,3))
        colors.append(rgb[depth!=0].reshape(-1,3))
    # print(np.concatenate(tmp).shape,np.concatenate(colors).shape)
    pc.points = o3d.utility.Vector3dVector(np.concatenate(tmp))
    pc.colors = o3d.utility.Vector3dVector(np.concatenate(colors)/255)
    return pc

choices=['756x1008', '1428x1904']
input_dir = '/Users/qhong/Desktop/1224/e57a32120d'
sp = os.path.join(input_dir,'pointcloud')
check_is_stray_scanner_app_capture(input_dir)
step = 10
# extract rgb images
os.makedirs(os.path.join(input_dir, 'rgb'), exist_ok=True)
if len(jhelp_file(os.path.join(input_dir, 'rgb')))==0:  
    cmd = f'ffmpeg -i {input_dir}/rgb.mp4 -q:v 2 {input_dir}/rgb/%06d.jpg'
    os.system(cmd)

# Loading & Inference
image_path = os.path.join(input_dir, 'rgb')
images = jhelp_file(image_path)
depths = jhelp_file(os.path.join(input_dir, 'depth'))
confidences = jhelp_file(os.path.join(input_dir, 'confidence'))
odometry = np.loadtxt(os.path.join(input_dir, 'odometry.csv'), delimiter=',', skiprows=1)
poses = []
for line in odometry:
    # timestamp, frame, x, y, z, qx, qy, qz, qw
    position = line[2:5]
    quaternion = line[5:]
    T_WC = np.eye(4)
    T_WC[:3, :3] = Rotation.from_quat(quaternion).as_matrix().transpose()
    T_WC[:3, 3] = position
    # T_WC = np.linalg.inv(T_WC)
    poses.append(T_WC)
pts,cls = [],[]

ixt_path = os.path.join(input_dir, f'camera_matrix.csv')
ixt = np.loadtxt(ixt_path, delimiter=',')
intrinsics = get_intrinsics(ixt)
pcd = point_clouds(images,depths,confidences,poses,intrinsics,step=10)
zeros = np.zeros((len(pcd.points), 3))  # 与点数量一致的零向量
pcd.normals = o3d.utility.Vector3dVector(zeros)

sparse_path = os.path.join(sp,'sparse/0')
ply_path = os.path.join(sparse_path , "points3D.ply")
mkdir(os.path.join(sp , "images"))

h,w,_ = read(images[0],type='image').shape
image_infos,cam_infos = [],[]
for i in range(0,len(images),step):
    # _,_,tvec0,tvec1,tvec2,qx, qy, qz, qw = odometry[i]
    # tvec0,tvec1,tvec2 = w2c[:3, 3]
    # w2c = poses[i]
    w2c = np.linalg.inv(poses[i])
    qx, qy, qz ,qw = R.from_matrix(w2c[:3, :3]).as_quat()
    tvec0,tvec1,tvec2 = w2c[:3, 3]
    image_info = ImageInfo(uid=i+1,extrinsic=np.array([qw,qx,qy,qz,tvec0,tvec1,tvec2]),rub=None)
    image_infos.append(image_info)
    cam_info = CameraInfo(uid=i+1, fx=ixt[0,0],fy=ixt[1,1],cx=ixt[2,0] ,cy=ixt[2,1],image_name=os.path.basename(images[i]),image_path = images[i], width=w, height=h,model="PINHOLE")
    cam_infos.append(cam_info)
    shutil.copy(images[i], os.path.join(sp , "images",os.path.basename(images[i])))

mkdir(sparse_path)
write_colmap_model(sparse_path,cam_infos,image_infos)
o3d.io.write_point_cloud(ply_path, pcd)