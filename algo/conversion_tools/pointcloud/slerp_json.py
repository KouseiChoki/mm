import json
import numpy as np
from scipy.spatial.transform import Rotation as R
# 读取 JSON 文件
def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # 解析 JSON 数据
            return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到.")
    except json.JSONDecodeError:
        print("JSON 解析出错.")
    except Exception as e:
        print(f"发生错误: {e}")

def ja_ajust(prev_extrinsic,extrinsic,ja):
    # prev_extrinsic = prev_image_info.extrinsic
    # extrinsic = image_info.extrinsic
    new_q = slerp(prev_extrinsic[:4],extrinsic[:4],t=ja)
    new_t = interpolate_translation(prev_extrinsic[4:],extrinsic[4:],t=ja)
    #change uid and name
    new_extrinsic=np.concatenate([new_q,new_t])
    return new_extrinsic

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

def write_json(sp,datas):
    with open(sp, 'w') as file:
        json.dump(datas, file)

if __name__ == "__main__":
    file_path = '/Users/qhong/Desktop/0913/cameras.json'  # 替换为你的 JSON 文件路径
    json_datas = read_json(file_path)
    start_num = 0
    split_num = 4
    res_data = []
    id = 0
    for i in range(len(json_datas)-start_num):
        jd = json_datas[i+start_num]
        # res_data.append(jd)
        # jd_next = json_datas[i+start_num+1]
        W2C = np.eye(4)
        rot = np.array(jd['rotation']).transpose()
        pos = np.array(jd['position'])
        W2C[:3,:3] = rot
        W2C[:3, 3] = pos
        # W2C = np.linalg.inv(Rt)
        # qx, qy, qz ,qw = R.from_matrix(W2C[:3, :3]).as_quat()
        qx, qy, qz ,qw = rotmat2qvec(W2C[:3, :3])
        tx,ty,tz = W2C[:3, 3]
        res_data.append([qx,qy,qz,qw,tx,ty,tz])

    final_res = []
    for i in range(len(res_data)-1):
        dof = np.array(res_data[i])
        dof_next = np.array(res_data[i+1])
        final_res.append(dof)
        for step in range(split_num-1):
            extr = ja_ajust(dof,dof_next,(1+step)/split_num)
            final_res.append(extr)
    final_res.append(np.array(res_data[-1]))
    
    width = jd['width']
    height = jd['height']
    sp = file_path.replace('cameras.json',f'cameras_step_{split_num}.json')
    res_json = []
    for i in range(len(final_res)):
        js = {}
        data = final_res[i]
        js['id'] = i
        js['img_name'] = str(i)
        js['width'] = jd['width']
        js['height'] = jd['height']

        R_ = np.transpose(qvec2rotmat(data[:4]))
        T_ = data[4:]
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R_.transpose()
        Rt[:3, 3] = T_
        Rt[3, 3] = 1.0
        W2C = np.linalg.inv(Rt)
        p = W2C[:3, 3]
        r = W2C[:3, :3]
        js['rotation'] = r.tolist()
        # js['position'] = p.tolist()
        js['position'] = T_.tolist()
        js['fx'] = jd['fx']
        js['fy'] = jd['fy']

        res_json.append(js)
    write_json(sp,res_json)


