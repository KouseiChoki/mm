'''
Author: Qing Hong
Date: 2022-09-26 15:01:24
LastEditors: QingHong
LastEditTime: 2023-07-26 13:34:29
Description: file content
'''
import os,sys
from tqdm import tqdm
import numpy as np 
import re
from glob import glob
import os.path as osp
TAG_CHAR = np.array([202021.25], np.float32)

def mkdir(path):
        if  not os.path.exists(path):
            os.makedirs(path,exist_ok=True)

def jhelp(c):
    return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:       # little-endian
        endian = '<'
        scale = -scale
    else:               # big-endian
        endian = '>'

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert uv.ndim == 3
        assert uv.shape[2] == 2
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert u.shape == v.shape
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:, np.arange(width)*2] = u
    tmp[:, np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

if __name__ == '__main__':
    # a = [os.path.join(root,i) for i in sorted(list(filter(lambda x:x[0]!='.',os.listdir(root))))]
    # assert len(sys.argv)==3 ,'usage: python exr_resize_mergedepth.py root save_path'
    root = sys.argv[1]
    for split_dir in ['TEST','TRAIN']:
        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                flow_dirs = sorted(glob(osp.join(root, f'optical_flow/{split_dir}/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])
                for  fdir in flow_dirs:
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in tqdm(range(len(flows))):
                        flow_ = flows[i]
                        flow = readPFM(flow_).astype(np.float32)
                        if len(flow.shape) != 2:
                            flow =  flow[:, :, :-1]
                        sp = flow_.replace('.pfm','.flo').replace('/flyingchairs/','/flyingchairs_flos/')
                        if not os.path.exists(os.path.dirname(sp)):  # 判断目录是否存在
                            os.makedirs(os.path.dirname(sp),exist_ok=False) 
                        writeFlow(sp,flow)

                    

    
    