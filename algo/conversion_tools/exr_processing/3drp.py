'''
Author: Qing Hong
Date: 2024-01-22 14:25:55
LastEditors: QingHong
LastEditTime: 2024-01-25 13:03:50
Description: file content
'''
'''
Author: Qing Hong
Date: 2024-01-22 14:25:55
LastEditors: QingHong
LastEditTime: 2024-01-25 11:31:16
Description: file content
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
from file_utils import mvwrite,read,custom_refine
from tqdm.contrib.concurrent import process_map
import warnings
import re
import argparse
DEPTH_MAX_VAL = 65535
TC_EPS = 0.00001
def gotrans(x):
        if len(x.shape)==4:
            tmp = x[...,3,:].copy()
            x[...,3,:] = x[...,2,:]
            x[...,2,:] = tmp
        else:
            tmp = x[...,3].copy()
            x[...,3] = x[...,2]
            x[...,2] = tmp
        return x

def is_inside_triangle(x, y,pTriVertices):
    ab_x = pTriVertices[1,0] - pTriVertices[0,0]
    ab_y = pTriVertices[1,1] - pTriVertices[0,1]
    bc_x = pTriVertices[2,0] - pTriVertices[1,0]
    bc_y = pTriVertices[2,1] - pTriVertices[1,1]
    ca_x = pTriVertices[0,0] - pTriVertices[2,0]
    ca_y = pTriVertices[0,1] - pTriVertices[2,1]

    ap_x = x - pTriVertices[0,0]
    ap_y = y - pTriVertices[0,1]
    bp_x = x - pTriVertices[1,0]
    bp_y = y - pTriVertices[1,1]
    cp_x = x - pTriVertices[2,0]
    cp_y = y - pTriVertices[2,1]

    f1 = ab_x * ap_y - ab_y * ap_x
    f2 = bc_x * bp_y - bc_y * bp_x
    f3 = ca_x * cp_y - ca_y * cp_x

    negtive_zero = 0
    positive_zero = 0

    if f1 == 0 and f2 == 0 and f3 == 0:
        return False

    if f1 >= negtive_zero and f2 >= negtive_zero and f3 >= negtive_zero:
        return True
    else:
        return False

def is_inside_triangle_batch(x, y,pTriVertices):
    ab_x = pTriVertices[...,1,0] - pTriVertices[...,0,0]
    ab_y = pTriVertices[...,1,1] - pTriVertices[...,0,1]
    bc_x = pTriVertices[...,2,0] - pTriVertices[...,1,0]
    bc_y = pTriVertices[...,2,1] - pTriVertices[...,1,1]
    ca_x = pTriVertices[...,0,0] - pTriVertices[...,2,0]
    ca_y = pTriVertices[...,0,1] - pTriVertices[...,2,1]

    ap_x = x - pTriVertices[...,0,0]
    ap_y = y - pTriVertices[...,0,1]
    bp_x = x - pTriVertices[...,1,0]
    bp_y = y - pTriVertices[...,1,1]
    cp_x = x - pTriVertices[...,2,0]
    cp_y = y - pTriVertices[...,2,1]

    f1 = ab_x * ap_y - ab_y * ap_x
    f2 = bc_x * bp_y - bc_y * bp_x
    f3 = ca_x * cp_y - ca_y * cp_x

    negtive_zero = 0
    positive_zero = 0
    res = np.zeros_like(x).astype('bool')
    res[np.where((f1 >= negtive_zero)&(f2 >= negtive_zero)&(f3>= negtive_zero))] = True
    res[np.where((f1==0)&(f2==0)&(f3==0))] = False
    return res


def compute_barycentric_2d(x, y, pTriVertices):
    v1 = pTriVertices[0]
    v2 = pTriVertices[1]
    v3 = pTriVertices[2]

    detT = (v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (v1[1] - v3[1])
    lamda1 = (v2[1] - v3[1]) * (x - v3[0]) + (v3[0] - v2[0]) * (y - v3[1])
    lamda2 = (v3[1] - v1[1]) * (x - v3[0]) + (v1[0] - v3[0]) * (y - v3[1])

    if np.abs(detT) <= 0.000001:
        lamda1 = 0
        lamda2 = 0
    else:
        lamda1 = lamda1 / detT
        lamda2 = lamda2 / detT

    lamda3 = 1 - lamda1 - lamda2

    return np.stack([lamda1, lamda2, lamda3],axis=-1)


def compute_barycentric_2d_batch(x, y, pTriVertices):
    v1 = pTriVertices[...,0,:]
    v2 = pTriVertices[...,1,:]
    v3 = pTriVertices[...,2,:]

    detT = (v2[...,1] - v3[...,1]) * (v1[...,0] - v3[...,0]) + (v3[...,0] - v2[...,0]) * (v1[...,1] - v3[...,1])
    lamda1 = (v2[...,1] - v3[...,1]) * (x - v3[...,0]) + (v3[...,0] - v2[...,0]) * (y - v3[...,1])
    lamda2 = (v3[...,1] - v1[...,1]) * (x - v3[...,0]) + (v1[...,0] - v3[...,0]) * (y - v3[...,1])
    lamda1[np.where(detT<= 0.000001)] = 0
    lamda2[np.where(detT<= 0.000001)] = 0
    tmp = np.where(detT> 0.000001)
    lamda1[tmp] = lamda1[tmp] / detT[tmp]
    lamda2[tmp] = lamda2[tmp] / detT[tmp]
    lamda3 = 1 - lamda1 - lamda2

    return np.stack([lamda1, lamda2, lamda3],axis=-1)

def depth_triangle_interp(d,pTriCoef):
    interp_o = 0
    alpha = pTriCoef[0]
    beta  = pTriCoef[1]
    gamma = pTriCoef[2]
    if (alpha<=1 and beta <=1 and alpha>=0 and beta>=0 and(alpha+beta)<=1):
        interp_o = alpha * d[0] + beta * d[1] + gamma * d[2]
    else:
        interp_o = min(min(d[0],d[1]),d[2])
    return interp_o

def frame_data_triangle_interp(pTriData, pTriCoef):
    alpha, beta, gamma = pTriCoef

    interp_o = [0, 0, 0, 0]

    if 0 <= alpha <= 1 and 0 <= beta <= 1 and alpha + beta <= 1:
        interp_o[0] = alpha * pTriData[0,0] + beta * pTriData[1,0] + gamma * pTriData[2,0]
        interp_o[1] = alpha * pTriData[0,1] + beta * pTriData[1,1] + gamma * pTriData[2,1]
        interp_o[2] = alpha * pTriData[0,2] + beta * pTriData[1,2] + gamma * pTriData[2,2]
        interp_o[3] = alpha * pTriData[0,3] + beta * pTriData[1,3] + gamma * pTriData[2,3]
    else:
        interp_o = pTriData[0]

    return interp_o

def mv_triangle_interp(pTriData, pTriCoef):
    alpha, beta, gamma = pTriCoef

    interp_o = [0, 0, 0]

    if 0 <= alpha <= 1 and 0 <= beta <= 1 and alpha + beta <= 1:
        interp_o[0] = alpha * pTriData[0,0] + beta * pTriData[1,0] + gamma * pTriData[2,0]
        interp_o[1] = alpha * pTriData[0,1] + beta * pTriData[1,1] + gamma * pTriData[2,1]
        interp_o[2] = alpha * pTriData[0,2] + beta * pTriData[1,2] + gamma * pTriData[2,2]
    else:
        interp_o = pTriData[0]

    return interp_o


def read_3drp(img_file,mv_file):
    pSRCIn = read(img_file,type='image')
    nHeight,nWidth = pSRCIn.shape[:2]
    pCurMVD = read(mv_file,type='flo')
    #denormalized
    pCurMVD[...,0] *= nWidth
    pCurMVD[...,1] *= nHeight
    pCurMVD = pCurMVD
    #no alpha
    ones_array = np.ones((nHeight, nWidth, 1)) * 255
    # 将原始数组和 ones_array 沿着第三个维度（通道维度）合并
    pSRCIn = np.concatenate((pSRCIn, ones_array), axis=2)
    return pCurMVD,pSRCIn
        
        

    # def motion_edge_det(self,):
    #     pMVIn = pCurMVD
    #     nHeight,nWidth = nHeight,nWidth
    #     x_gain = nWidth
    #     y_gain = nHeight
    #     # z_gain = 0
    #     # fg_bg_depth_thr = 1
    #     pMotEdge = np.zeros((nHeight,nWidth,2))
    #     for y in range(nHeight):
    #         for x in range(nWidth):
    #             nIndex = x + y * nWidth
    #             u = [x, x+1, x, x+1]
    #             v = [y, y, y+1, y+1]
    #             mv_ = np.zeros((4,4))
    #             depth_ = [0]*4
    #             for idx in range(4):
    #                 u[idx] = max(0, min(u[idx], nWidth - 1))
    #                 v[idx] = max(0, min(v[idx], nHeight - 1))
    #                 cx = u[idx]
    #                 cy = v[idx]
    #                 mv_[idx] = pMVIn[cy,cx]  # Assuming pMVIn is a list of MVD objects
    #                 depth_[idx] = mv_[idx,-1]
    #                 mv_[idx,0] /= x_gain
    #                 mv_[idx,1] /= y_gain
    #             edge_u = max(abs(mv_[0,0] - mv_[1,0]) * x_gain + abs(mv_[0,1] - mv_[1,1]) * y_gain,abs(mv_[0,0] - mv_[2,0]) * x_gain + abs(mv_[0,1] - mv_[2,1]) * y_gain)
    #             edge_u = max(edge_u, abs(mv_[2,0] - mv_[1,0]) * x_gain + abs(mv_[2,1] - mv_[1,1]) * y_gain)

    #             edge_l = max(abs(mv_[3,0] - mv_[1,0]) * x_gain + abs(mv_[3,1] - mv_[1,1]) * y_gain,abs(mv_[3,0] - mv_[2,0]) * x_gain + abs(mv_[3,1] - mv_[2,1]) * y_gain)
    #             edge_l = max(edge_l, abs(mv_[2,0] - mv_[1,0]) * x_gain + abs(mv_[2,1] - mv_[1,1]) * y_gain)

    #             pMotEdge[y,x,0] = edge_u > mv_edge_thr
    #             pMotEdge[y,x,1] = edge_l > mv_edge_thr
    
def motion_edge_det(pCurMVD):
    pMVIn = pCurMVD
    nHeight, nWidth = pCurMVD.shape[:2]
    x_gain = nWidth
    y_gain = nHeight
    pMotEdge = np.zeros((nHeight, nWidth, 2))

    # 创建网格表示每个像素位置
    y_grid, x_grid = np.mgrid[0:nHeight, 0:nWidth]

    # 创建 u 和 v 的网格
    u = np.stack([x_grid, x_grid + 1, x_grid, x_grid + 1], axis=-1)
    v = np.stack([y_grid, y_grid, y_grid + 1, y_grid + 1], axis=-1)

    # 限制 u 和 v 在图像范围内
    u = np.clip(u, 0, nWidth - 1)
    v = np.clip(v, 0, nHeight - 1)

    # 获取对应的运动矢量
    mv_ = pMVIn[v, u]

    # 计算归一化后的 x 和 y
    mv_[:, :, :, 0] /= x_gain
    mv_[:, :, :, 1] /= y_gain

    # 计算边缘强度
    edge_u = np.maximum(np.abs(mv_[:, :, 0, 0] - mv_[:, :, 1, 0]) * x_gain +
                        np.abs(mv_[:, :, 0, 1] - mv_[:, :, 1, 1]) * y_gain,
                        np.abs(mv_[:, :, 0, 0] - mv_[:, :, 2, 0]) * x_gain +
                        np.abs(mv_[:, :, 0, 1] - mv_[:, :, 2, 1]) * y_gain)
    edge_u = np.maximum(edge_u,
                        np.abs(mv_[:, :, 2, 0] - mv_[:, :, 1, 0]) * x_gain +
                        np.abs(mv_[:, :, 2, 1] - mv_[:, :, 1, 1]) * y_gain)

    edge_l = np.maximum(np.abs(mv_[:, :, 3, 0] - mv_[:, :, 1, 0]) * x_gain +
                        np.abs(mv_[:, :, 3, 1] - mv_[:, :, 1, 1]) * y_gain,
                        np.abs(mv_[:, :, 3, 0] - mv_[:, :, 2, 0]) * x_gain +
                        np.abs(mv_[:, :, 3, 1] - mv_[:, :, 2, 1]) * y_gain)
    edge_l = np.maximum(edge_l,
                        np.abs(mv_[:, :, 2, 0] - mv_[:, :, 1, 0]) * x_gain +
                        np.abs(mv_[:, :, 2, 1] - mv_[:, :, 1, 1]) * y_gain)

    # 设置运动边缘标记
    pMotEdge[:, :, 0] = edge_u > mv_edge_thr
    pMotEdge[:, :, 1] = edge_l > mv_edge_thr
    return pMotEdge
    
def core(img_file,mv_file,nPhaseIn=1):
    pMVIn,pDataIn = read_3drp(img_file,mv_file)
    pMotEdge = motion_edge_det(pMVIn)
    nHeight,nWidth = pDataIn.shape[:2]
    pZBufferIn = np.ones((nHeight,nWidth))*DEPTH_MAX_VAL
    pFIBuffer,pFIMVOut = np.zeros((nHeight,nWidth,4)),np.zeros((nHeight,nWidth,4))
    y_grid, x_grid = np.mgrid[0:nHeight, 0:nWidth]

    # 创建 u 和 v 的网格
    u = np.stack([x_grid, x_grid + 1, x_grid, x_grid + 1], axis=-1)
    v = np.stack([y_grid, y_grid, y_grid + 1, y_grid + 1], axis=-1)

    # 限制 u 和 v 在图像范围内
    u = np.clip(u, 0, nWidth - 1)
    v = np.clip(v, 0, nHeight - 1)

    # 获取对应的输入数据和运动矢量
    pInTriData = pDataIn[v, u]
    mv_ = pMVIn[v, u]

    # 计算输出顶点位置和运动向量插值
    off_x = 0
    off_y = 0
    pInTriVertex = np.stack([u, v], axis=-1)
    pOutTriVertex = pMVIn[v, u, :2] * nPhaseIn + pInTriVertex + np.array([off_x, off_y])

    mv_fi = pMVIn[v, u] * nPhaseIn
    mv_fi[...,3] = 0

    # 计算深度
    nDepthFI = np.minimum(mv_[...,3] + mv_[...,2] * nPhaseIn, DEPTH_MAX_VAL)
    
    i_grid, j_grid = np.meshgrid(np.arange(nHeight), np.arange(nWidth), indexing='ij')
    depth0 = np.min(mv_[..., :3, 3], axis=2) 
    depth0_num = np.argmin(mv_[..., :3, 3], axis=2)

    depth1 = np.min(mv_[..., 1:, 3], axis=2) 
    depth1_num = np.argmin(mv_[..., 1:, 3], axis=2)

    # depth0_max = mv_[...,-1].max()
    # depth0_num_max = np.unravel_index(mv_[...,:3].argmax(), mv_[...,:3].shape)
    
    isFIEnable_u = 1 - pMotEdge[...,0]
    isFIEnable_l = 1 - pMotEdge[...,1]
    
    # has_fb = False
    # up_alpha_en  = True
    # isFIEnable_u = isFIEnable_u and up_alpha_en
    nDepthFI[...,:3] = nDepthFI[...,:3].clip(0,DEPTH_MAX_VAL - 1)

    pOutTriVertex_l,mv_fi_l,nDepthFI_l,pInTriData_l = gotrans(pOutTriVertex.copy()),gotrans(mv_fi.copy()),gotrans(nDepthFI.copy()),gotrans(pInTriData.copy())
    nDepthFI_l = nDepthFI_l.clip(0,DEPTH_MAX_VAL - 1)


    FIEnable_u_pos = np.column_stack(np.where(isFIEnable_u == 0))
    FIEnable_l_pos = np.column_stack(np.where(isFIEnable_l == 0))
    pZBufferIn, pFIBuffer,pFIMVOut = MCRender(pInTriData, mv_fi,pOutTriVertex, nDepthFI,pZBufferIn, pFIBuffer,pFIMVOut,FIEnable_u_pos,offset=0,edge_flag=0)
    #m_nFillHoleWithFG
    for i in range(3):
        pOutTriVertex[...,i,:2] = mv_[...,i,:2] * nPhaseIn + pInTriVertex[...,i,:2]
        mv_fi[...,i,:3]=  mv_[i_grid, j_grid,depth0_num,:3] * nPhaseIn
        nDepthFI[...,i] = mv_[i_grid, j_grid,depth0_num,3] + mv_[i_grid, j_grid,depth0_num,2] * nPhaseIn
        pInTriData[...,i,:] = pInTriData[i_grid, j_grid,depth0_num]
    nDepthFI = nDepthFI.clip(-DEPTH_MAX_VAL,DEPTH_MAX_VAL)
    pZBufferIn, pFIBuffer,pFIMVOut = MCRender(pInTriData, mv_fi,pOutTriVertex, nDepthFI,pZBufferIn, pFIBuffer,pFIMVOut,FIEnable_u_pos,offset=0,edge_flag=1)
    # pFIMVOut[FIEnable_u_pos],pZBufferIn[FIEnable_u_pos], pFIBuffer[FIEnable_u_pos] = pFIMVOut_[FIEnable_u_pos],pZBufferIn_[FIEnable_u_pos], pFIBuffer_[FIEnable_u_pos]

    pZBufferIn, pFIBuffer,pFIMVOut = MCRender(pInTriData_l, mv_fi_l,pOutTriVertex_l, nDepthFI_l,pZBufferIn, pFIBuffer,pFIMVOut,FIEnable_l_pos,offset=1,edge_flag=0)
    #m_nFillHoleWithFG
    for i in range(1,4):
        pOutTriVertex_l[...,i,:2] = mv_[...,i,:2] * nPhaseIn + pInTriVertex[...,i,:2]
        mv_fi_l[...,i,:3]=  mv_[i_grid, j_grid,depth1_num,:3] * nPhaseIn
        nDepthFI_l[...,i] = mv_[i_grid, j_grid,depth1_num,3] + mv_[i_grid, j_grid,depth1_num,2] * nPhaseIn
        pInTriData_l[...,i,:] = pInTriData_l[i_grid, j_grid,depth1_num]
    pZBufferIn, pFIBuffer,pFIMVOut = MCRender(pInTriData_l, mv_fi_l,pOutTriVertex_l, nDepthFI_l,pZBufferIn, pFIBuffer,pFIMVOut,FIEnable_l_pos,offset=1,edge_flag=1)

    return pFIBuffer


def MCRender(pTriDataIn_, pTriMVFI_,pRenderTriVeterx_, pTriDepthFI_,pZBuffer, pDataOut,pMVOut,FIEnable_pos,offset=0,edge_flag=1):
    pTriDataIn, pTriMV,pRenderTriVeterx, pTriDepthFI = pTriDataIn_[...,offset:,:].copy(), pTriMVFI_[...,offset:,:].copy(),pRenderTriVeterx_[...,offset:,:].copy(), pTriDepthFI_[...,offset:].copy()
    nHeight,nWidth = pTriDataIn.shape[:2]
    x_min = np.min(pRenderTriVeterx[...,0],axis=2)
    x_max = np.max(pRenderTriVeterx[...,0],axis=2)
    y_min = np.min(pRenderTriVeterx[...,1],axis=2)
    y_max = np.max(pRenderTriVeterx[...,1],axis=2)

    pix_center = 0
    off_x = pix_center
    off_y = pix_center
    x_min_int = np.floor(x_min)
    y_min_int = np.floor(y_min)
    isXInt_ = np.abs(x_min - x_min_int)

    isXInt = np.where(isXInt_<TC_EPS)
    notXInt = np.where(isXInt_>=TC_EPS)


    isYInt_ = np.abs(y_min - y_min_int)

    isYInt = np.where(isYInt_<TC_EPS)
    notYInt = np.where(isYInt_>=TC_EPS)
    
    x_max_int = np.floor(x_max)
    y_max_int = np.floor(y_max)
    
    upleft_x = x_min_int
    upleft_y = y_min_int
    x_num = np.ones_like(upleft_x)
    y_num = np.ones_like(upleft_y)

    x_num[isXInt] = (x_max_int - x_min_int + 1)[isXInt]
    upleft_x[isXInt] = x_min_int[isXInt]
    x_num[notXInt] = (x_max_int - x_min_int)[notXInt]
    upleft_x[notXInt] = (x_min_int+1)[notXInt]
    
    y_num[isYInt] = (y_max_int - y_min_int + 1)[isYInt]
    upleft_y[isYInt] = y_min_int[isYInt]
    y_num[notYInt] = (y_max_int - y_min_int)[notYInt]
    upleft_y[notYInt] = (y_min_int+1)[notYInt]

    x_num = x_num.astype('uint8')
    y_num = y_num.astype('uint8')
    nHitNum = 0

    for i in range(nHeight):
        for j in range(nWidth):
            y_num_single = int(y_num[i,j])
            x_num_single = int(x_num[i,j])
            for ii in range(x_num_single):
                for jj in range(y_num_single):
                    x_out = int((upleft_x[i,j] + ii).clip(0,nWidth-1))
                    y_out = int((upleft_y[i,j] + jj).clip(0,nHeight-1))
                    if is_inside_triangle(x_out + off_x, y_out + off_y, pRenderTriVeterx[i,j]):
                        pTriCoef = compute_barycentric_2d(x_out + off_x, y_out + off_y, pRenderTriVeterx[i,j])
                        nInterpDepth = depth_triangle_interp(pTriDepthFI[i,j], pTriCoef)
                        if (nInterpDepth < pZBuffer[i,j] and edge_flag == 0):
                            pZBuffer[y_out,x_out]       = nInterpDepth
                            pDataOut[y_out,x_out]       = frame_data_triangle_interp(pTriDataIn[i,j], pTriCoef)
                            pMVOut[y_out,x_out,:3]      = mv_triangle_interp(pTriMV[i,j], pTriCoef)
                            pMVOut[y_out,x_out,-1]   = nInterpDepth

    # ii = np.arange(nWidth)
    # jj = np.arange(nHeight)[:, np.newaxis]  # 增加维度以匹配广播规则


    return pZBuffer, pDataOut,pMVOut

img_file = '/Users/qhong/Desktop/1220test/image/1.FinalImage.0010.png'
mv_file = '/Users/qhong/Desktop/1220test/mv0/mv_00000010.exr'


nPhaseIn=1
mv_edge_thr = 2
pCurMVD = None
pSRCIn = None
h,w = None,None
ObjectRPEn = True
res = core(img_file,mv_file)

