'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-08-16 12:32:31
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

import os
import time
import shutil
import numpy as np
from scipy.optimize import leastsq
from skimage.metrics import peak_signal_noise_ratio
import imageio,cv2
from tqdm import tqdm
from PIL import Image
cur_time = str(time.gmtime().tm_mon) + '_' + str(time.gmtime().tm_mday)
cur_time_sec = cur_time+'/'+str(time.gmtime().tm_hour)+'/'+str(time.gmtime().tm_sec)
import torch.nn.functional as F
from file_utils import *
import torch
import sys
LUT = None

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == 'kitti400':
            self._pad = [0, 0, 0, 400 - self.ht]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def build_dir(mk_dir,algo):
    if  not os.path.exists(mk_dir):
        os.makedirs(mk_dir)

    save_dir =  mk_dir + '/' + algo +'/'+ cur_time_sec
    if  os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    print('saving file created:{}'.format(save_dir))

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

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


# def imresize(image, ratio=None, out_size=None, method='bicubic', start='auto,auto', out_offset=None, padding='symmetric', clip=True):
#     '''
#     Parameters
#     ----------
#     image     : ndarray, 1 channel or n channels interleaved
#     ratio     : scale ratio. It can be a scalar, or a list/tuple/numpy.array.
#                 If it's a scalar, the ratio applies to both H and V.
#                 If it's a list/numpy.array, it specifies the hor_ratio and ver_ratio.
#     out_size  : output size [wo, ho]
#     method    : 'bicubic' | 'bilinear' | 'nearest'
#     start     : string seperated by ',' specify the start position of x and y
#     out_offset: offset at output domain [xoffset, yoffset]
#     padding   : 'zeros' | 'edge','replicate','border' | 'symmetric'. Default: 'symmetric' (TBD)
#     clip      : only effect for float image data (uint8/uint16 image output is alway clipped)

#     Returns
#     -------
#     result: ndarray

#     History
#     2021/07/10: changed ratio order [H,W] -> [W,H]
#                 add out_offset
#     2021/07/11: add out_size
#     2021/07/31: ratio cannot be used as resolution any more

#     Notes：
#     如果 ratio 和 out_size 都没有指定，则 ratio = 1
#     如果只指定 out_size，则 ratio 按输入图像尺寸和 out_size 计算
#     如果只指定 ratio，则输出尺寸为输入图像尺寸和 ratio 的乘积并四舍五入
#     如果同时指定 ratio 和 out_size，则按  ratio 输出 out_size 大小的图，这时既保证 ratio，也保证输出图像尺寸
#     '''
#     startx, starty = start.split(',')
#     ih, iw = image.shape[:2]

#     if ratio is None:
#         ratio = 1 if out_size is None else [out_size[0]/iw, out_size[1]/ih]

#     if isinstance(ratio, list) or isinstance(ratio, np.ndarray) or isinstance(ratio, tuple):
#         hratio, vratio = ratio[0], ratio[1]
#     else:
#         hratio, vratio = ratio, ratio

#     if out_offset is None: out_offset = (0, 0)
#     if out_size   is None: out_size   = (None, None)

#     if method == 'bicubic':
#         outv = ver_interp_bicubic(image, vratio, out_size[1], starty, out_offset[1], clip)
#         out  = hor_interp_bicubic(outv, hratio, out_size[0], startx, out_offset[0], clip)
#     else:
#         xinc, yinc = 1/hratio, 1/vratio
#         ow = round(iw * hratio) if out_size[0] is None else out_size[0]
#         oh = round(ih * vratio) if out_size[1] is None else out_size[1]
#         x0 = (-.5 + xinc/2 if startx == 'auto' else float(startx)) + out_offset[0] * xinc # (x0, y0) is in input domain
#         y0 = (-.5 + yinc/2 if starty == 'auto' else float(starty)) + out_offset[1] * yinc 
#         x = x0 + np.arange(ow) * xinc
#         y = y0 + np.arange(oh) * yinc
#         xaux = np.r_[np.arange(iw), np.arange(iw-1,-1,-1)] # 0, 1, ..., iw-2, iw-1, iw-1, iw-2, ..., 1, 0
#         yaux = np.r_[np.arange(ih), np.arange(ih-1,-1,-1)]
#         if method == 'nearest':
#             x = np.floor(x + .5).astype('int32') # don't use np.round() as it rounds to even value (w,)
#             y = np.floor(y + .5).astype('int32')
#             xind = xaux[np.mod(np.int32(x), xaux.size)]
#             yind = yaux[np.mod(np.int32(y), yaux.size)]
#             out = image[np.ix_(yind, xind)]
#         elif method == 'bilinear':
#             tlx = np.floor(x).astype('int32')
#             tly = np.floor(y).astype('int32')
#             wy, wx = np.ix_(y - tly, x - tlx) # wy: (h, 1), wx: (1, w)
#             brx = xaux[np.mod(tlx + 1, xaux.size)]
#             bry = yaux[np.mod(tly + 1, yaux.size)]
#             tlx = xaux[np.mod(tlx    , xaux.size)]
#             tly = yaux[np.mod(tly    , yaux.size)]
#             if image.ndim == 3:
#                 wy, wx = wy[..., np.newaxis], wx[..., np.newaxis]
#             out = (image[np.ix_(tly, tlx)] * (1-wx) * (1-wy) + image[np.ix_(tly, brx)] * wx * (1-wy)
#                  + image[np.ix_(bry, tlx)] * (1-wx) *    wy  + image[np.ix_(bry, brx)] * wx *    wy)
#         else:
#             print('Error: Bad -method argument {}. Must be one of \'bilinear\', \'bicubic\', and \'nearest\''.format(method))
#         if   image.dtype == 'uint8' : out = np.uint8(out + 0.5)
#         elif image.dtype == 'uint16': out = np.uint16(out + 0.5)
#     return out

    
# import png
# def saveUint16(path,z):
#     # Use pypng to write zgray as a grayscale PNG.
#     with open(path, 'wb') as f:
#         writer = png.Writer(width=z.shape[1], height=z.shape[0], bitdepth=16, greyscale=True)
#         zgray2list = z.tolist()
#         writer.write(f, zgray2list)

# def depthToint16(dMap, minVal=0, maxVal=10):
#     #Maximum and minimum distance of interception 
#     dMap[dMap>maxVal] = maxVal
#     # print(np.max(dMap),np.min(dMap))
#     dMap = ((dMap-minVal)*(pow(2,16)-1)/(maxVal-minVal)).astype(np.uint16)
#     return dMap

# def normalizationDepth(depthfile, savepath):
#     correctDepth = readDepth(depthfile)
#     depth = depthToint16(correctDepth, 0, 10)
#     saveUint16(depth,savepath)


def immc(input, mv,mvoffset=None):
    indtype = input.dtype
    input = input.astype(np.float32)
    output = np.zeros_like(input, dtype=np.float32)
    h, w = input.shape[:2]
    if mvoffset is None:
        mvoffset = 0,0
    ratio = (input.shape[0] + mv.shape[0] - 1) // mv.shape[0]
    mv = mv.repeat(ratio, axis=0).repeat(ratio, axis=1)[:h, :w]

    mvoffset = mvoffset[0]*ratio, mvoffset[1]*ratio

    if mvoffset[0] != 0 or mvoffset[1] != 0:
        mv[:,:] += mvoffset

    mv_i = np.floor(mv).astype(np.intp)
    mv_f = mv - mv_i
    w0 = (1 - mv_f[...,0]) * (1 - mv_f[...,1])
    w1 = (    mv_f[...,0]) * (1 - mv_f[...,1])
    w2 = (1 - mv_f[...,0]) * (    mv_f[...,1])
    w3 = (    mv_f[...,0]) * (    mv_f[...,1])
    if input.ndim == 3:
        w0, w1, w2, w3 = w0[...,np.newaxis], w1[...,np.newaxis], w2[...,np.newaxis], w3[...,np.newaxis]
    y, x = np.ix_(np.arange(h, dtype=np.intp), np.arange(w, dtype=np.intp)) # y: (h,1), x: (1, w)
    x0 = x + mv_i[...,0]
    y0 = y + mv_i[...,1]
    x1 = (x0 + 1).clip(0, w-1)
    y1 = (y0 + 1).clip(0, h-1)
    x0 = x0.clip(0, w-1)
    y0 = y0.clip(0, h-1)
    output = input[y0, x0] * w0 + input[y0, x1] * w1 + input[y1, x0] * w2 + input[y1, x1] * w3

    return output.astype(indtype)

def fitting_func(p, x):
        """
        获得拟合的目标数据数组
        :param p: array[int] 多项式各项从高到低的项的系数数组
        :param x: array[int] 自变量数组
        :return: array[int] 拟合得到的结果数组
        """
        f = np.poly1d(p)    # 获得拟合后得到的多项式
        return f(x)         # 将自变量数组带入多项式计算得到拟合所得的目标数组

def error_func(p, x, y):
    """
    计算残差
    :param p: array[int] 多项式各项从高到低的项的系数数组
    :param x: array[int] 自变量数组
    :param y: array[int] 原始目标数组(因变量)
    :return: 拟合得到的结果和原始目标的差值
    """
    err = fitting_func(p, x) - y
    return err

def n_poly(n, x, y):
    """
    n 次多项式拟合函数
    :param n: 多项式的项数(包括常数项)，比如n=3的话最高次项为2次项
    :return: 最终得到的系数数组
    """
    p_init = np.random.randn(n)   # 生成 n个随机数，作为各项系数的初始值，用于迭代修正
    parameters = leastsq(error_func, p_init, args=(np.array(x), np.array(y)))    # 三个参数：误差函数、函数参数列表、数据点
    return parameters[0]	# leastsq返回的是一个元组，元组的第一个元素时多项式系数数组[wn、...、w2、w1、w0]


def cal_epe(image,target_image):
    target_image = target_image.astype('float32')
    image = image.astype('float32')
    epe  = np.abs(target_image.astype('float32')-image.astype('float32'))
    return epe.mean()

def cal_psnr(image,target_image):
    psnr=peak_signal_noise_ratio(image,target_image)
    return -psnr

def cpu_algorithm(algorithm):
    if algorithm.lower() in ['farneback','deepflow','simpleflow','sparse_to_dense_flow','pca_flow','rlof'] or 'cpu'  in algorithm.lower():
        return True
    return False

def get_board_length(image):
    h,w,c = image.shape
    res = [0,h,0,w,h,w]
    if all(image[0,0] == [0,0,0]):
        for i in range(h):
            if image[i].sum()>h*10:
                break
            res[0] = i+1
        for i in range(w):
            if image[:,i].sum()>w*10:
                break
            res[2] = i+1
    if all(image[-1,-1] == [0,0,0]):
        for i in range(h-1,-1,-1):
            if image[i].sum()>h*10:
                break
            res[1] = i
        for i in range(w-1,-1,-1):
            if image[:,i].sum()>w*10:
                break
            res[3] = i
    return res

def create_lut(root):
    import colour
    from colour.algebra import table_interpolation_trilinear


def img_equal(img1,img2):
    return (img1==img2).all()
    
def interpolation_err(im1,im2):
    diff_rgb = 128.0 + im2 - im1
    ie = np.mean(np.mean(np.mean(np.abs(diff_rgb - 128.0))))
    return ie

'''
description: 根据mask计算最优mv并融合
param {*} best_lr_mv0 当前mv
param {*} lr0_other 其他mv
param {*} image
param {*} mask
return {*} 融合最优解的mv,以及选取信息
'''
def compare_with(mv,mv_other,image,mask,test=False):
    if mask.shape[2] == 3:
        mask_d = np.concatenate((mask,mask.copy()[...,0][...,None]),axis=2)
    else:
        mask_d = mask
    result = np.zeros_like(mv,dtype='float32')
    result_info = [0,0]
    recovered_image = immc(image,mv)
    recovered_image_other = immc(image,mv_other)
    th = mask.mean()
    front = np.where(mask>th)
    back = np.where(mask<=th)
    front_d = np.where(mask_d>th)
    back_d = np.where(mask_d<=th)
    #计算前景
    image_front = np.zeros_like(image)
    image_front[front] = image[front]

    recovered_image_front = np.zeros_like(recovered_image)
    recovered_image_front[front] = recovered_image[front]

    recovered_image_other_front = np.zeros_like(recovered_image_other)
    recovered_image_other_front[front] = recovered_image_other[front]

    #计算前景inter error
    ie_front = interpolation_err(image_front,recovered_image_front)
    ie_front_other = interpolation_err(image_front,recovered_image_other_front)
    if test:
        print(f'ie_front:{ie_front},ie_front_other:{ie_front_other}')
    if ie_front<=ie_front_other:
        result[front_d] = mv[front_d]
    else:
        result[front_d] = mv_other[front_d]
        result_info[0] = 1 #用于提示用了哪一个
    
    #同样 计算背景
    image_back = np.zeros_like(image)
    image_back[back] = image[back]

    recovered_image_back = np.zeros_like(recovered_image)
    recovered_image_back[back] = recovered_image[back]

    recovered_image_other_back = np.zeros_like(recovered_image_other)
    recovered_image_other_back[back] = recovered_image_other[back]

    #计算前景inter error
    ie_back = interpolation_err(image_back,recovered_image_back)
    ie_back_other = interpolation_err(image_back,recovered_image_other_back)
    if test:
        print(f'ie_back:{ie_back},ie_back_other:{ie_back_other}')
    if ie_back<=ie_back_other:
        result[back_d] = mv[back_d]
    else:
        result[back_d] = mv_other[back_d]
        result_info[1] = 1 #用于提示用了哪一个
    return result,result_info

'''
description: 膨胀操作
param {*} image
param {*} kernel_size 核大小
return {*}
'''
def custom_dilatation(image,kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    #膨胀
    dst = cv2.dilate(image, kernel)
    return dst

'''
description: 腐蚀操作
param {*} image
param {*} kernel_size 核大小
return {*}
'''
def custom_erodition(image,kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    #腐蚀
    dst = cv2.erode(image, kernel)
    return dst

'''
description: 图像后处理
param {*} image
param {*} dilatation
param {*} erodition
return {*}
'''
def reprocessing(image,args,mask=None):
    #膨胀腐蚀
    if args.dilatation>0:
        image = custom_dilatation(image,args.dilatation)
    if args.erodition>0:
        image = custom_erodition(image,args.erodition)
    #边缘滤波
    # if args.edge_filter and mask is not None:
    #     image = edge_filter(image,mask,args)
    return image

'''
description:  mask 灰度图
return {*}
'''
def mask_to_gray(mask,reverse=True):
    mask = np.array(mask)
    #类别判断
    if len(mask.shape)==3:
        mask = mask[...,0]
    #黑白处理
    th = mask.mean()
    if mask.max()<=1:
        mask *= 255
    if reverse:
        mask[np.where(mask<=th)] = 1
        mask[np.where(mask>th)] = 0
    else:
        mask[np.where(mask<=th)] = 0
        mask[np.where(mask>th)] = 1
    return mask*255

'''
description: 
param {*} mask
param {*} lf low 阈值
param {*} hf high 阈值
return {*}
'''
def mask_to_edges(mask,lf=100,hf=200):
    mask = mask_to_gray(mask)
    #轮廓获取
    edges = cv2.Canny(mask,lf,hf)
    return np.array(edges,dtype='uint8')

'''
description:  获取边缘区域坐标
param {*} mask
param {*} distance
param {*} lt_hre
param {*} h_thre
return {*}
'''
def custom_rdp(mask_gray,distance=100,l_thre=0.2,h_thre=0.8):
    kernel = np.ones((distance, distance), np.float32)
    res = cv2.filter2D(src=mask_gray.astype('float32'), ddepth=-1, kernel=kernel)
    roi = res.copy()
    roi[np.where(roi<distance**2*l_thre)] = 0
    roi[np.where(roi>distance**2*h_thre)] = 0
    roi[np.where(mask_gray!=0)] = 0
    # roi = np.repeat(roi[...,None],4,axis=2)
    roi[np.where(roi>0)] = 255
    return roi.astype('uint8')

def mask_dilatation(mask,reverse = False,kernel=5):
    if kernel ==0 :
        return mask
    return (mask_to_gray(custom_dilatation(mask, kernel),reverse)/255).round().astype('uint8')

def mask_erodition(mask,reverse = False,kernel=5):
    if kernel ==0 :
        return mask
    return (mask_to_gray(custom_erodition(mask, kernel),reverse)/255).astype('uint8')

def mask_adjust(mask,reverse=False,size=1):
    istorch = False
    if torch.is_tensor(mask):
        mask = mask.numpy()
        istorch = True
    if size>0:
        res = mask_dilatation(mask,reverse = reverse,kernel=size+2)
    else:
        res = mask_erodition(mask,reverse = reverse,kernel=-size+2)
    if istorch:
        res = torch.FloatTensor(res)
    return res
'''
description: 图像边缘滤波
param {*} image
return {*}
'''
def edge_filter(image,mask,args):
    # ksize = args.edge_filter_ksize
    # distance = args.edge_filter_distance
    # iters = args.edge_filter_iters
    # revers = args.edge_filter_reverse
    ksize = 8
    distance = 40
    iters = 4
    revers = 0
    # h,w,c = image.shape
    #提取前景
    # if mask.max()>1:
    #     cimage[np.where(mask<10)] = 0
    # else:
    #     cimage[np.where(mask<(10/255))] = 0
    #获取mask边缘
    # edges = mask_to_edges(mask)
    # edges_arr = np.where(edges==255)
    #使用rdp扩充edges
    
    # mask = mask_adjust(mask,revers,kernel=-10) #先腐蚀
    
    threshold = 0.6
    test = False
    h,w,c = image.shape
    mask_gray = (mask_to_gray(mask,revers)/255).astype('uint8')
    for it in range(iters):
        adjust_base = -15+3*it
        mask_gray_ero= mask_adjust(mask_gray.copy(),revers,kernel=adjust_base) #先腐蚀
        mask_gray_dila = mask_adjust(mask_gray_ero.copy(),revers,kernel=adjust_base+10)#膨胀
        roi = custom_rdp(mask_gray_dila,distance) #修正区域
        edges_final = np.where(roi>0)
        # print(len(edges_final[0]))
        mask_b =  cv2.copyMakeBorder(mask_gray_ero, ksize, ksize, ksize, ksize, cv2.BORDER_REPLICATE)[...,None].repeat(c,axis=2)
        # roi_b =  cv2.copyMakeBorder(roi, ksize, ksize, ksize, ksize, cv2.BORDER_REPLICATE)
        cimage =  cv2.copyMakeBorder(image, ksize, ksize, ksize, ksize, cv2.BORDER_REPLICATE)
        for i in range(len(edges_final[0])):
            x,y = edges_final[0][i],edges_final[1][i]
            tmp = cimage[x:x+2*ksize+1,y:y+2*ksize+1] * mask_b[x:x+2*ksize+1,y:y+2*ksize+1] 
            valid_k = len(np.where(tmp[...,0]!=0)[0])
            # tmp = tmp.reshape(-1)
            value_x = tmp[...,0].sum() / valid_k if valid_k !=0 else 0
            value_y = tmp[...,1].sum() / valid_k if valid_k !=0 else 0
            # value = tmp[sorted(np.argsort(tmp)[-5:])].mean()
            # if tmp.max()-value>1:
            #     value = tmp.max
            # image[x,y,c] = 255
            # print(valid_k,value)
            # if abs(image[x,y,c]-value)>0.3:
            if test :
                image[x,y,c] = 255//(it+1)
            else:
                if not(value_x ==0 and value_y ==0) and np.sqrt(((image[x,y,0]-value_x)*w)**2+((image[x,y,1]-value_y)*h)**2) >= threshold:
                    image[x,y,0] = value_x  
                    image[x,y,1] = value_y 
    # #滤波操作
    # #滤波操作
    # # Creating the kernel(2d convolution matrix)
    # kernel = np.ones((ksize, ksize), np.float32)/ksize**2
    # # Applying the filter2D() function
    # res = cv2.filter2D(src=cimage, ddepth=-1, kernel=kernel)
    #获取最终图像
    # image[edges_final] = res[edges_final]
    return image

# def flow_to_image_torch(flow):
#     flow = torch.from_numpy(np.transpose(flow, [2, 0, 1]))
#     flow_im = flow_to_image(flow)
#     img = np.transpose(flow_im.numpy(), [1, 2, 0])
#     return img

def flow2rgb(flow, color_wheel=None, unknown_thr=1e6):
    """Convert flow map to RGB image.

    Args:
        flow (ndarray): Array of optical flow.
        color_wheel (ndarray or None): Color wheel used to map flow field to
            RGB colorspace. Default color wheel will be used if not specified.
        unknown_thr (str): Values above this threshold will be marked as
            unknown and thus ignored.

    Returns:
        ndarray: RGB image that can be visualized.
    """
    assert flow.ndim == 3 and flow.shape[-1] == 2
    if color_wheel is None:
        color_wheel = make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]
    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()

    ignore_inds = (
        np.isnan(dx) | np.isnan(dy) | (np.abs(dx) > unknown_thr) |
        (np.abs(dy) > unknown_thr))
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0

    rad = np.sqrt(dx**2 + dy**2)  # HxW
    if np.any(rad > np.finfo(float).eps):
        max_rad = np.max(rad)  # 使用最大模长来放缩坐标值
        dx /= max_rad
        dy /= max_rad

    rad = np.sqrt(dx**2 + dy**2)  # HxW
    angle = np.arctan2(-dy, -dx) / np.pi  # HxW（-1, 1]

    bin_real = (angle + 1) / 2 * (num_bins - 1)  # HxW (0, num_bins-1]
    bin_left = np.floor(bin_real).astype(int)  # HxW 0,1,...,num_bins-1
    bin_right = (bin_left + 1) % num_bins  # HxW 1,2,...,num_bins % num_bins -> 1, 2, ..., num_bins, 0
    w = (bin_real - bin_left.astype(np.float32))[..., None]  # HxWx1
    flow_img = (1 - w) * color_wheel[bin_left, :] + w * color_wheel[bin_right, :]  # 线性插值计算实际的颜色值
    small_ind = rad <= 1  # 以模长为1作为分界线来分开处理，个人理解这里主要是用来控制颜色的饱和度，而前面的处理更像是控制色调。
    # 小于1的部分拉大
    flow_img[small_ind] = 1 - rad[small_ind, None] * (1 - flow_img[small_ind])
    # 大于1的部分缩小
    flow_img[np.logical_not(small_ind)] *= 0.75

    flow_img[ignore_inds, :] = 0

    return flow_img

def show(image):
    if len(image.shape)<=2:
        return Image.fromarray(image.astype('uint8'))
    if image.shape[2]==2:
        return show(flow2rgb(image))
    if image.max()<=1:
        return Image.fromarray((image*255).astype('uint8'))
    return Image.fromarray(image.astype('uint8'))

def make_color_wheel(bins=None):
    if bins is None:
        bins = [15, 6, 4, 11, 13, 6]
    assert len(bins) == 6

    RY, YG, GC, CB, BM, MR = tuple(bins)
    # print(RY)
    ry = [1, np.arange(RY) / RY, 0]
    yg = [1 - np.arange(YG) / YG, 1, 0]
    gc = [0, 1, np.arange(GC) / GC]
    cb = [0, 1 - np.arange(CB) / CB, 1]
    bm = [np.arange(BM) / BM, 0, 1]
    mr = [1, 0, 1 - np.arange(MR) / MR]
    # print(ry)
    num_bins = RY + YG + GC + CB + BM + MR
    # print(num_bins)
    color_wheel = np.zeros((3, num_bins), dtype=np.float32)
    # print(color_wheel)

    col = 0
    for i, color in enumerate([ry, yg, gc, cb, bm, mr]):
        if i == 0:
            # print(i, color)
            pass
        for j in range(3):
            color_wheel[j, col:col + bins[i]] = color[j]
        col += bins[i]

    return color_wheel.T


'''
description: 叠加全部光流
param {*} imgs 光流+mask组合
return {*} [h,w,4] 4个纬度分别是：xy光流，mask，depth
'''
def multi_opt_overlap(flos):
    nums = len(flos)
    result = np.zeros(0)
    ranked_flo = [] #用于排序
    for i in range(nums):
        flo,mask_ = flos[i]
        mask = read(mask_,type='mask')
        if mask.shape != flo[...,0].shape:
            mask = cv2.resize(mask,(flo.shape[1],flo.shape[0]))
        # avg_depth[np.where(mask[...,0] == 0)] = 0 #通过depth平均数对先后关系进行排序
        ranked_flo.append([flo,mask])

    ranked_flo = sorted(ranked_flo,key=lambda x:-x[0][...,-1].mean()) #深度从大到小排序
    for flo,mask in ranked_flo: 
        if result.sum()==0:
            result = flo.copy()
            result[...,2]  = mask
        else:
            #叠加处理
            result[...,:2][np.where(mask!=0)] = flo[...,:2][np.where(mask!=0)]
            result[...,-1][np.where(mask!=0)] = flo[...,-1][np.where(mask!=0)]
            result[...,2][np.where(mask>result[...,2])]  = mask[np.where(mask>result[...,2])]
            #memo 0404 选择了强叠加，后续需要重新训练无过度网络+叠加逻辑
    return result #[h,w,4]



def algorithm_check(algo,all_algorithm_):
    all_algorithm = all_algorithm_.rstrip().split(',')
    for algorithm in all_algorithm:
        if algorithm.lower() in algo.lower():
            return
    raise ValueError('not supported algorithm:{}'.format(algo))
def algorithm_pick(images,args):
    # 2k use v0, 4k use v1, 2023/10/08 update
    algo = 'v0'
    if len(images.keys()) == 0:
        raise FileNotFoundError(f'[MM ERROR][input file]wrong root path,please check your config')
    image = images[list(images.keys())[0]][0][0] #first image
    try:
        h,w = read(image,type='image').shape[:2]
        if h > 1080 or w > 1920:
            algo = 'v1'
    except:
        sys.exit('[MM ERROR][main process]valid image type,should be tiff,png,exr,etc..') 

    if args.mask_mode:
        args.model = args.model.replace('kousei-mask','kousei-mask-{}'.format(algo))
    else:
        args.model = args.model.replace('kousei','kousei-{}'.format(algo))
    





'''
description: 使用mask对光流进行约束,去除mask之外的像素
param {*} flow 光流
param {*} mask_ mask文件
param {*} threshold 过滤阈值
param {*} reverse 转置mask
return {*} 约束后的光流
'''
def restrain(flow,mask,threshold,reverse=False,ldr=True,value=0):
    if mask is None:
        return flow
    if type(mask) == str:
        mask = read(mask,type='mask')
    if mask.shape != flow[...,0].shape:
        mask = cv2.resize(mask,flow[...,0].shape[::-1])
    if reverse:
        mask = 1 - mask
    if mask.sum() < mask.max() * 40 * 40:#过滤小mask以及0mask
        return flow
    if threshold==0:
        threshold = 1e-9
    # flow[np.where(mask<threshold)] = 0
    # 0915 ldr and hdr 0 to -1 , valid to -100 (0918 change)
    flow[np.where(mask<threshold)] = value
    return flow





def data_refine(data):
    if data.shape[2] == 2:
        data = np.insert(data,2,0,axis=2)
        data = np.insert(data,3,0,axis=2)
    elif data.shape[2]==3:
        data = np.insert(data,3,0,axis=2)
    return data

def appendzero(a,length=6):
    res = str(a)
    while(len(res)<length):
        res = '0'+ res
    return res

def getname(image):
        tmp = image.split('/')[-1]
        tmp = tmp[:-1-tmp[::-1].find('.')]
        tmp = tmp[-tmp[::-1].find('.'):]
        return tmp

def write_np_2_txt(path,datas,describes=None):
    with open(path, 'w') as f:
        for i in range(len(datas)):
            if describes is None:
                f.write(' '.join(map(str, datas[i])) + '\n')
            else:
                f.write(f'{describes[i]}:\t')
                f.write(' '.join(map(str, datas[i])) + '\n')

def write_txts(output,txts):
    for seq,txt in txts.items():
        sp = os.path.join(output,seq,'scene_change.txt')
        write_txt(sp,txt)

def write_txt(output,txt):
    with open(output,'w') as f:
        for line in txt:
	        f.write(str(line)+'\n')
                
def read_txt(path):
    with open(path,'r') as f:
        result = f.readlines()
    return result



def trilinear_interpolation(point, lut):
    size = lut.shape[-1]
    # 计算索引
    idx_x = int(point[...,0])
    idx_y = int(point[...,1])
    idx_z = int(point[...,2])
    
    # 计算权重
    w_x = point[...,0] - idx_x
    w_y = point[...,1] - idx_y
    w_z = point[...,2] - idx_z
    
    # 插值计算
    c000 = lut[idx_x, idx_y, idx_z]
    c001 = lut[idx_x, idx_y, idx_z + 1]
    c010 = lut[idx_x, idx_y + 1, idx_z]
    c011 = lut[idx_x, idx_y + 1, idx_z + 1]
    c100 = lut[idx_x + 1, idx_y, idx_z]
    c101 = lut[idx_x + 1, idx_y, idx_z + 1]
    c110 = lut[idx_x + 1, idx_y + 1, idx_z]
    c111 = lut[idx_x + 1, idx_y + 1, idx_z + 1]
    
    interpolated_color = (1 - w_x) * (1 - w_y) * (1 - w_z) * c000 + \
                         (1 - w_x) * (1 - w_y) * w_z * c001 + \
                         (1 - w_x) * w_y * (1 - w_z) * c010 + \
                         (1 - w_x) * w_y * w_z * c011 + \
                         w_x * (1 - w_y) * (1 - w_z) * c100 + \
                         w_x * (1 - w_y) * w_z * c101 + \
                         w_x * w_y * (1 - w_z) * c110 + \
                         w_x * w_y * w_z * c111
                         
    return interpolated_color



# def imwrite(save_path,image):
#     r,g,b,a,d = [image[...,i] for i in range(5)]
#     imwrite(save_path,r,g,b,a,d)

# def imwrite(save_path,r,g,b,a,d):
#     h,w = r.shape
#     if not a:
#         a=np.zeros((h,w))
#     if not d:
#         d=np.zeros((h,w))
#     hd = OpenEXR.Header(h,w)
#     hd['channels'] = {'B': FLOAT, 'G': FLOAT, 'R': FLOAT,'A': FLOAT,'D': FLOAT}
#     exr = OpenEXR.OutputFile(save_path,hd)
#     exr.writePixels({'R':r.tobytes(),'G':g.tobytes(),'B':b.tobytes(),'A':a.tobytes(),'D':d.tobytes()})
# # a,b = pre_treatment('/Users/qhong/Documents/data/test_data','player','video')



def get_resize_rate(algorithm,h,w):
    resize_rate_x = resize_rate_y = 1
    if '-v' in algorithm:
        if 'v0' in algorithm:
            resize_rate_x = 1920/w if w > 1920 else 1
            resize_rate_y = 1080/h if h > 1080 else 1
            # resize_rate_x = 1920/w 
            # resize_rate_y = 880/h 
        elif 'v1' in algorithm:
            resize_rate_x = 960/w if w > 960 else 1
            resize_rate_y = 432/h if h > 432 else 1
            # resize_rate_x = 1920/w 
            # resize_rate_y = 880/h 
        else:
            resize_rate_x = 1920/w if w > 1920 else 1
            resize_rate_y = 1080/h if h > 1080 else 1
    # if '-v' in algorithm:
    #     if 'kousei' in algorithm:
    #         resize_rate_x = 1560/w
    #         resize_rate_y = 920/h
    #         if 'small' in algorithm:
    #             resize_rate_x = 1400/w
    #             resize_rate_y = 800/h
    #         if 'v0' in algorithm:
    #             resize_rate_x = 3840/w
    #             resize_rate_y = 2160/h
    #         if 'mulframes' in algorithm:
    #             resize_rate_x = 1360/w
    #             resize_rate_y = 660/h
    #         if 'ori' in algorithm:
    #             resize_rate_x = 1
    #             resize_rate_y = 1
    #     if 'resize' in algorithm:
    #         if 'quad' in algorithm:
    #             resize_rate_x *= 0.25
    #             resize_rate_y *= 0.25
    #         elif 'oct' in algorithm:
    #             resize_rate_x *= 0.125
    #             resize_rate_y *= 0.125
    #         elif '_1k' in algorithm:
    #             resize_rate_x = 960/w
    #             resize_rate_y = 540/h
    #         elif '_2k' in  algorithm:
    #             resize_rate_x = 1920/w
    #             resize_rate_y = 1080/h
    #         elif '_4k' in  algorithm:
    #             resize_rate_x = 3840/w
    #             resize_rate_y = 2160/h
    #         elif '_rate_' in  algorithm:
    #             resize_rate_x *= float( algorithm[ algorithm.find('_rate_')+6:].split('_')[0])
    #             resize_rate_y *= float( algorithm[ algorithm.find('_rate_')+6:].split('_')[1])
    #         else:
    #             resize_rate_x *= 0.5
    #             resize_rate_y *= 0.5
    # else: # auto mode
    #     if w > 3840:
    #         resize_rate_x = 3840/w
    #     if h > 2160:
    #         resize_rate_y = 2160/h
    return resize_rate_x,resize_rate_y






# def prepare_images(images,masks,task,start,end,args):
#     res = []
#     for i in range(start,end):
#         res.append(get_pairs(i,task,images,masks,args))
#     return res


def add_grain(image, grain_amount):
    # 为图像添加"film grain"效果，噪声的强度由grain_amount控制
    # image: 原始图像
    # grain_amount: 噪声的强度
    
    # 将图像转换为float32以保持精度
    image = np.array(image, dtype=np.float32)
    
    # 生成与图像同样大小的噪声矩阵
    noise = np.random.randn(*image.shape) * grain_amount
    
    # 将噪声添加到图像中
    noisy_image = image + noise
    
    # 确保结果仍在合理的像素值范围内
    noisy_image = np.clip(noisy_image, 0, 255)
    
    # 将结果图像转换回原始的数据类型
    noisy_image = np.array(noisy_image, dtype=np.uint8)
    
    return noisy_image

def input_valid_check(images,args):
    if len(images)< min(2,args.num_frames):
        raise FileNotFoundError(f'[MM ERROR][input file]wrong root path and image size should > {args.num_frames}')
    if min(args.frame_step)>1 and len(images)< min(2,min(args.frame_step))*2:
        raise FileNotFoundError(f'[MM ERROR][input file]wrong root path and image size should > {args.num_frames}*2')





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



def motion_blur(img,size=15):
    mask = None
    if img.shape[2] == 4:
        mask = img[...,-1]
        img =  img[...,:3]
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    res = cv2.filter2D(img, -1, kernel_motion_blur)
    if mask is not None:
        res = np.concatenate((res,mask[...,None]),axis=2)
    return res





def custom_filter(image,kernel_size=3):
    output_image = np.zeros_like(image)
    kernel_radius = kernel_size // 2
    # 只对有值的区域进行计算
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 计算邻域的边界
            i_min, i_max = max(0, i-kernel_radius), min(image.shape[0], i+kernel_radius+1)
            j_min, j_max = max(0, j-kernel_radius), min(image.shape[1], j+kernel_radius+1)
            
            # 提取邻域
            neighborhood = image[i_min:i_max, j_min:j_max]
            if neighborhood[neighborhood != 0].size ==0:
                continue
            
            # 计算邻域内的有值像素的平均值
            valid_values_x = neighborhood[...,0][neighborhood[...,0] != 0] # 假设“有值”的定义为非零
            valid_values_y = neighborhood[...,1][neighborhood[...,1] != 0] # 假设“有值”的定义为非零
            if valid_values_x.size > 0:
                # mean_value = valid_values.mean()
                mean_value_x = max(valid_values_x, key=abs)
            else:
                mean_value_x = 0 # 如果邻域内没有有值的像素，则设置为0或其他适当的值
            if valid_values_y.size > 0:
                # mean_value = valid_values.mean()
                mean_value_y = max(valid_values_y, key=abs)
            else:
                mean_value_y = 0 # 如果邻域内没有有值的像素，则设置为0或其他适当的值
            
            # 更新输出图像
            output_image[i, j] = [mean_value_x,mean_value_y]
    return output_image

def mv_magnitude(flow):
    # flow = read('/Users/qhong/Desktop/0221/0222_mask_enhance_test/064000/mv0/mv_0.flo')
    flow_cp = flow.copy()
    tflow = np.zeros_like(flow)
    mask = np.ones((flow.shape[0],flow.shape[1],3))*255
    gradient_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    gradient_magnitude_x,gradient_magnitude_y = flow[...,0],flow[...,1]

    tt_x,tt_y = np.gradient(gradient_magnitude_x, axis=(0, 1))
    yy_x,yy_y = np.gradient(gradient_magnitude_y, axis=(0, 1))
    uu_x,uu_y = np.gradient(gradient_magnitude, axis=(0, 1))
    threshold = 1
    transition_region = (tt_x > threshold) | (tt_y > threshold) | (yy_x > threshold) | (yy_y > threshold) | (uu_x>threshold) | (uu_y>threshold) & ((np.abs(gradient_magnitude_x) < 10) | (np.abs(gradient_magnitude_y) < 10))
    mask[transition_region] = 0
    # tflow= flow_cp
    tflow[transition_region] = flow_cp[transition_region]
    # 应用自定义滤波器
    tflow =custom_filter(tflow)
    flow[transition_region] = tflow[transition_region]
    # (flow==filtered_flow).all()
    # show(mask)
    # write('/Users/qhong/Desktop/0221/0222_mask_enhance_test/test.flo',flow)
    # write('/Users/qhong/Desktop/0221/0222_mask_enhance_test/tflow.flo',tflow)
    return flow
