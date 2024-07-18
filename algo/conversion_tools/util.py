'''
Author: Qing Hong
Date: 2022-03-08 17:21:31
LastEditors: QingHong
LastEditTime: 2024-04-22 17:18:38
Description: file content
'''
import os
import time
import shutil
import numpy as np
from scipy.optimize import leastsq
from skimage.metrics import peak_signal_noise_ratio
import imageio,cv2
import argparse,configparser
from tqdm import tqdm
# from torchvision.utils import flow_to_image
# import torch
cur_time = str(time.gmtime().tm_mon) + '_' + str(time.gmtime().tm_mday)
cur_time_sec = cur_time+'/'+str(time.gmtime().tm_hour)+'/'+str(time.gmtime().tm_sec)
import torch.nn.functional as F
class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
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

import matplotlib.pyplot as plt
def pt(image):
    plt.imshow(image[...,::-1])
def pt2(image):
    plt.imshow(np.insert(image,1,0,axis=2)[...,::-1])
def plot(image):
    plt.imshow(image)
from PIL import Image

def show(image):
    im = Image.fromarray(image.astype('uint8')).convert('RGB')
    im.show()

def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]




def imresize(image, ratio=None, out_size=None, method='bicubic', start='auto,auto', out_offset=None, padding='symmetric', clip=True):
    '''
    Parameters
    ----------
    image     : ndarray, 1 channel or n channels interleaved
    ratio     : scale ratio. It can be a scalar, or a list/tuple/numpy.array.
                If it's a scalar, the ratio applies to both H and V.
                If it's a list/numpy.array, it specifies the hor_ratio and ver_ratio.
    out_size  : output size [wo, ho]
    method    : 'bicubic' | 'bilinear' | 'nearest'
    start     : string seperated by ',' specify the start position of x and y
    out_offset: offset at output domain [xoffset, yoffset]
    padding   : 'zeros' | 'edge','replicate','border' | 'symmetric'. Default: 'symmetric' (TBD)
    clip      : only effect for float image data (uint8/uint16 image output is alway clipped)

    Returns
    -------
    result: ndarray

    History
    2021/07/10: changed ratio order [H,W] -> [W,H]
                add out_offset
    2021/07/11: add out_size
    2021/07/31: ratio cannot be used as resolution any more

    Notes：
    如果 ratio 和 out_size 都没有指定，则 ratio = 1
    如果只指定 out_size，则 ratio 按输入图像尺寸和 out_size 计算
    如果只指定 ratio，则输出尺寸为输入图像尺寸和 ratio 的乘积并四舍五入
    如果同时指定 ratio 和 out_size，则按  ratio 输出 out_size 大小的图，这时既保证 ratio，也保证输出图像尺寸
    '''
    startx, starty = start.split(',')
    ih, iw = image.shape[:2]

    if ratio is None:
        ratio = 1 if out_size is None else [out_size[0]/iw, out_size[1]/ih]

    if isinstance(ratio, list) or isinstance(ratio, np.ndarray) or isinstance(ratio, tuple):
        hratio, vratio = ratio[0], ratio[1]
    else:
        hratio, vratio = ratio, ratio

    if out_offset is None: out_offset = (0, 0)
    if out_size   is None: out_size   = (None, None)

    if method == 'bicubic':
        outv = ver_interp_bicubic(image, vratio, out_size[1], starty, out_offset[1], clip)
        out  = hor_interp_bicubic(outv, hratio, out_size[0], startx, out_offset[0], clip)
    else:
        xinc, yinc = 1/hratio, 1/vratio
        ow = round(iw * hratio) if out_size[0] is None else out_size[0]
        oh = round(ih * vratio) if out_size[1] is None else out_size[1]
        x0 = (-.5 + xinc/2 if startx == 'auto' else float(startx)) + out_offset[0] * xinc # (x0, y0) is in input domain
        y0 = (-.5 + yinc/2 if starty == 'auto' else float(starty)) + out_offset[1] * yinc 
        x = x0 + np.arange(ow) * xinc
        y = y0 + np.arange(oh) * yinc
        xaux = np.r_[np.arange(iw), np.arange(iw-1,-1,-1)] # 0, 1, ..., iw-2, iw-1, iw-1, iw-2, ..., 1, 0
        yaux = np.r_[np.arange(ih), np.arange(ih-1,-1,-1)]
        if method == 'nearest':
            x = np.floor(x + .5).astype('int32') # don't use np.round() as it rounds to even value (w,)
            y = np.floor(y + .5).astype('int32')
            xind = xaux[np.mod(np.int32(x), xaux.size)]
            yind = yaux[np.mod(np.int32(y), yaux.size)]
            out = image[np.ix_(yind, xind)]
        elif method == 'bilinear':
            tlx = np.floor(x).astype('int32')
            tly = np.floor(y).astype('int32')
            wy, wx = np.ix_(y - tly, x - tlx) # wy: (h, 1), wx: (1, w)
            brx = xaux[np.mod(tlx + 1, xaux.size)]
            bry = yaux[np.mod(tly + 1, yaux.size)]
            tlx = xaux[np.mod(tlx    , xaux.size)]
            tly = yaux[np.mod(tly    , yaux.size)]
            if image.ndim == 3:
                wy, wx = wy[..., np.newaxis], wx[..., np.newaxis]
            out = (image[np.ix_(tly, tlx)] * (1-wx) * (1-wy) + image[np.ix_(tly, brx)] * wx * (1-wy)
                 + image[np.ix_(bry, tlx)] * (1-wx) *    wy  + image[np.ix_(bry, brx)] * wx *    wy)
        else:
            print('Error: Bad -method argument {}. Must be one of \'bilinear\', \'bicubic\', and \'nearest\''.format(method))
        if   image.dtype == 'uint8' : out = np.uint8(out + 0.5)
        elif image.dtype == 'uint16': out = np.uint16(out + 0.5)
    return out

def adjust_weight(args):
    if '.pth' in args.gma_weight_file:
        args.gma_weight_file = args.gma_weight_file.replace('.pth','')
    args.algorithm_fullname = args.gma_weight_file
    args.algorithm = args.gma_weight_file.rstrip().split('_')[0].split(' ')[0]
    if '_resize' in args.gma_weight_file:
        gw = args.gma_weight_file
        args.algorithm += gw[gw.find('_resize'):]
        args.gma_weight_file = gw[:gw.find('_resize')]
    args.gma_weight_file += '.pth'
    return args
    
def init_param(path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help="Enable Eval mode.")
    parser.add_argument('--test', action='store_true', help="Enable Test mode.")
    parser.add_argument('--viz', action='store_true', help="Enable Viz mode.")
    parser.add_argument('--fixed_point_reuse', action='store_true', help="Enable fixed point reuse.")
    parser.add_argument('--warm_start', action='store_true', help="Enable warm start.")

    parser.add_argument('--name', default='deq-flow', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 

    parser.add_argument('--total_run', type=int, default=1, help="total number of runs")
    parser.add_argument('--start_run', type=int, default=1, help="begin from the given number of runs")
    parser.add_argument('--restore_name', help="restore experiment name")
    parser.add_argument('--resume_iter', type=int, default=-1, help="resume from the given iterations")

    parser.add_argument('--tiny', action='store_true', help='use a tiny model for ablation study')
    parser.add_argument('--large', action='store_true', help='use a large model')
    parser.add_argument('--huge', action='store_true', help='use a huge model')
    parser.add_argument('--gigantic', action='store_true', help='use a gigantic model')
    parser.add_argument('--old_version', action='store_true', help='use the old design for flow head')

    parser.add_argument('--restore_ckpt', help="restore checkpoint for val/test/viz")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--test_set', type=str, nargs='+')
    parser.add_argument('--viz_set', type=str, nargs='+')
    parser.add_argument('--viz_split', type=str, nargs='+', default=['test'])
    parser.add_argument('--output_path', help="output path for evaluation")

    parser.add_argument('--eval_interval', type=int, default=5000, help="evaluation interval")
    parser.add_argument('--save_interval', type=int, default=5000, help="saving interval")
    parser.add_argument('--time_interval', type=int, default=500, help="timing interval")

    parser.add_argument('--gma', action='store_true', help='use gma')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--schedule', type=str, default="onecycle", help="learning rate schedule")

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--vdropout', type=float, default=0.0, help="variational dropout added to BasicMotionEncoder for DEQs")
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--active_bn', action='store_true')
    parser.add_argument('--all_grad', action='store_true', help="Remove the gradient mask within DEQ func.")
    parser.add_argument('--wnorm', action='store_true', help="use weight normalization")
    parser.add_argument('--f_solver', default='anderson', type=str, choices=['anderson', 'broyden', 'naive_solver'],
                        help='forward solver to use (only anderson and broyden supported now)')
    parser.add_argument('--b_solver', default='broyden', type=str, choices=['anderson', 'broyden', 'naive_solver'],
                        help='backward solver to use')
    parser.add_argument('--f_thres', type=int, default=40, help='forward pass solver threshold')
    parser.add_argument('--b_thres', type=int, default=40, help='backward pass solver threshold')
    parser.add_argument('--f_eps', type=float, default=1e-3, help='forward pass solver stopping criterion')
    parser.add_argument('--b_eps', type=float, default=1e-3, help='backward pass solver stopping criterion')
    parser.add_argument('--f_stop_mode', type=str, default="abs", help="forward pass fixed-point convergence stop mode")
    parser.add_argument('--b_stop_mode', type=str, default="abs", help="backward pass fixed-point convergence stop mode")
    parser.add_argument('--eval_factor', type=float, default=1.5, help="factor to scale up the f_thres at test for better convergence.")
    parser.add_argument('--eval_f_thres', type=int, default=0, help="directly set the f_thres at test.")

    parser.add_argument('--indexing_core', action='store_true', help="use the indexing core implementation.")
    parser.add_argument('--ift', action='store_true', help="use implicit differentiation.")
    parser.add_argument('--safe_ift', action='store_true', help="use a safer function for IFT to avoid potential segment fault in older pytorch versions.")
    parser.add_argument('--n_losses', type=int, default=1, help="number of loss terms (uniform spaced, 1 + fixed point correction).")
    parser.add_argument('--indexing', type=int, nargs='+', default=[], help="indexing for fixed point correction.")
    parser.add_argument('--phantom_grad', type=int, nargs='+', default=[1], help="steps of Phantom Grad")
    parser.add_argument('--tau', type=float, default=1.0, help="damping factor for unrolled Phantom Grad")
    parser.add_argument('--sup_all', action='store_true', help="supervise all the trajectories by Phantom Grad.")

    parser.add_argument('--sradius_mode', action='store_true', help="monitor the spectral radius during validation")






    parser.add_argument('--model', help="restore checkpoint",default='checkpoints/gma-sintel.pth')
    parser.add_argument('--model_name', help="define model name", default="GMA")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args([])
    config = configparser.ConfigParser()
    config.read(path, encoding="utf-8")

    args.image_file = config.get('opticalflow','image_file')
    args.front_mask_file = config.get('opticalflow','front_mask_file')
    # args.back_mask_file = config.get('opticalflow','back_mask_file')
    args.gma_weight_file = config.get('opticalflow','algorithm')
    # adjust to auto mode
    args = adjust_weight(args)
    args.mt_backend = config.get('opticalflow','mt_backend')
    args.mv_ref = config.getboolean('opticalflow','mv_ref')
    args.evaluate_epe = config.getboolean('opticalflow','evaluate_epe')
    args.ground_truth_file = config.get('opticalflow','ground_truth_file')
    args.refine = config.getboolean('opticalflow','refine')
    args.root = config.get('opticalflow','root')
    args.output = config.get('opticalflow','output')
    args.cal_depth = config.getboolean('opticalflow','cal_depth')
    args.threeD_mode = config.getboolean('opticalflow','3D_mode')
    args.cal_virtual_depth = True if (not args.threeD_mode and args.cal_depth) else False
    args.cal_disparity = True if (args.threeD_mode and args.cal_depth) else False
    # args.right_eye_file = config.get('opticalflow','right_eye_file')
    # args.right_eye_back_mask_file = config.get('opticalflow','right_eye_back_mask_file')
    #extra mode
    args.enable_extra_input_mode = config.getboolean('opticalflow','enable_extra_input_mode')
    args.left_root = config.get('opticalflow','left_root')
    args.right_root = config.get('opticalflow','right_root')
    #exr
    args.savetype = config.get('opticalflow','savetype')
    args.depth_range = config.getint('opticalflow','depth_range')
    args.mask_mode= config.getboolean('opticalflow','mask_mode')
    # args.bg= config.getboolean('opticalflow','bg')
    args.restrain = config.getboolean('opticalflow','restrain')

    args.enable_dump = config.getboolean('opticalflow','enable_dump')
    args.load_dump = config.getboolean('opticalflow','load_dump')
    # args.fusion_mode = config.get('opticalflow','fusion_mode')   
    args.matting = config.getboolean('opticalflow','matting')
    # args.rm_ori_res = config.getboolean('opticalflow','rm_ori_res')
    args.n_limit = config.getint('opticalflow','n_limit')
    args.n_start = config.getint('opticalflow','n_start')
    args.gpu = config.get('opticalflow','gpu')
    args.restrain_all = config.getboolean('opticalflow','restrain_all')
    args.time_cost = config.getboolean('opticalflow','time_cost')
    args.dump_masked_file = config.getboolean('opticalflow','dump_masked_file')
    args.cf_use_full = config.getboolean('opticalflow','cf_use_full')
    args.use_bounding_box = config.getboolean('opticalflow','use_bounding_box')
    # args.bounding_with_no_restrain = config.getboolean('opticalflow','bounding_with_no_restrain')
    args.dump_bounding_image = config.getboolean('opticalflow','dump_bounding_image')
    args.enable_single_input_mode = config.getboolean('opticalflow','enable_single_input_mode')
    args.pass_mv = config.getboolean('opticalflow','pass_mv')
    args.pass_depth = config.getboolean('opticalflow','pass_depth')
    args.dump_restored_image = config.getboolean('opticalflow','dump_restored_image')
    args.dump_low_mv = config.getboolean('opticalflow','dump_low_mv')
    args.pass_when_exist = config.getboolean('opticalflow','pass_when_exist')
    args.depth_to_disparity = config.getboolean('opticalflow','depth_to_disparity')
    args.time_cost = False if args.pass_mv else args.time_cost
    args.depth_name = config.get('opticalflow','depth_name')  
    args.film_border = config.getboolean('opticalflow','film_border')
    args.edge_filter = config.getboolean('opticalflow','edge_filter')
    args.mv0 = config.getboolean('opticalflow','mv0')
    args.mv1 = config.getboolean('opticalflow','mv1')

    args.cal_depth = config.getboolean('opticalflow','cal_depth')
    args.dilatation = config.getint('opticalflow','dilatation')
    args.erodition = config.getint('opticalflow','erodition')
    args.clean_tmp_file = config.getboolean('opticalflow','clean_tmp_file')
    args.cal_best_mv = config.getboolean('opticalflow','cal_best_mv')
    args.cal_best_mask_left = config.get('opticalflow','cal_best_mask_left')
    args.cal_best_mask_right = config.get('opticalflow','cal_best_mask_right')
    args.cal_best_target_algorithm = config.get('opticalflow','cal_best_target_algorithm')

    #refine threshold
    args.threshold = config.getint('opticalflow','threshold')
    args.IMAGEIO_USERDIR = config.get('opticalflow','IMAGEIO_USERDIR') 
    args.TORCH_HUB = config.get('opticalflow','TORCH_HUB') 

    #multi masks
    args.multi_mask_mode = config.getboolean('opticalflow','multi_mask_mode')
    args.multi_mask_file = config.get('opticalflow','multi_mask_file') if args.multi_mask_mode else None
    # args.auto_generate_depth = config.getboolean('opticalflow','auto_generate_depth') 

    #last
    args.empty_cache = config.getboolean('opticalflow','empty_cache')
    args.bounding_box_mode = config.getint('opticalflow','bounding_box_mode')
    
    args.all_algorithm = config.get('opticalflow','all_algorithm')
    
    
    #output merge
    args.MM9_format = config.getboolean('opticalflow','MM9_format')
    args.clean_source = config.getboolean('opticalflow','clean_source')
    args.scene_change = config.getboolean('opticalflow','scene_change')

    args.fix_inexistent_mv = True if args.MM9_format else config.getboolean('opticalflow','fix_inexistent_mv')

    args.compress_method = 'none' if args.fix_inexistent_mv else config.get('opticalflow','compress_method')
    
    return args
    
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


def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w, h = np.fromfile(f, np.int32, count=2)
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d 


def write_flo_file(flow, filename): # flow: H x W x 2
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    if flow.ndim == 4: # has batch
        flow = flow[0]

    outpath = os.path.dirname(filename)
    if outpath != '' and not os.path.isdir(outpath):
        os.makedirs(outpath)

    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    height, width = flow.shape[:2]
    magic.tofile(f)
    np.int32(width).tofile(f)
    np.int32(height).tofile(f)
    data = np.float32(flow).flatten()
    data.tofile(f)
    f.close()


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

def read(path,algorithm='',type='flo'):
    #mask 取最后一个通道    
    res = None
    if type == 'flo':
        if '.exr' in path:
            res =  imageio.imread(path).astype('float32')
        elif '.flo' in path:
            res =  read_flo_file(path).astype('float32')
    if type == 'mask':
        if '.exr' in path:
            tmp = imageio.imread(path)
            if tmp.max()>=100:
                res = tmp.astype('float32')
            else:
                tmp = np.clip(imageio.imread(path),0,1)
                res = (tmp*255).astype('float32')
        else:
            res = cv2.imread(path)[...,::-1].astype('float32')
        res = res[...,0]
    if type == 'image':
        if '.exr' in path:
            tmp = np.clip(imageio.imread(path),0,1)
            res = (tmp*255)[...,:3].astype('uint8')
        else:
            res =  cv2.imread(path).astype('uint8')[...,::-1]
        res = res[...,:3]
    return np.ascontiguousarray(res)
    
def write(path,flow):
    if '.exr' in path:
        if flow.shape[2] == 2:
            flow = np.insert(flow,2,0,axis=2)
        if flow.shape[2] == 3:
            flow = np.insert(flow,2,0,axis=2)
        imageio.imwrite(path,flow[...,:4],flags=imageio.plugins.freeimage.IO_FLAGS.EXR_ZIP|imageio.plugins.freeimage.IO_FLAGS.EXR_FLOAT)
    elif '.flo' in path:
        write_flo_file(flow[...,:2],path)
    else:
        if len(flow.shape) == 2:
            flow = np.repeat(flow[...,None],3,axis=2)
        if flow.shape[2] == 2:
            flow = np.insert(flow,2,0,axis=2)
        cv2.imwrite(path,flow[...,:3][...,::-1])
    
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
    #修正special
    if 'special' in args.gma_weight_file:
        image *= 2
    #膨胀腐蚀
    if args.dilatation>0:
        image = custom_dilatation(image,args.dilatation)
    if args.erodition>0:
        image = custom_erodition(image,args.erodition)
    #边缘滤波
    if args.edge_filter and mask is not None:
        image = edge_filter(image,mask,args)
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

def mask_adjust(mask,reverse=False,kernel=5):
    if kernel>0:
        return mask_dilatation(mask,reverse = reverse,kernel=kernel)
    return mask_erodition(mask,reverse = reverse,kernel=-kernel)
'''
description: 图像边缘滤波
param {*} image
return {*}
'''
def edge_filter(image,mask,args):
    ksize = args.edge_filter_ksize
    distance = args.edge_filter_distance
    iters = args.edge_filter_iters
    revers = args.edge_filter_reverse
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

def ctest():
    a = cv2.imread('/home/rg0775/QingHong/dataset/gma_datasets/mhyang/Titanic/clip03/clip03_left/mask/clip03_l_00000001.png')
    b = a.copy()
    b =  mask_erodition(b,True,kernel=110)
    Image.fromarray(b*255)
    roi = custom_rdp(b,200,0.2,0.8)
    Image.fromarray(roi)
    edges_final = np.where(roi>0)
    c = np.zeros_like(a)[...,:3]
    c[edges_final] = 255
    Image.fromarray(c)



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

def myplt(image):
    if len(image.shape)<=2:
        return Image.fromarray(image.astype('uint8'))
    if image.shape[2]==2:
        return myplt(flow2rgb(image))
    if image.max()<=1:
        return Image.fromarray((image*255).astype('uint8'))
    return Image.fromarray(image[...,::-1].astype('uint8'))

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
    

    #front masked area set to 0.5 and back to 1 

def save_file(save_path,flow,refine=False,savetype='exr',zero_to_one=True,compress_method='none'):
        flow = flow.astype("float32")
        #append z axis
        if flow.shape[2] == 2:
            flow = np.insert(flow,2,0,axis=2)
        #refine
        if refine:
            flow = custom_refine(flow,zero_to_one=zero_to_one)
        #append
        if flow.shape[2] == 3 and savetype=='exr':
            flow = np.insert(flow,2,0,axis=2)
        # if savetype == 'exr':           
        if compress_method.lower() == 'none':
            imageio.imwrite(save_path,flow.astype("float32"),flags=imageio.plugins.freeimage.IO_FLAGS.EXR_NONE)
        else:
            imageio.imwrite(save_path,flow.astype("float32"))

def save_depth_file(save_path,flow,zero_to_one = True,half=False,depth_range=128,reverse=False):
    # if zero_to_one:
    #     flow = -flow
    # flow = np.abs(flow)
    # flow = (flow + dp_value) / dp_value*2
    
    if not zero_to_one:
        flow *= -1
    if half:
        flow *=0.5
    flow = (flow + depth_range/2).clip(0, depth_range) / depth_range
    h,w,_ = flow.shape
    res = np.ones((h,w,4)).astype("float32")
    if reverse:
        flow *= -1
    res[...,3] = flow[...,0]
    res[...,2] = flow[...,1]
    imageio.imwrite(save_path,res.astype("float32"),flags=imageio.plugins.freeimage.IO_FLAGS.EXR_NONE)


def save_depth_file_flo(save_path,flow,zero_to_one = True,half=False,depth_range=128,reverse=False):
    if not zero_to_one:
        flow *= -1
    if half:
        flow *=0.5
    if reverse:
        flow *= -1
    write_flo_file(flow.astype("float32"),save_path.replace('.exr','.flo'))

def save_depth_file_single(save_path,flow,zero_to_one = True,half=False,depth_range=128,reverse=False):
    if not zero_to_one:
        flow *= -1
    if half:
        flow *=0.5
    if reverse:
        flow *= -1
    np.savetxt(save_path.replace('.exr','.log'), flow[...,1],fmt='%f',delimiter=' ')



'''
description: 使用mask对光流进行约束,去除mask之外的像素
param {*} flow 光流
param {*} mask_ mask文件
param {*} threshold 过滤阈值
param {*} mv_ref 颠倒过滤
param {*} reverse 转置mask
return {*} 约束后的光流
'''
def restrain(flow,mask,threshold,mv_ref,reverse=False):
    if mask is None:
        return flow
    if type(mask) == str:
        mask = read(mask,type='mask')
    if mask.shape != flow[...,0].shape:
        mask = cv2.resize(mask,flow[...,0].shape[::-1])
    if mask.max()<=1 and mask.min()>=-1:
        threshold = 0.5
    if reverse:
        mask = threshold - mask
    if mask.sum() < mask.max() * 40 * 40:#过滤小mask以及0mask
        return flow
    if mv_ref:
        flow[np.where(mask==0)] = 0
    else:
        flow[np.where(mask<=threshold//2)] = 0
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



def write_txts(txts,output):
    for seq,txt in txts.items():
        sp = os.path.join(output,seq,'scene_change.txt')
        write_txt(txt,sp)

def write_txt(txt,output):
    with open(output,'w') as f:
        for line in txt:
	        f.write(str(line)+'\n')
def read_txt(path):
    with open(path,'r') as f:
        result = f.readlines()
    return result
def combine_result(args,masks=None):
    import OpenEXR,Imath
    output = args.output
    scenes = jhelp(output)
    save_root = os.path.join(output,'MM')
    mkdir(save_root)
    for scene in scenes:
        if '/MM' in scene[-4:] or not os.path.isdir(scene):
            continue
        contents = jhelp(scene)
        folder = set()
        for t in contents:
            if os.path.isdir(t):
                folder.add(t.replace('mv0','{}').replace('mv1','{}'))
        assert len(folder)>0,'no result to combine!'
        while(len(folder)!=0):
            f = folder.pop()
            mv0_path = f.format('mv0')
            mv1_path = f.format('mv1')
            tmp = os.path.join(os.path.abspath(os.path.join(mv0_path,'..')),'scene_change.txt')
            scene_change_file = None if not os.path.isfile(tmp) else tmp
            scene_change = []
            if scene_change_file is not None:
                scene_change = [int(a.rstrip()) for a in read_txt(scene_change_file)]
            mv0s = jhelp(mv0_path)
            mv1s = jhelp(mv1_path)
            total_range = range(len(mv0s)) if not args.use_tqdm else tqdm(range(len(mv0s)), desc='{}:{}'.format('MM9 format',os.path.basename(scene)))
            for i in total_range:
                mv0 = read(mv0s[i],type='flo')
                mv1 = read(mv1s[i],type='flo')
                mv0_R,mv0_G,mv0_B,mv0_A = mv0[...,0],mv0[...,1],mv0[...,2],mv0[...,3]
                mv1_R,mv1_G,mv1_B,mv1_A = mv1[...,0],mv1[...,1],mv1[...,2],mv1[...,3]
                sp = mv0s[i].replace('/'+os.path.basename(scene)+'/','/MM/'+os.path.basename(scene)+'/')
                sp = sp.replace('_mv0/','/')
                mkdir(os.path.abspath(os.path.join(sp,'..')))
                sz = mv0.shape[:2]
                header = OpenEXR.Header(sz[1], sz[0])
                header['channels'] = {
                    'MV0.x': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                    'MV0.y': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                    'MV0.z': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                    'MV1.x': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                    'MV1.y': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                    'MV1.z': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                    'Depth': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    'Matte': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                }
                header['SC_Flag'] = False if i not in scene_change else True
                header['BA_Step'] = True
                out = OpenEXR.OutputFile(sp, header)
                Matte = np.zeros(sz,dtype=np.float16).tobytes()
                if masks is not None:
                    mask = read(masks[os.path.basename(scene)][i],type='mask')/255
                    if mask.sum() != 0:
                        Matte = mask.astype(np.float16).tobytes()
                Depth_value = np.zeros(sz,dtype=np.float32) if mv0_A.sum() == 0 else mv0_A.tobytes()
                mv0r_value = np.zeros(sz,dtype=np.float16) if mv0_R.sum() == 0 else mv0_R.astype(np.float16).tobytes()
                mv0g_value = np.zeros(sz,dtype=np.float16) if mv0_G.sum() == 0 else mv0_G.astype(np.float16).tobytes()
                mv0b_value = np.zeros(sz,dtype=np.float16) if mv0_B.sum() == 0 else mv0_B.astype(np.float16).tobytes()
                mv1r_value = np.zeros(sz,dtype=np.float16) if mv1_R.sum() == 0 else mv1_R.astype(np.float16).tobytes()
                mv1g_value = np.zeros(sz,dtype=np.float16) if mv1_G.sum() == 0 else mv1_G.astype(np.float16).tobytes()
                mv1b_value = np.zeros(sz,dtype=np.float16) if mv1_B.sum() == 0 else mv1_B.astype(np.float16).tobytes()
                out.writePixels({'MV0.x' : mv0r_value, 'MV0.y' : mv0g_value, 'MV0.z' : mv0b_value ,'MV1.x' : mv1r_value, 'MV1.y' : mv1g_value, 'MV1.z' : mv1b_value ,'Depth':Depth_value,'Matte':Matte})
            if scene_change_file is not None:
                shutil.copy(scene_change_file,scene_change_file.replace('/'+os.path.basename(scene)+'/','/MM/'+os.path.basename(scene)+'/'))
        if args.clean_source:
            shutil.rmtree(scene)


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