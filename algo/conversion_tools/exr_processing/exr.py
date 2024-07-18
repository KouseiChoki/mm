import numpy as np
import OpenEXR, Imath, array
import os,sys
import cv2
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
pt = Imath.PixelType(Imath.PixelType.FLOAT)
'''
description: 读取exr文件
param {*} filePath 文件地址
return {*} 返回rgb图像，mv1，mask
'''
def get_channel_data(img_exr,keyword,type='f'):
    data = np.array(array.array(type, img_exr.channels(keyword,pt)))
    return data


def exr_imread(filePath):
    img_exr = OpenEXR.InputFile(filePath)
    dw = img_exr.header()['dataWindow']
    size = (dw.max.y - dw.min.y + 1,dw.max.x - dw.min.x + 1)

    #获取mv
    mv1_x = np.array(array.array('f', img_exr.channel('FinalImageMovieRenderQueue_MotionVectors.R', pt)))
    mv1_y = np.array(array.array('f', img_exr.channel('FinalImageMovieRenderQueue_MotionVectors.G', pt)))
    ##解除归一化
    mv1_x = (mv1_x - 0.5) * 2 * size[1] * -1
    mv1_y = (mv1_y - 0.5) * 2 * size[0]

    mv1_x = mv1_x.reshape(size)
    mv1_y = mv1_y.reshape(size)
    mv1 = np.stack([mv1_x,mv1_y],axis=2)
    # 获取rgb
    r_str, g_str, b_str = img_exr.channels('RGB',pt)
    red = np.array(array.array('f', r_str))
    green = np.array(array.array('f', g_str))
    blue = np.array(array.array('f', b_str))
    red = red.reshape(size)
    green = green.reshape(size)
    blue = blue.reshape(size)
    image = np.stack([red,green,blue],axis=2).clip(0,1)*255
    # 获取mask
    mask = np.array(array.array('f', img_exr.channel('FinalImageMovieRenderQueue_MotionVectors.B', pt)))
    mask = mask.reshape(size)

    return image.astype('uint8'),mv1.astype('float32'),mask.astype('uint8')

def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]



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

def subpro(args):
    source,half,resize_rate = args
    name = os.path.basename(source).replace('.exr','')
    imgpth = os.path.join(image_save_path,name)+'.png'
    maskpth = os.path.join(mask_save_path,name)+'.png'
    mimgpth = os.path.join(maskimg_save_path,name)+'.png'
    mv1pth = os.path.join(mv1_save_path,name)+'.flo'
    mmv1pth = os.path.join(mv1_masked_save_path,name)+'.flo'
    if os.path.isfile(mmv1pth):
        return
    image,mv1,mask = exr_imread(source)
    if half:
        image = cv2.resize(image,None,fx=resize_rate,fy=resize_rate)
        mv1 = cv2.resize(mv1,None,fx=resize_rate,fy=resize_rate)
        mask = cv2.resize(mask,None,fx=resize_rate,fy=resize_rate)
    
    cv2.imwrite(imgpth,image[...,::-1])
    write_flo_file(mv1, mv1pth)
    cv2.imwrite(maskpth,np.stack([mask,mask,mask],axis=2))

    image[np.where(mask==0)] = 0
    mv1[np.where(mask==0)] = 0
    cv2.imwrite(mimgpth,image[...,::-1])
    write_flo_file(mv1, mmv1pth)

if __name__=='__main__':
    assert len(sys.argv)==3,'usage python exr_get_rgb.py root output_root'
    source_root = sys.argv[1]
    save_path = sys.argv[2]
    resize_rate = 1
    half = False
    sources = jhelp(source_root)
    image_save_path = os.path.join(save_path,'image')
    mv1_save_path = os.path.join(save_path,'mv1')
    mask_save_path = os.path.join(save_path,'mask')
    maskimg_save_path = os.path.join(save_path,'image_masked')
    mv1_masked_save_path = os.path.join(save_path,'mv1_masked')

    mkdir(image_save_path)
    mkdir(mv1_save_path)
    mkdir(mask_save_path)
    mkdir(maskimg_save_path)
    mkdir(mv1_masked_save_path)

    data = [(sources[i],half,resize_rate) for i in range(len(sources))]
    process_map(subpro, data, max_workers=64)
    # for i in tqdm(range(len(sources))):
    #     source = sources[i]
    #     name = os.path.basename(source).replace('.exr','')
    #     imgpth = os.path.join(image_save_path,name)+'.png'
    #     maskpth = os.path.join(mask_save_path,name)+'.png'
    #     mimgpth = os.path.join(maskimg_save_path,name)+'.png'
    #     mv1pth = os.path.join(mv1_save_path,name)+'.flo'
    #     mmv1pth = os.path.join(mv1_masked_save_path,name)+'.flo'
    #     if os.path.isfile(mmv1pth):
    #         continue
    #     image,mv1,mask = exr_imread(source)
    #     if half:
    #         image = cv2.resize(image,None,fx=resize_rate,fy=resize_rate)
    #         mv1 = cv2.resize(mv1,None,fx=resize_rate,fy=resize_rate)
    #         mask = cv2.resize(mask,None,fx=resize_rate,fy=resize_rate)
        
    #     cv2.imwrite(imgpth,image[...,::-1])
    #     write_flo_file(mv1, mv1pth)
    #     cv2.imwrite(maskpth,np.stack([mask,mask,mask],axis=2))

    #     image[np.where(mask==0)] = 0
    #     mv1[np.where(mask==0)] = 0
    #     cv2.imwrite(mimgpth,image[...,::-1])
    #     write_flo_file(mv1, mmv1pth)

    
