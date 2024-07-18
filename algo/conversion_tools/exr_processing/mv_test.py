'''
Author: Qing Hong
Date: 2023-12-11 16:59:44
LastEditors: QingHong
LastEditTime: 2023-12-11 17:00:47
Description: file content
'''
import OpenEXR
import numpy as np
import array 
import Imath
from file_utils import mvwrite
pt = Imath.PixelType(Imath.PixelType.FLOAT)

filePath = ''
img_exr = OpenEXR.InputFile(filePath)
dw = img_exr.header()['dataWindow']
size = (dw.max.y - dw.min.y + 1,dw.max.x - dw.min.x + 1)
mv0_x = np.array(array.array('f', img_exr.channel('FinalImagePWMV0.R',pt))).reshape(size)
mv0_y = np.array(array.array('f', img_exr.channel('FinalImagePWMV0.G',pt))).reshape(size)
mv0_z = np.array(array.array('f', img_exr.channel('FinalImagePWMV0.B',pt))).reshape(size)
mv0_a = np.array(array.array('f', img_exr.channel('FinalImagePWMV0.A',pt))).reshape(size)
mv1_x = np.array(array.array('f', img_exr.channel('FinalImagePWMV1.R',pt))).reshape(size)
mv1_y = np.array(array.array('f', img_exr.channel('FinalImagePWMV1.G',pt))).reshape(size)
mv1_z = np.array(array.array('f', img_exr.channel('FinalImagePWMV1.B',pt))).reshape(size)
mv1_a = np.array(array.array('f', img_exr.channel('FinalImagePWMV1.A',pt))).reshape(size)
#denormalized

mv0_x = (mv0_x - 0.5) * 2 * size[1] * -1
mv0_y = (mv0_y - 0.5) * 2 * size[0]
mv1_x = (mv1_x - 0.5) * 2 * size[1] * -1
mv1_y = (mv1_y - 0.5) * 2 * size[0]
mv0 = np.stack([mv0_x,mv0_y,mv0_z,mv0_a],axis=2)
mv1 = np.stack([mv1_x,mv1_y,mv1_z,mv1_a],axis=2)
mv0 /= size[1]
mv1 /= size[0]
mvwrite('/Users/qhong/Desktop/mv1.exr',mv0)
mvwrite('/Users/qhong/Desktop/mv0.exr',mv1)