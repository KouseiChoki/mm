'''
Author: Qing Hong
Date: 2023-07-18 16:34:39
LastEditors: QingHong
LastEditTime: 2023-07-20 11:33:52
Description: file content
'''

import cv2
import sys,os
import numpy as np
dir_mytest = os.path.dirname(os.path.abspath(__file__))+'/../'
sys.path.insert(0, dir_mytest)
# Load image
def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))
from file_utils import read

def cal_border_length(image,threshold=5):
    if isinstance(image, str):
        image = read(image,type='gray')
    height, width = image.shape
    top_border,bottom_border,left_border,right_border = 0,0,0,0
    # Find top border
    for y in range(height):
        if np.mean(image[y, :]) > threshold:
            top_border = y
            break

    # Find bottom border
    for y in range(height - 1, -1, -1):
        if np.mean(image[y, :]) > threshold:
            bottom_border = height - y -1
            break

    # Find left border
    for y in range(width):
        if np.mean(image[:, y]) > threshold:
            left_border = y
            break

    # Find right border
    for y in range(width - 1, -1, -1):
        if np.mean(image[:, y]) > threshold:
            right_border = width - y - 1
            break

    return [top_border,bottom_border,left_border,right_border]

if __name__ == '__main__':
    assert len(sys.argv)==2 ,'usage: python filmborder_detection.py root(or image path)'
    root = sys.argv[1]
    file = root if os.path.isfile(root) else jhelp_file(root)[0]
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    bl = cal_border_length(img)
    print(bl)