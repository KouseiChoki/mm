'''
Author: Qing Hong
Date: 2023-06-15 14:56:34
LastEditors: QingHong
LastEditTime: 2023-06-15 15:07:32
Description: file content
'''
import sys,os
sys.path.append('..')
import numpy as np
from myutil import jhelp_file,mkdir,write,read
from tqdm import tqdm
def process(file):
    image = read(file,type='image')
    return np.ascontiguousarray(image[:910,188:1732])



if __name__ == '__main__':
    assert len(sys.argv) ==3 ,'usage: python image_crop root output'
    root = sys.argv[1]
    output = sys.argv[2]
    mkdir(output)
    files = jhelp_file(root)
    for i in tqdm(range(len(files))):
        file = files[i]
        out_file = process(file)
        write(os.path.join(output,os.path.basename(file)),out_file)
        