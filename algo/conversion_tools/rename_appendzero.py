'''
Author: Qing Hong
Date: 2022-09-26 15:01:24
LastEditors: QingHong
LastEditTime: 2023-05-11 14:07:55
Description: file content
'''
import os,sys
from tqdm import tqdm
from myutil import jhelp,appendzero

if __name__ == '__main__':
    assert len(sys.argv)>1,'usage python '
    source = sys.argv[1]
    imgs = jhelp(source)
    for i in tqdm(range(len(imgs))):
        name = appendzero(os.path.basename(imgs[i]),)
        p1 = imgs[i]
        p2 = os.path.join(source,name)
        os.system('mv {} {}'.format(p1,p2))
        

    
    