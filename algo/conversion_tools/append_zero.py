'''
Author: Qing Hong
Date: 2023-07-10 10:48:12
LastEditors: QingHong
LastEditTime: 2023-07-10 10:54:15
Description: file content
'''
import re,os,sys,shutil
dir_mytest = os.path.dirname(os.path.abspath(__file__))+'/..'
sys.path.insert(0, dir_mytest)
from myutil import jhelp,write


if __name__ == '__main__':
    assert len(sys.argv) ==3 or  len(sys.argv) ==2,'usage: python append_zero root (append)'
    root = sys.argv[1]
    append = '' if len(sys.argv) ==2 else sys.argv[2]+'_'
    datas = jhelp(root)
    for data in datas:
        num = re.findall('\d+',data)[-1]
        typ = os.path.basename(data).split('.')[-1]
        res = os.path.join(os.path.dirname(data),'{}{:0>6}.'.format(append,num)+typ)
        shutil.move(data,res)