'''
Author: Qing Hong
Date: 2023-08-14 12:19:57
LastEditors: QingHong
LastEditTime: 2023-08-14 12:32:16
Description: file content
'''
from glob import glob
import sys,shutil,os


def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))

def go_rename(folder,name,targetname):
     if os.path.basename(folder) == name:
          dirname = os.path.dirname(folder)
          print('rename-----{}->{}'.format(os.path.join(dirname,name),os.path.join(dirname,targetname)))
          shutil.move(os.path.join(dirname,name),os.path.join(dirname,targetname))

def helper(root,name,targetname):
    s_folder = jhelp_folder(root)
    if len(s_folder)>0:
        for folder in s_folder:
            helper(folder,name,targetname)
            go_rename(folder,name,targetname)
         

if __name__ == '__main__':
    # a = [os.path.join(root,i) for i in sorted(list(filter(lambda x:x[0]!='.',os.listdir(root))))]
    assert len(sys.argv)==4 ,'usage: python x.py root name targetname'
    root = sys.argv[1]
    name = sys.argv[2]
    targetname = sys.argv[3]
    s_folder = helper(root,name,targetname)