'''
Author: Qing Hong
Date: 2023-08-14 12:19:57
LastEditors: Qing Hong
LastEditTime: 2024-05-27 15:00:24
Description: file content
'''
from glob import glob
import sys,shutil,os


def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))

if __name__ == '__main__':
    # a = [os.path.join(root,i) for i in sorted(list(filter(lambda x:x[0]!='.',os.listdir(root))))]
    assert len(sys.argv)==2 ,'usage: python x.py root name targetname'
    root = sys.argv[1]
    fss = jhelp_folder(root)
    assert len(fss)>0 ,'root error'
    for fs in fss:
        root_name = os.path.basename(fs)
        sfss = jhelp_folder(fs)
        for sfs in sfss:
            concanate_name = os.path.basename(fs) +'_'+ os.path.basename(sfs)
            
            newname = os.path.join(os.path.dirname(sfs),concanate_name)
            shutil.move(sfs,newname)
            shutil.move(newname,root)
            print(sfs,os.path.join(root,os.path.basename(concanate_name)))
        shutil.rmtree(fs,ignore_errors=True)
            
            # mkdir(concanate_name)
            
            # shutil.move(sfs,concanate_name)
            # print(f'{newname}->{fs}')