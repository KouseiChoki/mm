'''
Author: Qing Hong
Date: 2023-08-03 14:32:52
LastEditors: Qing Hong
LastEditTime: 2024-04-30 10:21:17
Description: file content
'''
import sys,os

def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

if __name__ == '__main__':
    assert len(sys.argv)==3 or len(sys.argv)==4,'usage: python mmzipsplit your_zip(or folder) save_path'
    root = sys.argv[1]
    save_path = sys.argv[2]
    
    size = 900 if len(sys.argv)==3 else int(sys.argv[3])
    if '.zip' in os.path.basename(root) or '.tar' in os.path.basename(root):
        assert os.path.isfile(root)
        cmd = f"gsplit -d -a 3 -b {size}m {root} {root}- && mv {root}-* {save_path}"
    else:
        assert os.path.isdir(root)
        path = root
        root = f'{root}.zip'
        cmd = f"tar cf {root} {path} && gsplit -d -a 3 -b {size}m {root} {root}- && mv {root}-* {save_path}"
    mkdir(save_path)
    # print(cmd)
    os.system(cmd)
    unzip = f'cat {os.path.basename(root)}-* > {os.path.basename(root)}'
    print(f'please use command:[{unzip}] to merge zip file')