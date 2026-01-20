'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2026-01-19 15:58:30
Description: 
         ▄              ▄
        ▌▒█           ▄▀▒▌     
        ▌▒▒▀▄       ▄▀▒▒▒▐
       ▐▄▀▒▒▀▀▀▀▄▄▄▀▒▒▒▒▒▐     ,-----------------.
     ▄▄▀▒▒▒▒▒▒▒▒▒▒▒█▒▒▄█▒▐     (Wow,kousei's code)
   ▄▀▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▀██▀▒▐     `-,---------------' 
  ▐▒▒▒▄▄▄▒▒▒▒▒▒▒▒▒▒▒▒▒▀▄▒▒▌  _.-'   ,----------.
  ▌▒▒▐▄█▀▒▒▒▒▄▀█▄▒▒▒▒▒▒▒█▒▐         (surabashii)
 ▐▒▒▒▒▒▒▒▒▒▒▒▀██▀▒▒▒▒▒▒▒▒▀▄▌        `-,--------' 
 ▌▒▀▄██▄▒▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒▌      _.-'
 ▌▀▐▄█▄█▌▄▒▀▒▒▒▒▒▒░░░░░░▒▒▒▐ _.-'
▐▒▀▐▀▐▀▒▒▄▄▒▄▒▒▒▒▒░░░░░░▒▒▒▒▌
▐▒▒▒▀▀▄▄▒▒▒▄▒▒▒▒▒▒░░░░░░▒▒▒▐
 ▌▒▒▒▒▒▒▀▀▀▒▒▒▒▒▒▒▒░░░░▒▒▒▒▌
 ▐▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▐
  ▀▄▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▄▒▒▒▒▌
    ▀▄▒▒▒▒▒▒▒▒▒▒▄▄▄▀▒▒▒▒▄▀
      ▀▄▄▄▄▄▄▀▀▀▒▒▒▒▒▄▄▀
         ▒▒▒▒▒▒▒▒▒▒▀▀
When I wrote this, only God and I understood what I was doing
Now, God only knows
'''
import argparse
from glob import glob
import os,sys
import re
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/..')
from file_utils import read,write

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    parser.add_argument('--kw', type=str, default='exr',help='exr or jpg')
    parser.add_argument('--root', type=str,required=True)
    parser.add_argument('--output', type=str,required=True)
    parser.add_argument('--reversey', action='store_true')
    parser.add_argument('--reversex', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--denormalize', action='store_true')
    args = parser.parse_args()

    
    filenames = glob(os.path.join(args.root, f'**/*.{args.kw}'),recursive=True)
    filenames = sorted(filenames, key=lambda x: int(re.findall(r'(\d+)', x)[-1]))
    for idx in tqdm(range(len(filenames))):
        filename = filenames[idx]
        data = read(filename)
        h,w,_ = data.shape
        if args.denormalize:
            data[...,0] *= w
            data[...,1] *= h
        if args.normalize:
            data[...,0] /= w
            data[...,1] /= h
        if args.reversey:
            data[...,1] *= -1
        if args.reversex:
            data[...,0] *= -1
        write(filename.replace(args.root,args.output),data)
    