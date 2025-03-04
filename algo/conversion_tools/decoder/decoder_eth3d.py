'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2025-02-19 16:16:17
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
import sys,os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/../..')
from file_utils import jhelp_file,write
import numpy as np
# import torch.nn.functional as F
import cv2
from tqdm import tqdm

HEIGHT, WIDTH = 4032, 6048
def decode_eth3d(root,res='2k'):
    datas = jhelp_file(root)
    for i in tqdm(range(len(datas))):
        data = datas[i]
        with open(data, "rb") as file:
                        binary_data = file.read()
        # Convert the binary data to a numpy array of 32-bit floats
        depth_decoded = np.frombuffer(binary_data, dtype=np.float32).copy()
        depth_decoded[depth_decoded == np.inf] = 0.0
        depth_decoded = depth_decoded.reshape((HEIGHT, WIDTH))
        new_pos = os.path.join(os.path.dirname(data)+'_decoded',os.path.basename(data).replace('.JPG','.exr'))
        if res=='2k':
            # depth_decoded = F.interpolate(depth_decoded[:, None], (1080, 1920), mode="bilinear", align_corners=True)[0, 0]
            depth_decoded = cv2.resize(depth_decoded, (1920, 1080), interpolation=cv2.INTER_LINEAR)
            new_pos = os.path.join(os.path.dirname(data)+'_decoded_2k',os.path.basename(data).replace('.JPG','.exr'))
        depth = np.repeat(depth_decoded[...,None],4,axis=2)
        d = depth[...,0]
        d = (d - d.min()) / (d.max() - d.min())
        # if args.norm:
        depth[...,-1] = d
        write(new_pos,depth)
        


if __name__ == '__main__':
    assert len(sys.argv)==2 ,'usage: python XXXX.py root'
    root = sys.argv[1]
    decode_eth3d(root)