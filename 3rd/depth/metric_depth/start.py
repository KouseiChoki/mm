'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-07-18 14:33:26
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
import cv2
import glob
import numpy as np
import os
import torch
from depth_anything_v2.dpt import DepthAnythingV2
from file_utils import mvwrite,read,mkdir
import matplotlib
import requests
from tqdm import tqdm
fp = os.path.dirname(os.path.abspath(__file__))
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
def download_file(url, destination):
    """下载文件，显示进度条"""
    response = requests.get(url, stream=True,timeout=30)
    if response.status_code != 200:
        return False
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("error: connection failed, please retry later。")
    else:
        print("finished")
    return True

def check_and_download_pth_file(file_path, download_url):
    flag = False
    """检查.pth文件是否存在，如果不存在，则从URL下载"""
    if not os.path.exists(file_path):
        print(f"file {file_path} not exist, downloading...")
        flag = download_file(download_url, file_path)
    return flag


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    parser.add_argument('--name', type=str, default='mono_depth',help='depth name')
    parser.add_argument('--root', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    # parser.add_argument('--output', type=str, default='./vis_depth')
    parser.add_argument('--device', type=str, default='cpu',help='cpu,cuda,mps')
    parser.add_argument('--algo', type=str, default='depth_anything_v2_metric_hypersim_vitl',help='depth mode')
    # parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    # parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    parser.add_argument('--server', type=str, default='http://10.35.116.93:8088')
    parser.add_argument('--img_folder_name', type=str, help='batch run"s folder name')
    
    # parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    # parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--color', action='store_true', help='apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = args.device.lower()
    assert DEVICE in ['cpu','mps','cuda'],f'not supported device :{DEVICE}!, please use mps , cuda or cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    mkdir(os.path.join(fp,'checkpoints'))
    ckpt_path = os.path.join(fp,'checkpoints',args.algo)+'.pth'
    if not os.path.isfile(ckpt_path):
        download_url = ckpt_path
        md = args.server
        md += '/depth'
        md += '/' + args.algo + '.pth'
        print(download_url,md)
        flag = check_and_download_pth_file(download_url,md)
        if not flag:
            raise NotImplementedError(f'[MM ERROR][model]model file not exists:{args.algo},please use mmalgo to check')
        # raise NotImplementedError(f'[MM ERROR][model]model file loss!{model_name}')
    args.load_from = ckpt_path
    encoder = args.algo.split('_')[-1]
    mmc = model_configs[encoder]
    depth_anything = DepthAnythingV2(**{**model_configs['vitl'], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    prepares = []
    if args.img_folder_name is not None:
        prepares = []
        for dirpath, dirnames, filenames in os.walk(args.root):
            if args.img_folder_name in dirnames:
                prepares.append(os.path.join(dirpath,args.img_folder_name))
    else:
        prepares = [args.root]

    for prepare in prepares:
        filenames = glob.glob(os.path.join(prepare, '**/*'), recursive=True)
        
        cmap = matplotlib.colormaps.get_cmap('Spectral')
        
        for k, filename in enumerate(filenames):
            print(f'Progress {k+1}/{len(filenames)}: {filename}')
            
            raw_image = cv2.imread(filename)
            
            depth = depth_anything.infer_image(raw_image, args.input_size)
            # if args.save_numpy:
            #     output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
            #     np.save(output_path, depth)
            
            # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            # depth = depth.astype(np.uint8)
            
            # if args.grayscale:
            #     depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            # else:
            #     depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            output = os.path.dirname(os.path.abspath(filename))+'/..'
            last = os.path.basename(filename).split('.')[-1]
            output_path = os.path.join(output, f'{args.name}',os.path.basename(filename).replace(f'.{last}',''))+'.exr'
            if args.color:
                import copy
                tmpdepth = copy.deepcopy(depth)
                tmpdepth = (tmpdepth - tmpdepth.min()) / (tmpdepth.max() - tmpdepth.min()) * 255.0
                tmpdepth = tmpdepth.astype(np.uint8)
                tmpdepth = (cmap(tmpdepth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                mkdir(os.path.join(output, f'{args.name}_color'))
                cv2.imwrite(os.path.join(output, f'{args.name}_color',os.path.basename(filename).replace(f'.{last}',''))+'.png', tmpdepth)
            # output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
            # if args.pred_only:
            #     cv2.imwrite(output_path, depth)
            # else:
            #     split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            #     combined_result = cv2.hconcat([raw_image, split_region, depth])
                
            #     cv2.imwrite(output_path, combined_result)
            depth = np.repeat(depth[...,None],4,axis=2)
            # if args.norm:
            if True:
                d = depth[...,0]
                d = (d - d.min()) / (d.max() - d.min())
                depth[...,-1] = d
            # mvwrite(os.path.join(args.outdir,os.path.basename(task),'mono_depth',os.path.basename(filename[:filename.rfind('.')]) + '.exr'),depth,precision='half')
            
            mvwrite(output_path,depth,precision='half')