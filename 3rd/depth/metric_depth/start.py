'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2025-02-19 14:16:26
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
    mkdir(os.path.dirname(destination))
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
    parser.add_argument('--root', type=str,required=True)
    parser.add_argument('--input-size', type=int, default=518)
    # parser.add_argument('--output', type=str, default='./vis_depth')
    parser.add_argument('--device', type=str, default='auto',help='auto,cpu,cuda,mps')
    parser.add_argument('--algo', type=str, default='depth_anything_v2_metric_hypersim_vitl',help='depth mode')
    # parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    # parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    parser.add_argument('--server', type=str, default='http://10.35.116.93:8088')
    parser.add_argument('--img_folder_name', type=str, help='batch run"s folder name')
    
    # parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    # parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--color', action='store_true', help='apply colorful palette')
    parser.add_argument('--inverse_depth', action='store_true', help='apply colorful palette')
    args = parser.parse_args()
    
    DEVICE = args.device.lower()
    if DEVICE == 'auto':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    assert DEVICE in ['cpu','mps','cuda'],f'not supported device :{DEVICE}!, please use mps , cuda or cpu'
    metric = True if '_metric_' in args.algo else False
    if metric:
        from depth_anything_v2.dpt import DepthAnythingV2
    else:
        from depth_anything_v2_s.dpt import DepthAnythingV2
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    mkdir(os.path.join(fp,'checkpoints'))
    dtype = args.algo.split('_')[-1]
    if 'vits' in args.algo:
        dtype = 'vits'
    elif 'vitb' in args.algo:
        dtype = 'vitb'
    elif 'vitl' in args.algo:
        dtype = 'vitl'
    elif 'vitg' in args.algo:
        dtype = 'vitg'
    else:
        raise NotImplementedError('not support algo')
    #处理DA-2模型参数
    # DA2-metric-b-ue-ep8-MD1000-R1918_1008
    input_size = (args.input_size,args.input_size)
    if 'DA2' in args.algo:
        import re
        md_match = re.search(r"MD(\d+)", args.algo)
        r_match = re.search(r"R(\d+_\d+)", args.algo)
        if md_match and r_match:
            md_number = md_match.group(1)  # 1000
            r_numbers = r_match.group(1).split('_')  # 1918_1008
        args.max_depth = int(md_number)
        input_size = (int(r_numbers[1]),int(r_numbers[0]))

        
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
    # encoder = args.algo.split('_')[-1]
    mmc = model_configs[dtype]
    depth_anything = DepthAnythingV2(**{**model_configs[dtype], 'max_depth': args.max_depth}) if metric else DepthAnythingV2(**{**model_configs[dtype]})
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
        
        
        
        for k, filename in enumerate(filenames):
            print(f'Progress {k+1}/{len(filenames)}: {filename}')
            
            raw_image = read(filename,type='image')
            # print(raw_image.shape, args.input_size,DEVICE)
            depth = depth_anything.infer_image(raw_image, args.input_size,DEVICE)
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
                cmap = matplotlib.colormaps.get_cmap('Spectral')
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
            d = depth[...,0]
            d = (d - d.min()) / (d.max() - d.min())
            # if args.norm:
            if metric:
                if args.inverse_depth:
                    d = 1 - d
                depth[...,-1] = d
            else:
                depth = np.repeat(d[...,None],4,axis=2)
                if args.inverse_depth:
                    depth[...,-1] = 1-depth[...,-1]
            # mvwrite(os.path.join(args.outdir,os.path.basename(task),'mono_depth',os.path.basename(filename[:filename.rfind('.')]) + '.exr'),depth,precision='half')
            
            mvwrite(output_path,depth,precision='half')
