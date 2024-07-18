'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-04-24 14:00:28
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
from myutil import jhelp,mkdir,read_txt
import os,shutil
from file_utils import read
import numpy as np
from collections import defaultdict
import Imath,OpenEXR
NO_COMPRESSION    = Imath.Compression(Imath.Compression.NO_COMPRESSION)
RLE_COMPRESSION   = Imath.Compression(Imath.Compression.RLE_COMPRESSION)
ZIPS_COMPRESSION  = Imath.Compression(Imath.Compression.ZIPS_COMPRESSION)
ZIP_COMPRESSION   = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
PIZ_COMPRESSION   = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
PXR24_COMPRESSION = Imath.Compression(Imath.Compression.PXR24_COMPRESSION)
def encode_pos(key):
    ncode = 0
    for i in range(6):
        if i in key:
            ncode += 16**i
    return ncode
def combine_result(args,masks_=None,right_masks_=None,multipro_nums=8):
    from tqdm.contrib.concurrent import process_map
    output = args.output
    scenes = jhelp(output)
    save_root = os.path.join(output,args.MM) 
    mkdir(save_root)
    for scene in scenes:
        if '/{}'.format(args.MM) in scene[-(len(args.MM)+1):] or not os.path.isdir(scene) or 'dumpedfile' in scene:
            continue
        if masks_ is not None:
            if scene.find('_mask[') != -1:
                mask_name = scene.split('_mask[')[1].split(']')[0]
            else:
                mask_name = args.front_mask_file
            masks = masks_[mask_name]
            if right_masks_ is not None:
                right_masks = right_masks_[mask_name]
        else:
            masks,right_masks = None,None
        
        contents = jhelp(scene)
        mvs = []
        for content in contents:
            if '_mv' in content:
                mvs.append((int)(content.split('_mv')[-1]))
        folder = set()
        for t in contents:
            if os.path.isdir(t) and 'dumpedfile' not in t:
                ctmp = '_mv{}'.format(t.split('_mv')[-1])
                folder.add(t.replace(ctmp,'_{}'))
        if len(folder) <= 0:
            raise FileNotFoundError('[MM ERROR][file]No result to combine!')
        while(len(folder)!=0):
            f = folder.pop()
            # mv_paths = [f.format(f'mv{i}') for i in mvs]
            tmp = os.path.join(os.path.abspath(os.path.join(f.format(f'mv{mvs[0]}'),'..')),'scene_change.txt')
            scene_change_file = None if not os.path.isfile(tmp) else tmp
            scene_change = []
            if scene_change_file is not None:
                scene_change = [int(a.rstrip()) for a in read_txt(scene_change_file)]
            mvall = defaultdict()
            meta_pos_encode = encode_pos(mvs)
            for i in mvs:
                mvall[f'mv{i}'] = jhelp(f.format(f'mv{i}'))
            # mvall = [jhelp(f.format(f'mv{i}')) for i in mvs]
            data = [(i,scene,mvall,masks,right_masks,scene_change,args.MM,args.compress_method.lower(),meta_pos_encode) for i in range(len(mvall[list(mvall.keys())[0]]))]
            process_map(combine_result_subpro, data, max_workers=multipro_nums)
            if scene_change_file is not None:
                keyword = '/{}/'.format(os.path.basename(scene))
                tmpnum = scene_change_file[::-1].find(keyword[::-1])
                sp = scene_change_file[:-tmpnum-len(keyword)] + scene_change_file[-tmpnum-len(keyword):].replace(keyword,'/{}'.format(args.MM)+keyword)
                shutil.copy(scene_change_file,sp)
        if args.clean_source:
            shutil.rmtree(scene)

def combine_result_subpro(args):
    i,scene,mvs_,masks,right_masks,scene_change,MM,compress,meta_pos_encode = args
    mvs = defaultdict()
    for key in mvs_.keys():
        tmp = read(mvs_[key][i],type='flo',OPENEXR=True)
        mvs[key] = [tmp[...,0],tmp[...,1],tmp[...,2],tmp[...,-1]]
    # sp = mv0s[i].replace('/'+os.path.basename(scene)+'/','/MM/'+os.path.basename(scene)+'/')
    keyword = '/' + os.path.basename(scene) + '/'
    mvtmp = mvs_[key]
    tmpnum = mvtmp[i][::-1].find(keyword[::-1])
    sp = mvtmp[i][:-tmpnum-len(keyword)] + mvtmp[i][-tmpnum-len(keyword):].replace(keyword,'/{}'.format(MM)+keyword)
    sp = sp.replace(f'_{key}/','/')
    #change to MM_FILE_LEFT
    basename = os.path.basename(os.path.abspath(os.path.join(sp,'..')))
    rflag = False
    if 'left' in basename:
        sp = sp.replace(basename,'MM_FILE_LEFT')
    elif 'right' in basename:
        sp = sp.replace(basename,'MM_FILE_RIGHT')
        rflag=True
    else:
        sp = sp.replace(basename,'MM_FILE')

    mkdir(os.path.abspath(os.path.join(sp,'..')))
    sz = mvs[key][0].shape[:2]
    header = OpenEXR.Header(sz[1], sz[0])
    # header['channels'] = {
    #     'MV0.x': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
    #     'MV0.y': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
    #     'MV0.z': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
    #     'MV1.x': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
    #     'MV1.y': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
    #     'MV1.z': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
    #     'Depth': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
    #     'Matte': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
    # }
    header['channels'] = {
    }
    for key in mvs.keys():
        header['channels'][f'{key}'.upper()+'.x'] = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'][f'{key}'.upper()+'.y'] = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'][f'{key}'.upper()+'.z'] = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    #extra depth and matte
    header['channels']['Depth'] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels']['Matte'] = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    if compress.lower() == 'rle':
        cpm = RLE_COMPRESSION
    elif compress.lower() == 'zips':
        cpm = ZIPS_COMPRESSION
    elif compress.lower() == 'zip':
        cpm = ZIP_COMPRESSION
    elif compress.lower() == 'piz':
        cpm = PIZ_COMPRESSION
    elif compress.lower() == 'pxr':
        cpm = PXR24_COMPRESSION
    else:
        cpm = NO_COMPRESSION
    header['compression'] = cpm
    header['SC_Flag'] = False if i not in scene_change else True
    header['BA_Step'] = True
    header['MV_Valid_Code']=meta_pos_encode
    out = OpenEXR.OutputFile(sp, header)
    Matte = np.zeros(sz,dtype=np.float16).tobytes()
    datas = defaultdict()
    if masks is not None:
        mask = read(masks[os.path.basename(scene).split('_mask[')[0]][0][i],type='mask') if not rflag else read(right_masks[os.path.basename(scene).split('_mask[')[0]][0][i],type='mask')
        if mask.sum() != 0:
            Matte = mask.astype(np.float16).tobytes()
    datas['Matte'] = Matte
    # Depth_value = np.zeros(sz,dtype=np.float32) if mv0_A.sum() == 0 else mv0_A.tobytes()
    # mv0r_value = np.zeros(sz,dtype=np.float16) if mv0_R.sum() == 0 else mv0_R.astype(np.float16).tobytes()
    # mv0g_value = np.zeros(sz,dtype=np.float16) if mv0_G.sum() == 0 else mv0_G.astype(np.float16).tobytes()
    # mv0b_value = np.zeros(sz,dtype=np.float16) if mv0_B.sum() == 0 else mv0_B.astype(np.float16).tobytes()
    # mv1r_value = np.zeros(sz,dtype=np.float16) if mv1_R.sum() == 0 else mv1_R.astype(np.float16).tobytes()
    # mv1g_value = np.zeros(sz,dtype=np.float16) if mv1_G.sum() == 0 else mv1_G.astype(np.float16).tobytes()
    # mv1b_value = np.zeros(sz,dtype=np.float16) if mv1_B.sum() == 0 else mv1_B.astype(np.float16).tobytes()
    for key in mvs.keys():
        datas[f'{key}'.upper()+'.x'] = np.zeros(sz,dtype=np.float16) if mvs[key][0].sum() == 0 else mvs[key][0].astype(np.float16).tobytes()
        datas[f'{key}'.upper()+'.y'] = np.zeros(sz,dtype=np.float16) if mvs[key][1].sum() == 0 else mvs[key][1].astype(np.float16).tobytes()
        datas[f'{key}'.upper()+'.z'] = np.zeros(sz,dtype=np.float16) if mvs[key][2].sum() == 0 else mvs[key][2].astype(np.float16).tobytes()
    mv0_A = mvs[list(mvs.keys())[0]][3]
    Depth_value = np.zeros(sz,dtype=np.float32) if mv0_A.sum() == 0 else mv0_A.tobytes()
    datas['Depth'] = Depth_value
    out.writePixels(datas)


def create_mdl(root,scs,inits,MM,fps=24):
    save_path = os.path.join(root,MM)
    framerate = round(fps)
    def time_code(framenum):
        nonlocal framerate
        frame_idx= framenum % framerate
        sec = framenum / framerate
        min = sec/60
        hour = min/60
        sec = sec % 60
        min = min % 60
        return '{:0>2}:{:0>2}:{:0>2}:{:0>3}'.format(int(hour),int(min),int(sec),int(frame_idx))
    gfn = 86400
    # scene_change = np.array([],dtype='uint8')
    # last_frame = 0
    # for scene in scs.keys():
    #     scene_change_result = scs[scene] + last_frame
    #     scene_change = np.concatenate((scene_change,scene_change_result))
    #     last_frame += scs[scene][-1]
    # if len(scene_change)>2:
    for scene in scs.keys():
        init = inits[scene]
        txt = "TITLE: {}\r\nFCM: NON-DROP FRAME\r\nFPS: {:.4f}\r\n\r\n".format(save_path,framerate)
        sp = os.path.join(save_path,scene,'{}.mdl'.format(scene))
        scene_change = scs[scene]
        if scene_change[0]!=0:
            scene_change = np.concatenate(([0],scene_change))
        if len(scene_change)<2:
            raise RuntimeError('[MM ERROR][file]scene_change files error')
        for i in range(len(scene_change)-1):
            sc = scene_change[i]+init
            se = scene_change[i+1]+init
            txt += "{:0>3} AX       V     C    {} {} {} {}\r\n".format(i+1,time_code(sc),time_code(se),time_code(sc+gfn),time_code(se+gfn))
            txt += '*TC_BAJA (0 360) (0 4 2 2) (1 2 1)\r\n\r\n'
        with open(sp,'w') as f:
            f.write(txt)