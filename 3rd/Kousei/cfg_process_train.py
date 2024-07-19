'''
Author: Qing Hong
Date: 2024-03-07 17:18:49
LastEditors: QingHong
LastEditTime: 2024-04-08 16:24:31
Description: file content
'''
import argparse,configparser
import re
def init_dict():
    bool_dict = ['reload_data','do_flip','kata_mask_mode']
    int_dict = ['nums_gpu','initial_config','vis_iter','save_iter','skip_frames','version']
    float_dict = ['nums_gpu','max_mv_x','max_mv_y','reverse_rate','repeat_frame_rate']
    arr_dict = ['sintel_clean','sintel_final','things_clean','things_final','unreal_clean','unreal_final','spring_clean','unreal_mrq_clean','unreal_mrq_final','fps']
    return bool_dict,int_dict,float_dict,arr_dict

def getarr(key):
    res = []
    assert '[' in key and ']' in key,'error config'
    c = re.findall('[-+]?[0-9]*\.?[0-9]+',key)
    for i in c:
        res.append(float(i))
    return res
    
def init_param(path):
    bool_dict,int_dict,float_dict,arr_dict = init_dict()
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help="test mode.")
    args = parser.parse_args([])
    config = configparser.ConfigParser()
    config.read(path, encoding="utf-8")
    for key in config['kousei'].keys():
        if key in bool_dict:
            trans_key = config.getboolean('kousei',key)
        elif key in int_dict:
            trans_key = config.getint('kousei',key)
        elif key in float_dict:
            trans_key = config.getfloat('kousei',key)
        elif key in arr_dict:
            trans_key = getarr(config.get('kousei',key))
        else:
            trans_key = config.get('kousei',key)
        args.__setattr__(key,trans_key)
    return args