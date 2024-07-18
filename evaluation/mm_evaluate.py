'''
Author: Qing Hong
Date: 2022-05-25 17:35:01
LastEditors: QingHong
LastEditTime: 2024-04-16 10:43:53
Description: file content
'''
import os,sys
cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cur_path+'/../algo')
import shutil
from config_process import init_param
from evaluate_core import mm_evaluate,show_evaluation
import re
import datetime
import argparse
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
##载入config文件
def run(args,skip=False,att=None):
    gpu = args.gpu
    gpu = gpu.rstrip().split(',')
    num_gpu = len(gpu)
    cmd = 'torchrun --nproc_per_node={} algo/main.py {}'.format(num_gpu,config_file)
    print('======================================== optical flow start ======================================== ')
    print('info:')
    print('algorithm:               {}'.format(args.algorithm))
    print('numbers of core:         {}'.format(num_gpu))
    if att:
        cmd +=f' {att}'
    try:
        if not skip:
            shutil.rmtree(args.output,ignore_errors=True)
            os.system(cmd)
            print('======================================== mm evaluation start ======================================== ')
        obj_mode,fg = False,True
        if '-mask-' in args.algorithm and '-mix-' not in args.algorithm:
            obj_mode = True
            if '-bg-' in args.algorithm:
                fg = False
        res = mm_evaluate(args.output,args.root,obj_mode=obj_mode,fg=fg)
        now = datetime.datetime.now()
        # 格式化为年月日小时分钟秒
        formatted_date = now.strftime("%Y%m%d_%H%M%S")  # 输出格式类似 '20220429_153142'
        # 使用这个字符串来命名文件
        filename = f"data_{formatted_date}_{args.algorithm}.txt"  # 文件名类似 'data_20220429_153142.txt'
        filename = os.path.join(cur_path,filename)
        show_evaluation(res,args.algorithm,sp=filename)
        # pass
    except Exception as e:
        print(e)
        sys.exit('[MM ERROR][main process]main process error')
def change_algorithm(filename, target, new_value):
    pattern = re.compile(rf'^({re.escape(target)}\s*=\s*).+$')
    # 读取原始文件内容
    with open(filename, 'r') as file:
        lines = file.readlines()
    # 修改指定行的内容
    with open(filename, 'w') as file:
        for line in lines:
            if pattern.match(line):
                file.write(pattern.sub(rf'\1{new_value}', line))
            else:
                file.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', required=True, help="name your algorithm")
    parser.add_argument('--skip', action='store_true')
    args = parser.parse_args()
    config_file = os.path.join(cur_path,'config_evaluation')
    if not os.path.isfile(config_file):
        config_file = os.path.join(cur_path,'evaluation/config_evaluation')
    if not os.path.isfile(config_file):
        raise FileExistsError('[MM ERROR][config]config file not exist!')
    cfg_ = os.path.join(cur_path,config_file)
    change_algorithm(cfg_,'algorithm',args.algorithm)
    cfg = init_param(cfg_)
    run(cfg,skip=args.skip)

    

