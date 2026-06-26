'''
Author: Qing Hong
Date: 2024-01-09 11:14:18
LastEditors: Qing Hong
LastEditTime: 2026-06-23 11:23:08
Description: file content
'''
import os,sys
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import configparser
def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
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

def getlink(url):
    res = []
    response = requests.get(url,timeout=5)
    if response.status_code == 200:
        # 使用 BeautifulSoup 解析 HTML 内容
        soup = BeautifulSoup(response.content, 'html.parser')
        # 假设文件链接在 <a> 标签中
        for link in soup.find_all('a'):
            res.append(link.get('href'))
    else:
        raise RuntimeError('[MM ERROR][server] file server connect failed!')
    return res

def getsetsumei(url):
    response = requests.get(url, stream=True,timeout=30)
    response.encoding = 'utf-8'
    return response.text.splitlines()

def get_all_pth(url):
    # url = 'http://10.35.116.93:8088'
    fg = list(filter(lambda x:'.pth' in x,getlink(url + '/fg')))
    bg = list(filter(lambda x:'.pth' in x,getlink(url + '/bg')))
    fm = list(filter(lambda x:'.pth' in x,getlink(url + '/fm')))
    mix = list(filter(lambda x:'.pth' in x,getlink(url + '/mix')))
    depth= list(filter(lambda x:'.pth' in x,getlink(url + '/depth')))
    mma_fm = list(filter(lambda x:'.pth' in x,getlink(url + '/mma_fm')))
    denoise = list(filter(lambda x:'.pth' in x,getlink(url + '/denoise')))
    vfi = list(filter(lambda x:'.pth' in x,getlink(url + '/vfi')))
    response = getsetsumei(url+'/setsumei.txt')
    for p in [fm,fg,bg,mix,depth,mma_fm,denoise]:
        if p ==fm:
            print('--------------mm frame algorithm--------------')
        if p ==mma_fm:
            print('--------------mma frame algorithm--------------')
        if p ==fg:
            print('--------------fg algorithm--------------')
        if p ==bg:
            print('--------------bg algorithm--------------')
        if p ==mix:
            print('--------------mix algorithm--------------')
        if p ==depth:
            print('--------------depth algorithm--------------')
        if p ==denoise:
            print('--------------denoise algorithm--------------')
        if p ==vfi:
            print('--------------vfi algorithm--------------')
        for pp in p:
            pp = pp.replace('.pth','')
            setsu = 'No describe' if pp not in response else response[response.index(pp)+1]
            print(pp+'  :  '+setsu)
    print('-----------使用方式：直接在mm config中将algorithm替换成上述模型名即可，如出现网络原因请从nas云端获取------------')


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(os.path.dirname(os.path.abspath(__file__))+'/../../config', encoding="utf-8")
    url = config.get('opticalflow','file_server_ip')
    get_all_pth(url)
    
# # 示例使用
# pth_file_path = '/Users/qhong/Documents/1117test/MM/motionmodel/test/1.pth'
# download_url = 'http://10.35.116.93:8088/fg/kousei-mask-fg-v0-230808300.pth'
# config = configparser.ConfigParser()
# config.read(os.path.dirname(os.path.abspath(__file__))+'/../../config', encoding="utf-8")
# url = config.get('opticalflow','file_server_ip')
# check_and_download_pth_file(pth_file_path, download_url)