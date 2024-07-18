'''
Author: Qing Hong
Date: 2024-01-09 11:14:18
LastEditors: Qing Hong
LastEditTime: 2024-07-09 15:45:11
Description: file content
'''
import os,sys
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import configparser
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
    datas = str(response.content).split('/')

def get_all_pth(url):
    # url = 'http://10.35.116.93:8088'
    fg = list(filter(lambda x:'.pth' in x,getlink(url + '/fg')))
    bg = list(filter(lambda x:'.pth' in x,getlink(url + '/bg')))
    fm = list(filter(lambda x:'.pth' in x,getlink(url + '/fm')))
    mix = list(filter(lambda x:'.pth' in x,getlink(url + '/mix')))
    depth= list(filter(lambda x:'.pth' in x,getlink(url + '/depth')))
    response = requests.get(url+'/setsumei.txt', stream=True,timeout=30).text.splitlines()
    for p in [fm,fg,bg,mix,depth]:
        if p ==fm:
            print('--------------frame algorithm--------------')
        if p ==fg:
            print('--------------fg algorithm--------------')
        if p ==bg:
            print('--------------bg algorithm--------------')
        if p ==mix:
            print('--------------mix algorithm--------------')
        if p ==depth:
            print('--------------depth algorithm--------------')
        for pp in p:
            pp = pp.replace('.pth','')
            setsu = 'No describe' if pp not in response else response[response.index(pp)+1]
            print(pp+'  :  '+setsu)


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