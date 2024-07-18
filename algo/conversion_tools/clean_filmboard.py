'''
Author: Qing Hong
Date: 2023-01-10 18:58:08
LastEditors: QingHong
LastEditTime: 2023-01-11 15:02:24
Description: file content
'''
import os
import cv2
from tqdm import tqdm
def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

root = '/Volumes/Elements SE/tt'
target = '/Volumes/Elements SE/tt_output'
mkdir(target)
def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
root_files = jhelp(root)



def get_board_length(image):
    h,w,c = image.shape
    res = [0,h,0,w,h,w]
    if all(image[0,0] == [0,0,0]) or all(image[0,0] == [1,1,1]) or all(image[0,0] == [2,2,2]):
        for i in range(h):
            if image[i].sum()>h*11:
                break
            res[0] = i+1
        for i in range(w):
            if image[:,i].sum()>w*11 or True:
                break
            res[2] = i+1
    if all(image[-1,-1] == [0,0,0])or all(image[-1,-1] == [1,1,1]) or all(image[-1,-1] == [2,2,2]):
        for i in range(h-1,-1,-1):
            if image[i].sum()>h*11:
                break
            res[1] = i
        for i in range(w-1,-1,-1):
            if image[:,i].sum()>w*11 or True:
                break
            res[3] = i
    return res


for root_file in root_files:
    target_folder = os.path.join(target,os.path.basename(root_file))
    mkdir(target_folder)
    images = jhelp(root_file)
    film_board = [140, 940, 0, 1920, 1080, 1920]
    for index in tqdm(range(len(images))):
        image = images[index]
        img = cv2.imread(image)
        if len(film_board)==0:
            film_board = get_board_length(img)
        if sum(film_board)>0:
            img = img[film_board[0]:film_board[1],film_board[2]:film_board[3]]
        if img.shape[1]>1300:
            img = cv2.resize(img,None,fx=0.5,fy=0.5)
        target_pos = os.path.join(target_folder,os.path.basename(image).replace('.tiff','.png'))
        cv2.imwrite(target_pos,img)