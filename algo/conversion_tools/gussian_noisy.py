'''
Author: Qing Hong
Date: 2023-08-03 11:26:19
LastEditors: QingHong
LastEditTime: 2023-08-03 13:03:17
Description: file content
'''
import numpy as np
import torch

def add_grain(image, grain_amount):
    # 为图像添加"film grain"效果，噪声的强度由grain_amount控制
    # image: 原始图像
    # grain_amount: 噪声的强度
    
    # 将图像转换为float32以保持精度
    image = np.array(image, dtype=np.float32)
    
    # 生成与图像同样大小的噪声矩阵
    noise = np.random.randn(*image.shape) * grain_amount
    
    # 将噪声添加到图像中
    noisy_image = image + noise
    
    # 确保结果仍在合理的像素值范围内
    noisy_image = np.clip(noisy_image, 0, 255)
    
    # 将结果图像转换回原始的数据类型
    noisy_image = np.array(noisy_image, dtype=np.uint8)
    
    return noisy_image

def add_grain(image, grain_amount):
    # 生成与图像同样大小的噪声矩阵
    noise = torch.randn_like(image) * grain_amount

    # 将噪声添加到图像中
    noisy_image = image + noise

    # 确保结果仍在合理的像素值范围内
    noisy_image = torch.clamp(noisy_image, 0, 1) *255

    return noisy_image.to(image.device)