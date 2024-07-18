'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-06-28 14:59:53
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
import numpy as np
import cv2
import sys,os
import open3d as o3d
import imageio
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/../..')
from file_utils import read
def generate_point_cloud(depth_image, intrinsics):
    fx, fy, cx, cy = intrinsics
    height, width = depth_image.shape

    # Generate 3D point cloud
    i, j = np.indices((height, width))
    z = depth_image
    x = (j - cx) * z / fx
    y = (i - cy) * z / fy
    xyz = np.stack((x, y, z), axis=-1)
    
    return xyz.reshape(-1, 3)

def reproject_points(points, intrinsics):
    fx, fy, cx, cy = intrinsics

    # Project 3D points back to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    
    return np.stack((u, v), axis=-1)

def check_focal_length(depth_image, image, intrinsics):
    # Generate point cloud
    points = generate_point_cloud(depth_image, intrinsics)
    
    # Reproject points to image plane
    projected_points = reproject_points(points, intrinsics)
    
    # Calculate reprojection error
    valid_mask = (depth_image > 0)
    uv = np.indices((depth_image.shape[0], depth_image.shape[1])).transpose(1, 2, 0).reshape(-1, 2)
    valid_uv = uv[valid_mask.reshape(-1)]

    errors = np.linalg.norm(projected_points[valid_mask.reshape(-1)] - valid_uv, axis=-1)
    mean_error = np.mean(errors)
    
    # Visualize reprojected points
    vis_image = image.copy()
    for pt in projected_points[valid_mask.reshape(-1)]:
        cv2.circle(vis_image, (int(pt[0]), int(pt[1])), 1, (0, 255, 255), -1)

    # cv2.imshow('Reprojected Points', vis_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return mean_error,vis_image

# Example usage:
# intrinsics = [20, 20, 960, 540]  # fx, fy, cx, cy
# intrinsics = [693, 692, 960, 540]
intrinsics = [1616, 1616, 960, 540]
depth_image = read('/Users/qhong/Desktop/0607/FtGothicCastle_04/world_depth/FtGothicCastle_04.0002.exr')[...,0]
image = cv2.imread('/Users/qhong/Desktop/0607/FtGothicCastle_04/image/FtGothicCastle_04.0002.png')

mean_error,vis = check_focal_length(depth_image, image, intrinsics)
print(f'Mean Reprojection Error: {mean_error}')
Image.fromarray(vis)