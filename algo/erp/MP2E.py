'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2026-04-21 15:08:09
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
import os
import sys
import cv2
import numpy as np
class Perspective:
    """将单张透视图像投影到等距柱状图 (ERP) 空间"""

    def __init__(self, img_name, FOV, THETA, PHI):
        """
        参数:
            img_name: 图像文件路径 或 numpy 数组 (H,W,3)
            FOV: 水平视场角 (度)
            THETA: 水平旋转角 (度, 绕 Y 轴)
            PHI: 垂直俯仰角 (度, 绕 X 轴)
        """
        if isinstance(img_name, str):
            self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            if self._img is None:
                raise FileNotFoundError(f"无法读取图像: {img_name}")
        else:
            self._img = img_name.copy()
        self._height, self._width, _ = self._img.shape
        self.wFOV = FOV
        self.THETA = THETA
        self.PHI = PHI
        self.hFOV = float(self._height) / self._width * FOV

        self.w_len = np.tan(np.radians(self.wFOV / 2.0))
        self.h_len = np.tan(np.radians(self.hFOV / 2.0))

    def GetEquirec(self, height, width):
        """
        生成 ERP 投影图像及映射关系
        返回:
            persp: ERP 图像 (H,W,3) uint8
            mask: 有效区域掩膜 (H,W,3) uint8 (0 或 255)
            lon_map: X 方向映射表 (H,W) float32
            lat_map: Y 方向映射表 (H,W) float32
        """
        # 构建球面坐标网格
        x, y = np.meshgrid(np.linspace(-180, 180, width), np.linspace(90, -90, height))
        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))
        xyz = np.stack((x_map, y_map, z_map), axis=2)  # (H,W,3)

        # 旋转矩阵: 先绕 Z 转 THETA，再绕 Y 转 -PHI
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        R1, _ = cv2.Rodrigues(z_axis * np.radians(self.THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-self.PHI))

        # 求逆旋转，将 ERP 坐标变换到透视相机坐标系
        R1_inv = np.linalg.inv(R1)
        R2_inv = np.linalg.inv(R2)

        xyz_flat = xyz.reshape(-1, 3).T  # (3, N)
        xyz_rot = np.dot(R2_inv, xyz_flat)
        xyz_rot = np.dot(R1_inv, xyz_rot).T  # (N, 3)
        xyz_rot = xyz_rot.reshape(height, width, 3)

        # 仅保留 Z > 0 的前方点
        inverse_mask = np.where(xyz_rot[:, :, 0] > 0, 1, 0).astype(np.float32)

        # 透视除法：除以 Z 得到归一化平面坐标
        xyz_norm = xyz_rot / np.maximum(xyz_rot[:, :, 0:1], 1e-8)

        # 计算在透视图像中的像素坐标
        u = (xyz_norm[:, :, 1] + self.w_len) / (2 * self.w_len) * self._width
        v = (-xyz_norm[:, :, 2] + self.h_len) / (2 * self.h_len) * self._height

        # 有效范围掩膜
        valid = (
            (-self.w_len <= xyz_norm[:, :, 1])
            & (xyz_norm[:, :, 1] <= self.w_len)
            & (-self.h_len <= xyz_norm[:, :, 2])
            & (xyz_norm[:, :, 2] <= self.h_len)
        ).astype(np.float32)

        # 无效区域置零，remap 时 borderMode 为 BORDER_CONSTANT (默认 0)
        lon_map = np.where(valid, u, 0).astype(np.float32)
        lat_map = np.where(valid, v, 0).astype(np.float32)

        # 重映射
        persp = cv2.remap(
            self._img,
            lon_map,
            lat_map,
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # 最终掩膜：结合方向与有效区域
        mask = valid * inverse_mask
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        persp = (persp * mask_3ch).astype(np.uint8)

        return persp, mask_3ch, lon_map, lat_map


class MPerspective:
    """多视角透视图像拼接为 ERP，同时转换光流"""

    def __init__(self, img_array, F_T_P_array):
        """
        参数:
            img_array: 列表，每个元素为 (img_path, mv0, mv1)
                - img_path: 图像文件路径 或 numpy 数组
                - mv0, mv1: 光流文件路径 (.flo) 或 numpy 数组 (H,W,2)
            F_T_P_array: 列表，每个元素为 (FOV, THETA, PHI)
        """
        assert len(img_array) == len(F_T_P_array), "数据长度不一致"
        self.img_array = img_array
        self.F_T_P_array = F_T_P_array

    @staticmethod
    def perspective_flow_to_erp(flow, lon_map, lat_map, erp_h, erp_w):
        """
        将透视图像上的光流重采样到 ERP 网格（未进行矢量变换）。
        注意：此方法仅对光流数值进行空间重采样，未考虑几何投影带来的方向与尺度变化。
        """
        if flow is None:
            return np.zeros((erp_h, erp_w, 2), dtype=np.float32)

        # 分别对光流的两个通道进行 remap
        flow_u = cv2.remap(
            flow[:, :, 0],
            lon_map,
            lat_map,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        flow_v = cv2.remap(
            flow[:, :, 1],
            lon_map,
            lat_map,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        erp_flow = np.stack([flow_u, flow_v], axis=-1)
        return erp_flow

    def GetEquirec(self, height, width):
        """
        融合所有视角生成 ERP 图像及平均光流
        返回:
            merge_image: 融合后的 ERP 图像 (H,W,3) uint8
            mask: 有效区域掩膜 (H,W,3) uint8
            merge_mv0: 融合后的前向光流 (H,W,2) float32
            merge_mv1: 融合后的后向光流 (H,W,2) float32
        """
        merge_image = np.zeros((height, width, 3), dtype=np.float32)
        merge_count = np.zeros((height, width), dtype=np.float32)  # 单通道计数

        merge_mv0 = np.zeros((height, width, 2), dtype=np.float32)
        merge_mv1 = np.zeros((height, width, 2), dtype=np.float32)

        for data, (F, T, P) in zip(self.img_array, self.F_T_P_array):
            img_input, flow0, flow1 = data

            # 创建透视投影对象
            per = Perspective(img_input, F, T, P)

            # 获取 ERP 图像、掩膜及映射表
            img_erp, mask_3ch, lon_map, lat_map = per.GetEquirec(height, width)
            mask_1ch = mask_3ch[:, :, 0]  # 单通道掩膜

            # 累加图像 (带权)
            merge_image += img_erp.astype(np.float32)
            merge_count += mask_1ch

            # 处理光流
            if flow0 is not None:
                mv0_erp = self.perspective_flow_to_erp(
                    flow0, lon_map, lat_map, height, width
                )
                # 加权累加
                merge_mv0 += mv0_erp * mask_1ch[:, :, np.newaxis]

            if flow1 is not None:
                mv1_erp = self.perspective_flow_to_erp(
                    flow1, lon_map, lat_map, height, width
                )
                merge_mv1 += mv1_erp * mask_1ch[:, :, np.newaxis]

        # 归一化图像
        merge_count_safe = np.where(merge_count == 0, 1, merge_count)
        merge_image = (merge_image / merge_count_safe[:, :, np.newaxis]).astype(np.uint8)

        # 归一化光流
        merge_mv0 = np.divide(
            merge_mv0,
            merge_count_safe[:, :, np.newaxis],
            out=np.zeros_like(merge_mv0),
            where=merge_count_safe[:, :, np.newaxis] != 0,
        )
        merge_mv1 = np.divide(
            merge_mv1,
            merge_count_safe[:, :, np.newaxis],
            out=np.zeros_like(merge_mv1),
            where=merge_count_safe[:, :, np.newaxis] != 0,
        )

        # 生成最终掩膜 (0/255)
        mask_final = np.where(merge_count > 0, 255, 0).astype(np.uint8)
        mask_final_3ch = np.repeat(mask_final[:, :, np.newaxis], 3, axis=2)

        return merge_image, mask_final_3ch, merge_mv0, merge_mv1


        