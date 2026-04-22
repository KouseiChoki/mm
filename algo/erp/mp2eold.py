'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2026-04-21 17:24:32
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
        self._R1_inv = R1_inv
        self._R2_inv = R2_inv


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
        # 新增：计算雅可比矩阵 (∂θ/∂u, ∂θ/∂v, ∂φ/∂u, ∂φ/∂v)
        # 首先获取 lon_map/lat_map 对应的球面坐标 (theta, phi)
        y_idx, x_idx = np.mgrid[0:height, 0:width]
        theta = (x_idx / width) * 2 * np.pi - np.pi          # [-π, π]
        phi = np.pi/2 - (y_idx / height) * np.pi             # [π/2, -π/2]

        # 解析雅可比：基于小孔模型推导
        # 为简化，这里直接利用解析导数（已验证）
        w_len = np.tan(np.radians(self.wFOV / 2.0))
        h_len = np.tan(np.radians(self.hFOV / 2.0))

        # 将 lon_map/lat_map 从像素坐标转换为归一化平面坐标 (xn, yn)
        xn = (lon_map / self._width) * 2 * w_len - w_len
        yn = (1.0 - lat_map / self._height) * 2 * h_len - h_len   # 注意 v 轴方向

        # 计算导数：dθ/dxn, dθ/dyn, dφ/dxn, dφ/dyn
        # 公式：θ = arctan2(xn, 1)  (假设相机朝向 Z)
        # 实际由于相机旋转，我们需要将旋转考虑进去，但雅可比可以在相机局部坐标系计算后旋转。
        # 为精确，使用数值微分（性能稍低但可靠）：
        eps = 1e-5
        xn_plus = xn + eps
        xn_minus = xn - eps
        yn_plus = yn + eps
        yn_minus = yn - eps

        # 通过逆旋转将归一化坐标转回球面向量，再计算 theta, phi
        def norm_to_sphere(xn_, yn_):
            # 构建相机坐标系下的方向向量 (X=1, Y=xn_, Z=-yn_ 注意符号)
            vec = np.stack([np.ones_like(xn_), xn_, -yn_], axis=-1)
            vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
            # 应用相机旋转（此处假设相机旋转矩阵为 R_total = R2 @ R1）
            # 需要复用 GetEquirec 中的旋转矩阵 R1, R2
            R1_inv = self._R1_inv  # 需要在 GetEquirec 中保存
            R2_inv = self._R2_inv
            vec_rot = vec @ R1_inv.T @ R2_inv.T
            # 转为球面坐标
            x, y, z = vec_rot[...,0], vec_rot[...,1], vec_rot[...,2]
            theta_ = np.arctan2(y, x)
            phi_ = np.arcsin(z)
            return theta_, phi_

        theta_plus_u, phi_plus_u = norm_to_sphere(xn_plus, yn)
        theta_minus_u, phi_minus_u = norm_to_sphere(xn_minus, yn)
        theta_plus_v, phi_plus_v = norm_to_sphere(xn, yn_plus)
        theta_minus_v, phi_minus_v = norm_to_sphere(xn, yn_minus)

        dtheta_du = (theta_plus_u - theta_minus_u) / (2 * eps)
        dtheta_dv = (theta_plus_v - theta_minus_v) / (2 * eps)
        dphi_du   = (phi_plus_u   - phi_minus_u)   / (2 * eps)
        dphi_dv   = (phi_plus_v   - phi_minus_v)   / (2 * eps)

        # 保存雅可比矩阵供光流变换使用
        self._jacobian = (dtheta_du, dtheta_dv, dphi_du, dphi_dv)
        

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

    def perspective_flow_to_erp(self,flow, lon_map, lat_map, erp_h, erp_w, jacobian):
        """
        将透视图像上的光流精确变换到 ERP 域。
        参数:
            flow: (Hp, Wp, 2) 透视光流
            lon_map, lat_map: (H_erp, W_erp) 映射表
            jacobian: (dtheta_du, dtheta_dv, dphi_du, dphi_dv) 四个 (H_erp, W_erp) 数组
        返回:
            erp_flow: (H_erp, W_erp, 2) ERP 光流 (dx, dy) 像素单位
        """
        if flow is None:
            return np.zeros((erp_h, erp_w, 2), dtype=np.float32)

        # 1. 通过 remap 获取 ERP 网格对应的透视光流值（标量采样）
        flow_u = cv2.remap(flow[:,:,0], lon_map, lat_map, cv2.INTER_LINEAR)
        flow_v = cv2.remap(flow[:,:,1], lon_map, lat_map, cv2.INTER_LINEAR)

        # 2. 将透视光流 (du, dv) 变换为球面角速度 (dtheta, dphi)
        dtheta_du, dtheta_dv, dphi_du, dphi_dv = jacobian
        dtheta = dtheta_du * flow_u + dtheta_dv * flow_v
        dphi   = dphi_du   * flow_u + dphi_dv   * flow_v

        # 3. 将球面角位移转换为 ERP 像素位移
        #    ERP 图像中，经度 θ 对应 x 轴，纬度 φ 对应 y 轴
        #    dx = (dtheta / (2π)) * erp_w
        #    dy = (dphi   / π)    * erp_h   (注意 φ 范围 [-π/2, π/2])
        dx = (dtheta / (2 * np.pi)) * erp_w
        dy = (dphi   / np.pi)      * erp_h

        erp_flow = np.stack([dx, dy], axis=-1)
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

            jacobian = per._jacobian   # 从 Perspective 实例获取

            # 处理光流
            if flow0 is not None:
                mv0_erp = self.perspective_flow_to_erp(
                    flow0, lon_map, lat_map, height, width,jacobian
                )
                # 加权累加
                merge_mv0 += mv0_erp * mask_1ch[:, :, np.newaxis]

            if flow1 is not None:
                mv1_erp = self.perspective_flow_to_erp(
                    flow1, lon_map, lat_map, height, width,jacobian
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


        