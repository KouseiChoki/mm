import cv2
import numpy as np


# =========================================================
# Perspective Projection
# =========================================================
class Perspective:
    def __init__(self, img, FOV, THETA, PHI):

        self._img = cv2.imread(img) if isinstance(img, str) else img.copy()
        self.H, self.W = self._img.shape[:2]

        self.wFOV = FOV
        self.THETA = THETA
        self.PHI = PHI

        self.w_len = np.tan(np.radians(FOV / 2.0))

        # camera intrinsics (pinhole approx)
        self.focal = self.W / (2 * self.w_len)
        self.cx = self.W / 2
        self.cy = self.H / 2

        # rotation
        y_axis = np.array([0, 1, 0], np.float32)
        z_axis = np.array([0, 0, 1], np.float32)

        R1, _ = cv2.Rodrigues(z_axis * np.radians(self.THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-self.PHI))

        self.R1_inv = np.linalg.inv(R1)
        self.R2_inv = np.linalg.inv(R2)

        # world -> cam
        self.R = self.R1_inv @ self.R2_inv


# =========================================================
# ERP <-> Ray
# =========================================================
def erp_to_ray(u, v, H, W):
    lon = (u / W) * 2 * np.pi - np.pi
    lat = np.pi / 2 - (v / H) * np.pi

    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)

    return np.stack([x, y, z], axis=-1)


def ray_to_erp(ray, H, W):
    x, y, z = ray[..., 0], ray[..., 1], ray[..., 2]

    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z, -1, 1))

    u = (lon + np.pi) / (2 * np.pi) * W
    v = (np.pi / 2 - lat) / np.pi * H

    return np.stack([u, v], axis=-1)


# =========================================================
# Core Flow Conversion (C++一致版本)
# =========================================================
class MPerspective:

    def __init__(self, img_array, F_T_P_array):
        self.img_array = img_array
        self.F_T_P_array = F_T_P_array

    # -----------------------------------------------------
    # STRICT C++ EQUIVALENT FLOW CONVERSION
    # -----------------------------------------------------
    def perspective_flow_to_erp(self, flow, persp: Perspective, H, W):

        Hp, Wp = persp.H, persp.W

        # -------------------------
        # 1. ERP grid
        # -------------------------
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))

        # -------------------------
        # 2. ERP -> world ray (vectorized)
        # -------------------------
        ray = erp_to_ray(uu, vv, H, W)   # (H,W,3)

        # world -> camera
        ray_c = ray @ persp.R.T

        z = ray_c[..., 2:3]
        valid = z > 1e-6

        z_safe = np.maximum(z, 1e-6)

        # -------------------------
        # 3. pinhole projection
        # -------------------------
        uc = (ray_c[..., 0] / z_safe[..., 0]) * persp.focal + persp.cx
        vc = (ray_c[..., 1] / z_safe[..., 0]) * persp.focal + persp.cy

        # -------------------------
        # 4. sample flow (FAST via remap)
        # -------------------------
        map_x = uc.astype(np.float32)
        map_y = vc.astype(np.float32)

        flow_u = cv2.remap(flow[..., 0].astype(np.float32),
                        map_x, map_y,
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0)

        flow_v = cv2.remap(flow[..., 1].astype(np.float32),
                        map_x, map_y,
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0)

        # -------------------------
        # 5. warp in image space
        # -------------------------
        uc2 = uc + flow_u
        vc2 = vc + flow_v

        # -------------------------
        # 6. back to ray (vectorized)
        # -------------------------
        x2 = (uc2 - persp.cx) / persp.focal
        y2 = (vc2 - persp.cy) / persp.focal
        z2 = np.ones_like(x2)

        ray_c2 = np.stack([x2, y2, z2], axis=-1)
        ray_c2 /= np.linalg.norm(ray_c2, axis=-1, keepdims=True) + 1e-8

        # camera -> world
        ray_w2 = ray_c2 @ persp.R

        ray_w2 /= np.linalg.norm(ray_w2, axis=-1, keepdims=True) + 1e-8

        # -------------------------
        # 7. ERP projection (vectorized)
        # -------------------------
        x, y, z = ray_w2[..., 0], ray_w2[..., 1], ray_w2[..., 2]

        lon = np.arctan2(y, x)
        lat = np.arcsin(np.clip(z, -1, 1))

        u2 = (lon + np.pi) / (2 * np.pi) * W
        v2 = (np.pi / 2 - lat) / np.pi * H

        # -------------------------
        # 8. MV
        # -------------------------
        mv = np.stack([u2 - uu, v2 - vv], axis=-1)

        mv[~valid[..., 0]] = 0

        return mv.astype(np.float32)


    # -----------------------------------------------------
    # Multi-view fusion
    # -----------------------------------------------------
    def run(self, H, W):

        mv0 = np.zeros((H, W, 2), np.float32)
        mv1 = np.zeros((H, W, 2), np.float32)

        img_acc = np.zeros((H, W, 3), np.float32)
        cnt = np.zeros((H, W), np.float32)

        for data, (F, T, P) in zip(self.img_array, self.F_T_P_array):

            img, flow0, flow1 = data

            persp = Perspective(img, F, T, P)

            # image projection (simplified)
            img_acc += np.zeros((H, W, 3), np.float32)
            cnt += 1

            if flow0 is not None:
                mv0 += self.perspective_flow_to_erp(flow0, persp, H, W)

            if flow1 is not None:
                mv1 += self.perspective_flow_to_erp(flow1, persp, H, W)

        cnt = np.maximum(cnt, 1)

        img_out = (img_acc / cnt[:, :, None]).astype(np.uint8)

        mv0 /= len(self.img_array)
        mv1 /= len(self.img_array)

        mask = (cnt > 0).astype(np.uint8) * 255
        mask = np.repeat(mask[:, :, None], 3, axis=2)

        return img_out, mask, mv0, mv1