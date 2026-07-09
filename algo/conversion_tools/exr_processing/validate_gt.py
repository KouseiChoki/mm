'''
MV 约定验证脚本
================
约定 (待验证):
  mv1@帧N = 帧N → 上一帧(N-1) 的位移    (即 VFI 监督里 gt→img0 的 F_t→0)
  mv0@帧N = 帧N → 下一帧(N+1) 的位移    (即 VFI 监督里 gt→img1 的 F_t→1)
  存储为归一化值 (x已除以W, y已除以H), 屏幕坐标右下为正

检验原理 (backward warp 金标准):
  out(x) = src(x + F(x));  flow定义在哪一帧, 重建出的就是哪一帧,
  采样源必须是 flow 指向的那一帧:
    backward_warp(帧N-1, mv1@N)  应重建出  帧N
    backward_warp(帧N+1, mv0@N)  应重建出  帧N

脚本对 (sx, sy) ∈ {+1,-1}² 四种符号组合全部测试, 输出 PSNR 表,
最高者即数据的真实符号约定。合成数据下正确组合应显著高于其余
(通常 30dB+, 其余组合 10~20dB)。

用法: 修改下方三个路径后直接运行。只需要两张相邻帧即可同时验证 mv1 与 mv0。
'''
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')
from file_utils import read, write

# ═════════════════ 配置区: 两张相邻帧 (N-1, N) ═════════════════
FRAME_PREV = '/Volumes/Spica/optical_flow/Unreal/clean044_Slay_19/24fps/image/TF0200_01.1019.png'   # 帧 N-1
FRAME_CUR  = '/Volumes/Spica/optical_flow/Unreal/clean044_Slay_19/24fps/image/TF0200_01.1020.png'   # 帧 N
OUT_DIR    = '/Users/qhong/Desktop/0709'
# ═══════════════════════════════════════════════════════════════

# mv 路径按你的目录结构自动推导:
#   mv1@N   : 帧N 的 mv1  (N→N-1),  用 帧N-1 做源, 重建 帧N
#   mv0@N-1 : 帧N-1 的 mv0 (N-1→N), 用 帧N 做源,  重建 帧N-1
MV1_AT_CUR  = FRAME_CUR.replace('.png', '.exr').replace('/image/', '/mv1/')
MV0_AT_PREV = FRAME_PREV.replace('.png', '.exr').replace('/image/', '/mv0/')


def backward_warp(img_src, flow_px):
    """out(x) = img_src(x + flow_px(x)); flow为像素位移, 右下为正。"""
    h, w = flow_px.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(img_src, gx + flow_px[..., 0], gy + flow_px[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def psnr_center(a, b, margin=16):
    """PSNR, 去掉边缘 margin 像素 (排除 border padding 采样误差)。"""
    a_ = a[margin:-margin, margin:-margin].astype(np.float32)
    b_ = b[margin:-margin, margin:-margin].astype(np.float32)
    mse = np.mean((a_ - b_) ** 2)
    return 99.0 if mse < 1e-10 else 10 * np.log10(255.0 ** 2 / mse)


def sweep_signs(mv_norm, img_src, img_target, w, h, tag, out_dir):
    """四种符号组合全测, 返回 [(psnr, sx, sy, rec)] 按 PSNR 降序。"""
    results = []
    for sx in (1, -1):
        for sy in (1, -1):
            mv = mv_norm.copy()
            mv[..., 0] *= sx * w      # 反归一化 + 符号
            mv[..., 1] *= sy * h
            rec = backward_warp(img_src, mv)
            p = psnr_center(rec, img_target)
            results.append((p, sx, sy, rec))
    results.sort(key=lambda r: -r[0])

    print(f'\n[{tag}]')
    print(f'  {"sx":>4} {"sy":>4} {"PSNR":>8}')
    for p, sx, sy, _ in results:
        mark = '  ← 最优' if (p, sx, sy) == (results[0][0], results[0][1], results[0][2]) else ''
        print(f'  {sx:>4} {sy:>4} {p:>8.2f}{mark}')

    best_p, best_sx, best_sy, best_rec = results[0]
    write(os.path.join(out_dir, f'{tag}_best_rec.png'), best_rec)
    return best_p, best_sx, best_sy


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frame_prev = read(FRAME_PREV, type='image')
    frame_cur = read(FRAME_CUR, type='image')
    h, w = frame_cur.shape[:2]

    write(os.path.join(OUT_DIR, 'frame_prev.png'), frame_prev)
    write(os.path.join(OUT_DIR, 'frame_cur.png'), frame_cur)

    verdicts = []

    # ── 测试1: mv1@N (N→N-1) — 源=帧N-1, 目标=帧N ─────────────────────────
    if os.path.isfile(MV1_AT_CUR):
        mv1 = read(MV1_AT_CUR, type='flo')[..., :2].astype(np.float32)
        p, sx, sy = sweep_signs(mv1, img_src=frame_prev, img_target=frame_cur,
                                w=w, h=h, tag='mv1', out_dir=OUT_DIR)
        verdicts.append(('mv1', p, sx, sy))
    else:
        print(f'mv1 文件不存在, 跳过: {MV1_AT_CUR}')

    # ── 测试2: mv0@N-1 (N-1→N) — 源=帧N, 目标=帧N-1 ───────────────────────
    if os.path.isfile(MV0_AT_PREV):
        mv0 = read(MV0_AT_PREV, type='flo')[..., :2].astype(np.float32)
        p, sx, sy = sweep_signs(mv0, img_src=frame_cur, img_target=frame_prev,
                                w=w, h=h, tag='mv0', out_dir=OUT_DIR)
        verdicts.append(('mv0', p, sx, sy))
    else:
        print(f'mv0 文件不存在, 跳过: {MV0_AT_PREV}')

    # ── 结论 ───────────────────────────────────────────────────────────────
    print('\n════════ 结论 ════════')
    for tag, p, sx, sy in verdicts:
        status = '✓ 约定正确, 可直接做监督' if p >= 30 else \
                 ('△ 勉强, 检查遮挡/运动大小后复测' if p >= 24 else '✗ 异常, 检查配对帧或数据')
        conv = '标准右下(+,+)' if (sx, sy) == (1, 1) else f'需符号修正 (x*{sx}, y*{sy})'
        print(f'  {tag}: 最优PSNR={p:.2f}dB  符号=({sx:+d},{sy:+d}) → {conv}  {status}')
    if len(verdicts) == 2:
        s1 = (verdicts[0][2], verdicts[0][3])
        s2 = (verdicts[1][2], verdicts[1][3])
        if s1 != s2:
            print('  ⚠ mv0 与 mv1 的最优符号不一致! 说明两者存储约定不同, 训练前必须统一。')


if __name__ == '__main__':
    main()