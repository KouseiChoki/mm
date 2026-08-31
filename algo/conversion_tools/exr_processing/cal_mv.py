'''
Unreal EXR → image / mv 提取脚本 (精简版: 仅保留原 mode 1 相机MV计算)
======================================================================
从 Unreal dump 的多层 exr 中提取:
  - image/        显示域 LDR png (或 --exrformat 时为线性 float32 exr)
  - mv0/ mv1/     光流 (相机运动 camera_tracking + 物体运动 objmv 合成, 归一化, half精度)
  - Mask/         物体mask (若exr中存在)
  - world_depth/  世界深度 (--dump_depth / --colormap)
  - ACESCG/ rec709/  HDR原色域输出 (--enable_colour_output)

用法:
  python exr_get_mv.py --path /data/dump --core 8
  python exr_get_mv.py --path /data/dump --core 8 --exposure 2.0   # 暗场景提亮

变更记录 (相对原版):
  - 删除 mode 2~6 (tonemapping / MRQ / trans / inverse / check) 及其专属函数与参数
  - 修复 --debug 分支在 data0 赋值前引用导致的 NameError
  - 修复 exr 缺少部分 MV 通道时 fnmv*_G/B/A 未定义导致的 NameError
  - Color_transform / unreal_ply 改为按需惰性加载 (无相关依赖时其余功能可用)
  - 清理重复 import 与死代码 (trans_json / get_channel_data / exr_read_rgb 等)
  - [本次] LDR 输出改为与 Unreal 编辑器观感一致:
      裸 gamma2.4  →  ACES filmic tone map (Hill RRT+ODT 拟合, 即UE Filmic Tonemapper
      的基础) + 标准 sRGB OETF; 新增 --exposure 曝光增益 (UE的auto exposure离线
      无法复现, 用固定值近似, 暗场景可调 2~4)
  - [本次] acescg 源的 LDR 转换不再依赖 --enable_colour_output 开关:
      无条件先做 AP1→rec709 色域转换再进 tone map (修复漏转导致的去饱和/色偏)
  - [本次] --exrformat 直接保存线性 float32 RGB, 不再经过 gamma/tone map/clip;
      负值会原样保留, 不会因分数次幂产生 NaN
  - [本次] 修复 --objmvonly: 开启后 mv0/mv1 都直接使用包含完整运动的
      obj MV，不再计算或混入 camera MV；增加等价别名 --obj-mv-only
'''

import os
import sys
import re
import shutil
import argparse
import warnings

import numpy as np
import cv2
import OpenEXR
import Imath
import array
from tqdm.contrib.concurrent import process_map

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')
from file_utils import mvwrite, read

warnings.filterwarnings("ignore")

pt = Imath.PixelType(Imath.PixelType.FLOAT)
D2R = np.pi / 180

# Unreal EXR 名称对照表。Unreal 升级或 dump 插件改名后，只需修改此表。
# 每项按优先级排列：新版名在前，旧版名在后，以同时兼容新旧数据。
UNREAL_EXR_NAME_MAP = {
    'headers': {
        'color_space_destination': (
            'unreal/colorSpace/destination',
        ),
        'focal_length': (
            'unreal/layerData/rgba/focalLength',
            'unreal/camera/FinalImage/focalLength',
        ),
        'sensor_width': (
            'unreal/layerData/rgba/sensorWidth',
            'unreal/camera/FinalImage/sensorWidth',
        ),
        'fov': (
            'unreal/layerData/rgba/fov',
            'unreal/camera/FinalImage/fov',
        ),
        'cur_pos_x': (
            'unreal/layerData/rgba/curPos/x', 'unreal/camera/curPos/x'),
        'cur_pos_y': (
            'unreal/layerData/rgba/curPos/y', 'unreal/camera/curPos/y'),
        'cur_pos_z': (
            'unreal/layerData/rgba/curPos/z', 'unreal/camera/curPos/z'),
        'cur_rot_pitch': (
            'unreal/layerData/rgba/curRot/pitch', 'unreal/camera/curRot/pitch'),
        'cur_rot_roll': (
            'unreal/layerData/rgba/curRot/roll', 'unreal/camera/curRot/roll'),
        'cur_rot_yaw': (
            'unreal/layerData/rgba/curRot/yaw', 'unreal/camera/curRot/yaw'),
        'prev_pos_x': (
            'unreal/layerData/rgba/prevPos/x', 'unreal/camera/prevPos/x'),
        'prev_pos_y': (
            'unreal/layerData/rgba/prevPos/y', 'unreal/camera/prevPos/y'),
        'prev_pos_z': (
            'unreal/layerData/rgba/prevPos/z', 'unreal/camera/prevPos/z'),
        'prev_rot_pitch': (
            'unreal/layerData/rgba/prevRot/pitch', 'unreal/camera/prevRot/pitch'),
        'prev_rot_roll': (
            'unreal/layerData/rgba/prevRot/roll', 'unreal/camera/prevRot/roll'),
        'prev_rot_yaw': (
            'unreal/layerData/rgba/prevRot/yaw', 'unreal/camera/prevRot/yaw'),
    },
    # 通道名使用子串匹配，允许 EXR 在前面增加 layer 前缀。
    'channels': {
        'depth_r': (
            'FinalImagePWWorldDepth.R',
            'FinalImageMovieRenderQueue_WorldDepth.R',
            'ImageDepth.R',
        ),
        'mv1_r': ('MV1.R', 'MotionVectors.R'),
        'mv1_g': ('MV1.G', 'MotionVectors.G'),
        'mv1_b': ('MV1.B', 'MotionVectors.B'),
        'mv1_a': ('MV1.A', 'MotionVectors.A'),
        'mv0_r': ('MV0.R',),
        'mv0_g': ('MV0.G',),
        'mv0_b': ('MV0.B',),
        'mv0_a': ('MV0.A',),
        'mask_r': ('ObjMask.R', 'PWMask.R'),
    },
}

# ── 色域基色坐标: 用于从 chromaticities 头识别色彩空间 ──────────────────────
type_dict = {
    "PW_PRM_BT601":       [0.640, 0.330, 0.290, 0.600, 0.150, 0.060],
    "rec709":             [0.640, 0.330, 0.300, 0.600, 0.150, 0.060],
    "PW_PRM_DCI_P3":      [0.680, 0.320, 0.265, 0.690, 0.150, 0.060],
    "PW_PRM_BT2020":      [0.708, 0.292, 0.170, 0.797, 0.131, 0.046],
    "PW_PRM_ARRI_WG":     [0.684, 0.313, 0.211, 0.848, 0.0861, -0.102],
    "PW_PRM_ACES_AP0":    [0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.0770],
    "PW_PRM_ACES_AP1":    [0.713, 0.293, 0.165, 0.83, 0.128, 0.0440],
    "PW_PRM_CINITY":      [0.705, 0.2872, 0.1205, 0.8029, 0.1557, 0.0288],
    "PW_PRM_GAMUT3":      [0.730, 0.280, 0.140, 0.855, 0.100, -0.050],
    "PW_PRM_GAMUT3_CINE": [0.766, 0.275, 0.225, 0.800, 0.089, -0.087],
    "PW_PRM_UNSPECIFIED": [0.708, 0.292, 0.170, 0.797, 0.131, 0.046],
}

# ── 色彩变换: 惰性初始化 (仅 acescg 源或 --enable_colour_output 时需要) ─────
_color_transforms = None


def get_color_transforms():
    global _color_transforms
    if _color_transforms is None:
        from color_convertion.colorutil import Color_transform
        _color_transforms = (Color_transform('lin_rec709', 'acescg'),
                             Color_transform('acescg', 'lin_rec709'))
    return _color_transforms


# ─────────────────────────────────────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────────────────────────────────────

def init_param():
    parser = argparse.ArgumentParser(description='Unreal exr 提取 image / mv (相机MV计算)')
    parser.add_argument('--path', '--root', dest='path', required=True,
                        help='dump 数据根目录')
    parser.add_argument('--output', help='输出根目录 (默认原地输出到各序列同级目录)')
    parser.add_argument('--extra_depth', help='用外部深度目录代替 exr 内深度计算 mv')
    mv_source = parser.add_mutually_exclusive_group()
    mv_source.add_argument(
        '--objmvonly', '--obj-mv-only', dest='objmvonly', action='store_true',
        help='直接使用EXR中的完整obj MV，不计算或混入camera MV')
    parser.add_argument('--onlymv', action='store_false',
                        help='加此参数则跳过 image 输出 (默认输出 image)')
    parser.add_argument('--debug', action='store_true', help='校验相邻帧相机数据一致性')
    parser.add_argument('--dump_depth', action='store_true', help='输出世界深度')
    parser.add_argument('--dump_ply', action='store_true', help='输出点云 ply')
    parser.add_argument('--down_scale', type=int, default=1, help='ply 降采样倍率')
    parser.add_argument('--depth_only', action='store_true', help='只输出深度, 跳过 mv')
    parser.add_argument('--colormap', action='store_true', help='深度以 colormap png 输出')
    parser.add_argument('--ACESCG', action='store_true',
                        help='(配合 --enable_colour_output) rec709 源额外输出 acescg')
    parser.add_argument('--f', action='store_true', help='强制重跑, 忽略已有输出')
    mv_source.add_argument(
        '--bg_mode', '--bg-mode', dest='bg_mode', action='store_true',
        help='只算相机(背景)运动, 忽略objmv')
    parser.add_argument('--step', type=int, default=1,
                        help='帧间隔; !=1 时只输出 mv{2(step-1)}/mv{2(step-1)+1}')
    parser.add_argument('--core', type=int, default=1, help='并行进程数, 0=单进程调试')
    parser.add_argument('--exrformat', action='store_true',
                        help='image 以线性 float32 exr 输出, 保留负值且不做 tone map')
    parser.add_argument('--enable_colour_output', action='store_true',
                        help='输出 HDR 原色域 (ACESCG/ 或 rec709/)')
    parser.add_argument('--mask_only', action='store_true', help='只输出 Mask')
    parser.add_argument('--exposure', type=float, default=1.0,
                        help='PNG tone map 前的曝光增益; EXR 输出不应用; '
                             'UE的auto exposure离线不可得, '
                             '暗场景建议 2~4 (默认 1.0, 全数据集统一, 不要逐帧调)')
    args = parser.parse_args()
    if args.depth_only or args.colormap:
        args.dump_depth = True
    return args


# ─────────────────────────────────────────────────────────────────────────────
# 目录/文件工具
# ─────────────────────────────────────────────────────────────────────────────

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def extract_number(file_path):
    number = re.findall(r'\d+', os.path.basename(file_path))
    return int(number[0]) if number else 0


def extract_last_4_digits(path):
    match = re.search(r'(\d+)\.exr$', path)
    return int(match.group(1)) if match else 0


def jhelp(c, restrict=True):
    key = extract_number if restrict else None
    items = sorted(filter(lambda x: x[0] != '.', os.listdir(c)), key=key) if restrict \
        else sorted(filter(lambda x: x[0] != '.', os.listdir(c)))
    return [os.path.join(c, i) for i in items]


def jhelp_folder(c, restrict=True):
    return [x for x in jhelp(c, restrict) if os.path.isdir(x)]


def jhelp_file(c, restrict=True):
    return [x for x in jhelp(c, restrict) if not os.path.isdir(x)]


def prune(c, keyword, mode='basename'):
    if mode == 'basename':
        return [x for x in c if keyword.lower() not in os.path.basename(x).lower()]
    return [x for x in c if keyword.lower() not in x.lower()]


def rename(source, target):
    try:
        os.rename(source, target)
        print(f"文件夹名已从 '{source}' 改为 '{target}'")
    except OSError as error:
        print(f"更改文件夹名时发生错误: {error}")


def restore_file_name(root):
    for file in jhelp_folder(root):
        base = os.path.basename(file)
        if base.lower() == 'orinal':
            rename(file, os.path.join(os.path.dirname(file), 'ori'))
        elif base == '12':
            rename(file, os.path.join(os.path.dirname(file), '12fps'))
        elif base == '24':
            rename(file, os.path.join(os.path.dirname(file), '24fps'))
        elif base == '48':
            rename(file, os.path.join(os.path.dirname(file), '48fps'))
        elif base == 'mask':
            rename(file, os.path.join(os.path.dirname(file), 'Mask'))


def loop_helper(root, key='.exr'):
    """递归查找 root 下所有直接包含 exr 文件的文件夹。"""
    exr_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any(fname.lower().endswith(key) for fname in filenames):
            exr_dirs.append(dirpath)
    return exr_dirs


# ─────────────────────────────────────────────────────────────────────────────
# EXR 读取
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_header_key(header, logical_name, required=True):
    """根据对照表解析 Unreal header 键。

    缺少必需字段时，报错会指出要修改的对照表项和 EXR 中实际存在的键。
    """
    candidates = UNREAL_EXR_NAME_MAP['headers'][logical_name]
    for key in candidates:
        if key in header:
            return key
    if not required:
        return None

    available = sorted(str(key) for key in header if str(key).startswith('unreal/'))
    available_text = ', '.join(available) if available else '<无 unreal/* header>'
    raise KeyError(
        f"EXR header 缺少逻辑字段 '{logical_name}'。"
        f"请修改 UNREAL_EXR_NAME_MAP['headers']['{logical_name}']。"
        f"当前候选: {candidates}; EXR 实际键: {available_text}"
    )


def _read_header_float(header, logical_name, required=True):
    key = _resolve_header_key(header, logical_name, required=required)
    return float(header[key]) if key is not None else None


def _find_channel_name(channels, logical_name, required=False):
    """根据对照表查找真实通道名；候选名按子串匹配。"""
    candidates = UNREAL_EXR_NAME_MAP['channels'][logical_name]
    for candidate in candidates:
        for channel_name in channels:
            if candidate in channel_name:
                return channel_name
    if not required:
        return None
    raise KeyError(
        f"EXR channel 缺少逻辑通道 '{logical_name}'。"
        f"请修改 UNREAL_EXR_NAME_MAP['channels']['{logical_name}']。"
        f"当前候选: {candidates}; EXR 实际通道: {sorted(channels)}"
    )


def check_type(exr):
    """识别色彩空间, 目前仅区分 rec709 / acescg。"""
    dtype = 'rec709'
    if 'chromaticities' in exr.header():
        chro = exr.header()['chromaticities']
        check = [chro.red.x, chro.red.y, chro.green.x, chro.green.y,
                 chro.blue.x, chro.blue.y]
        check_two = [round(i, 3) for i in check]
        loss = 1e9
        for item, key in type_dict.items():
            loss_ = sum(abs(a - b) for a, b in zip(check_two, key))
            if loss_ < loss:
                loss = loss_
                dtype = item
    else:
        color_space_key = _resolve_header_key(
            exr.header(), 'color_space_destination', required=False)
        if (color_space_key is not None
                and 'acescg' in str(exr.header()[color_space_key]).lower()):
            dtype = 'acescg'
    if 'pw_prm' in dtype.lower():
        dtype = 'acescg'
    assert dtype.lower() in ['rec709', 'acescg'], f'not supported algorithm {dtype}'
    return dtype


def _read_camera_header(img_exr, size):
    """从 exr header 读取相机公共参数, 返回基础 data dict。"""
    header = img_exr.header()
    data = {}
    focal_length = _read_header_float(header, 'focal_length', required=False)
    if focal_length is not None:
        data['focal_length'] = focal_length
    data['sensor_w'] = _read_header_float(header, 'sensor_width')
    data['fov'] = _read_header_float(header, 'fov')
    data['camera_type'] = 0 if focal_length is not None else 1
    data['h'], data['w'] = size
    return data


def _read_mv_channels(img_exr, size, ch_r, ch_g, ch_b, ch_a):
    """按通道名读取并反归一化一组 MV。任一通道缺失返回 None。"""
    if None in (ch_r, ch_g, ch_b, ch_a):
        return None
    mv_x = np.array(array.array('f', img_exr.channel(ch_r, pt))).reshape(size)
    mv_y = np.array(array.array('f', img_exr.channel(ch_g, pt))).reshape(size)
    mv_z = np.array(array.array('f', img_exr.channel(ch_b, pt))).reshape(size)
    mv_a = np.array(array.array('f', img_exr.channel(ch_a, pt))).reshape(size)
    mv_x = (mv_x - 0.5) * 2 * size[1] * -1
    mv_y = (mv_y - 0.5) * 2 * size[0]
    return np.stack([mv_x, mv_y, mv_z, mv_a], axis=2)


def exr_read_worldpos(filePath):
    """读取一帧 dump exr: 返回 (rgb, depth, prev相机, cur相机, mv0, mv1, mask, 色彩空间)。"""
    img_exr = OpenEXR.InputFile(filePath)
    header = img_exr.header()
    dtype = check_type(img_exr)
    dw = header['dataWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    # ── 通道名解析 (初始化为 None, 缺失通道不再触发 NameError) ────────────
    channels = header['channels']
    depth_R = _find_channel_name(channels, 'depth_r')
    mv1_R = _find_channel_name(channels, 'mv1_r')
    mv1_G = _find_channel_name(channels, 'mv1_g')
    mv1_B = _find_channel_name(channels, 'mv1_b')
    mv1_A = _find_channel_name(channels, 'mv1_a')
    mv0_R = _find_channel_name(channels, 'mv0_r')
    mv0_G = _find_channel_name(channels, 'mv0_g')
    mv0_B = _find_channel_name(channels, 'mv0_b')
    mv0_A = _find_channel_name(channels, 'mv0_a')
    mask_R = _find_channel_name(channels, 'mask_r')

    worldpos = (np.array(array.array('f', img_exr.channel(depth_R, pt))).reshape(size)
                if depth_R is not None else None)

    data = _read_camera_header(img_exr, size)
    data_pre, data_cur = data.copy(), data.copy()
    for key in ('x', 'y', 'z'):
        data_cur[key] = _read_header_float(header, f'cur_pos_{key}')
        data_pre[key] = _read_header_float(header, f'prev_pos_{key}')
    for key in ('pitch', 'roll', 'yaw'):
        data_cur[key] = _read_header_float(header, f'cur_rot_{key}')
        data_pre[key] = _read_header_float(header, f'prev_rot_{key}')

    mv0 = _read_mv_channels(img_exr, size, mv0_R, mv0_G, mv0_B, mv0_A)
    mv1 = _read_mv_channels(img_exr, size, mv1_R, mv1_G, mv1_B, mv1_A)
    mask = (np.array(array.array('f', img_exr.channel(mask_R, pt))).reshape(size)
            if mask_R is not None else None)

    r_str, g_str, b_str = img_exr.channels('RGB', pt)
    red = np.array(array.array('f', r_str)).reshape(size)
    green = np.array(array.array('f', g_str)).reshape(size)
    blue = np.array(array.array('f', b_str)).reshape(size)
    image = np.stack([red, green, blue], axis=2).astype('float32')
    return image, worldpos, data_pre, data_cur, mv0, mv1, mask, dtype


def exr_read_worldpos_next(filePath):
    """读取相邻帧的相机参数与 mv0 (轻量版, 不读 RGB/深度)。"""
    if not os.path.isfile(filePath):
        return None, None
    img_exr = OpenEXR.InputFile(filePath)
    header = img_exr.header()
    dw = header['dataWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    data_cur = _read_camera_header(img_exr, size)
    for key in ('x', 'y', 'z'):
        data_cur[key] = _read_header_float(header, f'cur_pos_{key}')
    for key in ('pitch', 'roll', 'yaw'):
        data_cur[key] = _read_header_float(header, f'cur_rot_{key}')

    channels = header['channels']
    mv0_channels = tuple(
        _find_channel_name(channels, f'mv0_{component}')
        for component in ('r', 'g', 'b', 'a')
    )
    if all(channel is not None for channel in mv0_channels):
        mv0 = _read_mv_channels(img_exr, size, *mv0_channels)
    else:
        mv0 = None
    return data_cur, mv0


# ─────────────────────────────────────────────────────────────────────────────
# 相机 MV 计算
# ─────────────────────────────────────────────────────────────────────────────

def GetViewMatrixFromEular(pitch, yaw, roll):
    half_yaw, half_pitch, half_roll = D2R * yaw, D2R * pitch, D2R * roll
    SP, SY, SR = np.sin(half_pitch), np.sin(half_yaw), np.sin(half_roll)
    CP, CY, CR = np.cos(half_pitch), np.cos(half_yaw), np.cos(half_roll)
    viewMat_T = np.zeros(16, dtype=np.float32)
    viewMat_T[0] = CP * CY
    viewMat_T[1] = CP * SY
    viewMat_T[2] = SP
    viewMat_T[4] = SR * SP * CY - CR * SY
    viewMat_T[5] = SR * SP * SY + CR * CY
    viewMat_T[6] = -SR * CP
    viewMat_T[8] = -(CR * SP * CY + SR * SY)
    viewMat_T[9] = CY * SR - CR * SP * SY
    viewMat_T[10] = CR * CP
    viewMat_T[15] = 1
    return viewMat_T.reshape((4, 4)).T


def GetViewMatrixFromEularAngle(pitch, yaw, roll):
    viewMat_Yaw = GetViewMatrixFromEular(0, yaw, 0)
    viewMat_Pitch = GetViewMatrixFromEular(pitch, 0, 0)
    viewMat_Roll = GetViewMatrixFromEular(0, 0, roll)
    return np.dot(np.dot(viewMat_Yaw, viewMat_Roll), viewMat_Pitch)


def inv_4x4_matrix(src):
    return np.linalg.inv(np.array(src).reshape(4, 4))


def camera_tracking(depth, o_data, p_data, factor=1, reverse=1):
    """由深度 + 两帧相机位姿计算相机运动引起的屏幕空间 MV。"""
    w, h = o_data['w'], o_data['h']
    o_view = GetViewMatrixFromEularAngle(-o_data['yaw'], o_data['roll'], o_data['pitch'])
    p_view = GetViewMatrixFromEularAngle(-p_data['yaw'], p_data['roll'], p_data['pitch'])
    inv_o_view = inv_4x4_matrix(o_view)

    def fx_of(d):
        if d['camera_type'] == 0:
            return w * d['focal_length'] / d['sensor_w']
        half_fov = D2R * d['fov'] / 2
        focal_len = d['sensor_w'] / (np.tan(half_fov) * 2)
        return w * focal_len / d['sensor_w']

    o_fx = fx_of(o_data)
    p_fx = fx_of(p_data)
    o_fy, p_fy = o_fx, p_fx

    o_cx = p_cx = w / 2.0
    o_cy = p_cy = h / 2.0

    delta_x = p_data['y'] - o_data['y']
    delta_y = p_data['z'] - o_data['z']
    delta_z = -p_data['x'] + o_data['x']

    o_CS_Z = depth
    oxx = np.repeat(np.arange(w)[::-1][None, ...], h, axis=0)
    oyy = np.repeat(np.arange(h)[..., None], w, axis=1)
    o_CS_X = (oxx - o_cx) / o_fx * o_CS_Z
    o_CS_Y = (oyy - o_cy) / o_fy * o_CS_Z

    o_WS_X = inv_o_view[0][0] * o_CS_X + inv_o_view[0][1] * o_CS_Y + inv_o_view[0][2] * o_CS_Z + inv_o_view[0][3]
    o_WS_Y = inv_o_view[1][0] * o_CS_X + inv_o_view[1][1] * o_CS_Y + inv_o_view[1][2] * o_CS_Z + inv_o_view[1][3]
    o_WS_Z = inv_o_view[2][0] * o_CS_X + inv_o_view[2][1] * o_CS_Y + inv_o_view[2][2] * o_CS_Z + inv_o_view[2][3]
    o_WS_W = inv_o_view[3][0] * o_CS_X + inv_o_view[3][1] * o_CS_Y + inv_o_view[3][2] * o_CS_Z + inv_o_view[3][3]
    o_WS_X = o_WS_X + delta_x
    o_WS_Y = o_WS_Y + delta_y
    o_WS_Z = o_WS_Z + delta_z

    p_CS_X = p_view[0][0] * o_WS_X + p_view[0][1] * o_WS_Y + p_view[0][2] * o_WS_Z + p_view[0][3] * o_WS_W
    p_CS_Y = p_view[1][0] * o_WS_X + p_view[1][1] * o_WS_Y + p_view[1][2] * o_WS_Z + p_view[1][3] * o_WS_W
    p_CS_Z = p_view[2][0] * o_WS_X + p_view[2][1] * o_WS_Y + p_view[2][2] * o_WS_Z + p_view[2][3] * o_WS_W

    p_SS_X = p_fx * p_CS_X / p_CS_Z + p_cx
    p_SS_Y = p_fy * p_CS_Y / p_CS_Z + p_cy

    mv_x = (p_SS_X - oxx) * reverse * -1
    mv_y = (p_SS_Y - oyy) * reverse
    mv_z = p_CS_Z - o_CS_Z
    mv_depth = o_CS_Z * reverse
    return np.stack([mv_x, mv_y, mv_z, mv_depth], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 图像/MV 后处理
# ─────────────────────────────────────────────────────────────────────────────

def adjust(res):
    """MV 归一化 (x除以宽, y除以高), 清零第3通道 mv_z。"""
    res[..., 0] /= res.shape[1]
    res[..., 1] /= res.shape[0]
    res[..., 2] *= 0
    return res


def select_mv(camera_mv, obj_mv, mask=None, *, obj_mv_only=False,
              bg_mode=False, label='mv'):
    """根据MV来源模式返回最终运动场。

    ``obj_mv_only`` 表示EXR的obj MV已包含相机与物体的完整运动，因此直接
    返回它，不再做mask替换。默认模式保持历史逻辑：camera MV作为背景，
    obj MV只覆盖运动物体；``bg_mode``则只保留camera MV。
    """
    if obj_mv_only and bg_mode:
        raise ValueError('obj_mv_only与bg_mode不能同时开启')
    if obj_mv_only:
        if obj_mv is None:
            raise ValueError(
                f'{label}: 已开启--obj-mv-only，但EXR中缺少对应obj MV通道')
        return np.array(obj_mv, copy=True)
    if camera_mv is None:
        raise ValueError(f'{label}: camera MV不能为空')
    if obj_mv is None or bg_mode:
        return camera_mv

    if mask is None:
        moving = np.where(
            (obj_mv[..., 0] != 0) | (obj_mv[..., 1] != 0))
        camera_mv[moving] = obj_mv[moving]
    else:
        # 保持原有mask合成语义：mask只替换屏幕空间XY，不改Z/A。
        mask_xy = np.repeat(mask[..., None], 2, axis=2)
        moving = np.where(mask_xy != 0)
        camera_mv[moving] = obj_mv[moving]
    return camera_mv


# ── UE 风格 tone map: ACES RRT+ODT 拟合 (Stephen Hill fit) ───────────────────
# UE 的 Filmic Tonemapper 即基于 ACES RRT/ODT; 此拟合与编辑器观感一致
# (剩余差异是 auto exposure, 由 --exposure 固定值近似)。
_ACES_INPUT_MAT = np.array([
    [0.59719, 0.35458, 0.04823],
    [0.07600, 0.90834, 0.01566],
    [0.02840, 0.13383, 0.83777],
], dtype=np.float32)

_ACES_OUTPUT_MAT = np.array([
    [1.60475, -0.53108, -0.07367],
    [-0.10208, 1.10813, -0.00605],
    [-0.00327, -0.07276, 1.07602],
], dtype=np.float32)


def _rrt_and_odt_fit(v):
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    return a / b


def aces_filmic(x):
    """输入: 线性 rec709 [0, inf), 输出: tone mapped 线性 [0, 1]。"""
    shp = x.shape
    v = x.reshape(-1, 3).astype(np.float32) @ _ACES_INPUT_MAT.T
    v = _rrt_and_odt_fit(v)
    v = v @ _ACES_OUTPUT_MAT.T
    return np.clip(v.reshape(shp), 0.0, 1.0)


def linear_to_srgb(x):
    """标准 sRGB OETF (分段), 输入输出 [0,1]。"""
    return np.where(x <= 0.0031308,
                    x * 12.92,
                    1.055 * np.power(np.clip(x, 1e-8, None), 1 / 2.4) - 0.055)


def hdr_to_rgb(hdr_image, exrformat=False, exposure=1.0):
    """生成 image 输出。

    EXR 保留线性 rec709 float32 数据（包括负值）；PNG 才执行
    exposure 增益、ACES filmic 和 sRGB 编码。将 EXR 与显示转换分开，
    可避免负值参与分数次幂时生成 NaN，也避免被裁剪到 0。
    """
    rgb = np.ascontiguousarray(hdr_image[..., :3], dtype=np.float32)
    if exrformat:
        return rgb.copy()

    x = np.clip(rgb * exposure, 0.0, None)
    x = linear_to_srgb(aces_filmic(x))
    return (x * 255.0 + 0.5).astype('uint8')


# ─────────────────────────────────────────────────────────────────────────────
# 核心 worker
# ─────────────────────────────────────────────────────────────────────────────

def mv_cal_core(datas):
    i, file_name, save_path, img, extra_depth, args = datas
    step = args.step
    base = os.path.basename(img)

    # 断点续跑早退
    if step == 1 and not args.f:
        if args.depth_only:
            if os.path.isfile(os.path.join(save_path, 'world_depth', base)):
                return
        elif os.path.isfile(os.path.join(save_path, 'mv1', base)):
            return

    mv0_sp = os.path.join(save_path, f'mv{(step - 1) * 2}')
    mv1_sp = os.path.join(save_path, f'mv{(step - 1) * 2 + 1}')

    hdr_image, depth, data_prev_hdr, data1, objmv0_, objmv1, mask, dtype = \
        exr_read_worldpos(file_name[i])

    if mask is not None:
        mvwrite(os.path.join(save_path, 'Mask', base),
                np.repeat(mask[..., None], 4, axis=2))
    if args.mask_only:
        return

    if extra_depth is not None:
        depth = read(extra_depth)[..., 0] * 100

    if depth is None and (args.dump_depth or args.dump_ply):
        return
    if args.depth_only:
        pass
    elif depth is None and not args.objmvonly:
        # camera MV依赖深度；完整obj MV模式不需要深度。
        pass
    else:
        # ── mv1: 当前帧 → 前一帧 (i-step) ─────────────────────────────────
        if i - step >= 0:
            data0, _ = exr_read_worldpos_next(file_name[i - step])
        else:
            data0, objmv1 = None, None

        # debug 校验: 前一帧的 cur 相机与当前帧 header 里的 prev 相机应一致
        if args.debug and data0 is not None:
            for sens in ['x', 'y', 'z', 'pitch', 'roll', 'yaw']:
                assert data0[sens] == data_prev_hdr[sens], \
                    f'data error! {sens}: {data0[sens]} != {data_prev_hdr[sens]}'

        if data0 is None:
            mv1 = np.zeros((hdr_image.shape[0], hdr_image.shape[1], 4))
        elif args.objmvonly:
            mv1 = select_mv(
                None, objmv1, obj_mv_only=True,
                label=f'{base} mv1(current->previous)')
        else:
            camera_mv1 = camera_tracking(depth, data1, data0)
            mv1 = select_mv(
                camera_mv1, objmv1, mask, bg_mode=args.bg_mode,
                label=f'{base} mv1(current->previous)')
        mvwrite(os.path.join(mv1_sp, base), adjust(mv1), precision='half')

        # ── mv0: 当前帧 → 后一帧 (i+step) ─────────────────────────────────
        if args.objmvonly or objmv0_ is not None or args.bg_mode:
            if i + step < len(file_name):
                data1_, objmv0 = exr_read_worldpos_next(file_name[i + step])
            else:
                data1_, objmv0 = None, None
            if data1_ is None:
                mv0 = np.zeros((hdr_image.shape[0], hdr_image.shape[1], 4))
            elif args.objmvonly:
                mv0 = select_mv(
                    None, objmv0, obj_mv_only=True,
                    label=f'{base} mv0(current->next)')
            else:
                camera_mv0 = camera_tracking(depth, data1, data1_)
                mv0 = select_mv(
                    camera_mv0, objmv0, mask, bg_mode=args.bg_mode,
                    label=f'{base} mv0(current->next)')
            mvwrite(os.path.join(mv0_sp, base), adjust(mv0), precision='half')

    if step != 1:
        return

    # ── HDR 原色域落盘 (可选, 仅控制是否额外保存HDR原图) ──────────────────
    if args.enable_colour_output:
        rec709_to_acescg, _ = get_color_transforms()
        if dtype == 'acescg':
            mvwrite(os.path.join(save_path, 'ACESCG', base), hdr_image)
        elif dtype == 'rec709':
            mvwrite(os.path.join(save_path, 'rec709', base), hdr_image)
            if args.ACESCG:
                mvwrite(os.path.join(save_path, 'ACESCG', base),
                        rec709_to_acescg.apply(hdr_image))

    # ── LDR 前的色域统一: acescg 源无条件先转 rec709 (不再依赖开关) ────────
    if dtype == 'acescg':
        _, acescg_to_rec709 = get_color_transforms()
        hdr_image = acescg_to_rec709.apply(hdr_image)

    # ── image 输出: PNG 走显示转换, EXR 保留线性 float32 ──────────────
    image = hdr_to_rgb(hdr_image, exrformat=args.exrformat, exposure=args.exposure)
    if not args.exrformat:
        image_path = os.path.join(save_path, 'image', base).replace('.exr', '.png')
    else:
        image_path = os.path.join(save_path, 'image', base)
    if args.onlymv and (args.f or not os.path.isfile(image_path)):
        # image EXR 用 float32 落盘，避免默认 half 进一步损失 HDR 范围/精度。
        precision = 'float' if args.exrformat else 'half'
        mvwrite(image_path, image, precision=precision)

    # ── 深度输出 ──────────────────────────────────────────────────────────
    if args.dump_depth or args.dump_ply:
        depth = depth / 100
        depth[np.where(depth > 1e5)] = 0        # 无效深度 (如天空盒)
        dpname = os.path.join(save_path, 'world_depth', base)
        depth = np.repeat(depth[..., None], 4, axis=2)
        d = depth[..., -1]
        d = (d - d.min()) / (d.max() - d.min())
        depth[..., -1] = d
        if args.colormap:
            d = (depth[..., -1] * 255.0).astype('uint8')
            depth = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
            dpname = dpname.replace('.exr', '.png')
        mvwrite(dpname, depth)


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = init_param()
    root = args.path
    restore_file_name(root)

    file_names = loop_helper(root)
    assert len(file_names) > 0, 'error root'

    for id_, file_name in enumerate(file_names):
        print('starting camera mv calculation({}/{}) {}'.format(
            id_ + 1, len(file_names), file_name))

        if args.output is None:
            # 原地模式: 先把裸帧收进 ori/ 子目录
            if len(jhelp_file(file_name)) != 0 and os.path.basename(file_name) != 'ori':
                ori_files = jhelp_file(file_name)
                if len(ori_files) > 0:
                    mkdir(os.path.join(file_name, 'ori'))
                    for ori_file in ori_files:
                        shutil.move(ori_file, os.path.join(file_name, 'ori'))
            if os.path.basename(file_name) != 'ori':
                file_name = os.path.join(file_name, 'ori')
            save_path = os.path.abspath(os.path.join(file_name, '..'))
        else:
            save_path = file_name.replace(root, args.output)

        if not os.path.isdir(file_name):
            continue
        name = os.path.basename(save_path)

        file_datas = jhelp_file(file_name)
        file_datas = prune(file_datas, 'finalimage')
        file_datas = sorted(file_datas, key=extract_last_4_digits)

        # 外部深度
        extra_depth = [None] * len(file_datas)
        if args.extra_depth is not None:
            extra_depth_path = os.path.join(args.extra_depth, name)
            depth_name = 'mono_depth'
            for f in jhelp_folder(extra_depth_path):
                if 'mono_depth' in f and 'mono_depth' != f:
                    depth_name = f
            extra_depth = jhelp_file(os.path.join(extra_depth_path, depth_name))

        data = [[i, file_datas, save_path, file_datas[i], extra_depth[i], args]
                for i in range(len(file_datas))]

        if args.core == 0:
            for d in data:                      # 单进程: 便于调试与看完整堆栈
                mv_cal_core(d)
        else:
            process_map(mv_cal_core, data, max_workers=args.core,
                        desc='processing:{}'.format(name))

        if args.dump_ply:
            from conversion_tools.pointcloud.unreal_reader import unreal_ply
            unreal_ply(save_path, args.down_scale)


if __name__ == '__main__':
    main()
