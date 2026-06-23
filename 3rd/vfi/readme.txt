# VFIMamba 用户手册

## 概述

VFIMamba 是一个基于 Mamba 架构的视频帧插值工具，输入相邻两帧图像，输出中间插值帧。支持图像序列批量处理，可输出图片或视频。

---

## 环境要求

- Python 3.10+
- PyTorch（支持 CUDA / MPS / CPU）
- 依赖库：`opencv-python`, `numpy`, `tqdm`, `requests`, `einops`
- macos需要额外安装mamba_ssm_macos库

---

## 快速开始
```zsh
mmvfi \
    --root /path/to/input_frames \
    --output /path/to/output
```

或者
```zsh
python start.py \
    --root /path/to/input_frames \
    --output /path/to/output
```

首次运行会自动从服务器下载模型权重。

---

## 参数说明

### 必填参数

| 参数 | 说明 |
|------|------|
| `--root` / `--path` | 输入图像序列的根目录 |
| `--output` | 输出结果的根目录 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--algo` | `VFIKousei_TEST` | 权重文件名（不含 `.pth`），用于自动下载,通过mmalgo获取 |
| `--server` | `http://10.35.180.69:80` | 模型权重下载服务器地址 |

### 输出参数

| 参数 | 默认值 | 可选值 | 说明 |
|------|--------|--------|------|
| `--output_mode` | `image` | `image` / `video` / `both` | 输出格式 |
| `--fps` | `24` | 任意浮点数 | 输出视频帧率（仅 `video`/`both` 模式有效） |
| `--video_ext` | `mp4` | `mp4` / `avi` | 视频容器格式 |

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scale` | `0` | 光流估计缩放比例，`0` 为自动,目前不支持任意scale |

### 调试参数

| 参数 | 说明 |
|------|------|
| `--dump_data` | 测试模式，同时输出中间结果（光流、warp图、mask等） |

---

## 输入格式

### 目录结构

工具会递归扫描 `--root` 下所有子文件夹，自动识别图像序列：

```
input_root/
├── scene_A/
│   ├── 000001.png
│   ├── 000002.png
│   └── 000003.png
└── scene_B/
    ├── frame_001.exr
    └── frame_002.exr
```

### 支持格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| PNG | `.png` | 8-bit 标准图像 |
| EXR | `.exr` | HDR 高动态范围图像 |
| TIFF | `.tif` | 16-bit 图像 |

### 命名规则

文件名中必须包含数字，工具按最后一个数字排序。例如：

```
frame_001.png → 编号 1
frame_002.png → 编号 2
000010.exr    → 编号 10
```

### 分辨率限制

- 目前不作限制，根据内存可最大支持4k,代码中限制为 **7280 × 4320**（8K）

---

## 输出格式

### 图像模式（`--output_mode image`）

输出目录结构与输入保持一致，插值帧编号为相邻帧编号的平均值：

```
output_root/
└── scene_A/
    ├── 000002.png   ← 原始第1帧（编号 1×2=2）
    ├── 000004.png   ← 插值帧（编号 (2+4)//2=3 → ×2=6，取中间）
    └── 000006.png   ← 原始第2帧（编号 2×2=4... ）
```

实际命名逻辑：原始帧编号 ×2，插值帧取两帧编号之和的一半。例如：

```
输入: 000001.png, 000002.png, 000003.png
输出: 000002.png(原帧1), 000003.png(插值), 000004.png(原帧2), 000005.png(插值), 000006.png(原帧3)
```

### 视频模式（`--output_mode video`）

每个子文件夹生成一个 `output.mp4`（或 `output.avi`）。

### Debug 模式（`--dump_data`）

额外输出以下中间结果到对应子文件夹：

```
output_root/scene_A/
├── warp0/      ← img1 warp 到 mid 的结果
├── warp1/      ← img0 warp 到 mid 的结果
├── mv0/        ← 前向光流 (.exr)
├── mv1/        ← 后向光流 (.exr)
├── mask/       ← 融合 mask (.exr)
├── merged/     ← warp 融合结果
└── res/        ← 残差图 (.exr)
```

---

## 模型权重

权重文件存放在：
```
mm/checkpoints/
```

首次使用时若本地不存在，自动从 `--server` 下载。也可手动放置权重文件跳过下载。

### 可用模型(其余模型可使用mmalgo查看)

| `--model` | `--algo` 示例 | 特点 |
|-----------|---------------|------|
| `VFIMamba` | `VFIMamba` | 官方自带模型，F=32 |
| `VFIMamba_KouSei` | `VFIKousei_TEST` | 初版测试模型，F=24 |

---

## 使用示例

### 基本插值（输出图像）

```bash
mmvfi \
    --root /data/input_sequence \
    --output /data/output_sequence
```

### 输出视频

```bash
mmvfi \
    --root /data/input_sequence \
    --output /data/output_sequence \
    --output_mode video \
    --fps 48
```

### 同时输出图像和视频

```bash
mmvfi \
    --root /data/input_sequence \
    --output /data/output_sequence \
    --output_mode both \
    --fps 24 \
    --video_ext mp4
```

### 使用标准版模型 + 调试输出

```bash
mmvfi \
    --root /data/input_sequence \
    --output /data/output_sequence \
    --algo VFIMamba \
    --dump_data
```


---

## 常见问题

**Q：支持批量处理多个场景吗？**
A：支持。`--root` 下所有子文件夹会被递归扫描并分别处理，结果保持原目录结构输出到 `--output`。

**Q：输入帧数必须是偶数吗？**
A：不需要。工具对相邻帧两两插值，N 帧输入会产生 2N-1 帧输出。

**Q：模型下载失败怎么办？**
A：手动将 `.pth` 文件放到 `mm/checkpoints/` 目录下，文件名与 `--algo` 参数一致即可。

**Q：EXR 文件支持 HDR 吗？**
A：支持，EXR 和 TIF 格式会以浮点数读写，保留完整动态范围。