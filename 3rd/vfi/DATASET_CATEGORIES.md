# VFI 数据分类

训练数据按目录和独立list分类。默认支持：

- `easy` / `normal` / `hard`：原有难度分类。
- `opensource`：Vimeo90K、X4K、SNU-FILM等开源数据。
- `illumination`：闪烁、曝光漂移、光暗突变和局部照明变化。
- `noise`：高ISO、低照度噪点、压缩噪声和合成噪声。
- `teacher`：带 `mv0/mv1` 光流监督的数据。

## 目录结构

`build_lists.py` 会递归搜索各分类下直接包含连续帧的目录，
因此开源数据集可以保留自身的多层目录结构：

```text
vfi_database/
├── opensource/Vimeo90K/sequences/00001/0001/*.png
├── opensource/X4K/train/scene0001/*.png
├── illumination/custom/scene0001/*.png
├── noise/high_iso/scene0001/*.png
├── easy/...
├── normal/...
├── hard/...
├── teacher/...
└── val/...
```

`opensource` / `illumination` / `noise` 默认允许只有3帧的triplet
scene。训练Dataset会根据scene帧数自动选择可用framestep；例如三帧
scene只会使用 `framestep=1`。

生成清单：

```bash
python data_prepare/build_lists.py \
  --root /path/to/vfi_database \
  --max_framestep 2
```

会新增：

```text
lists/opensource_train.txt
lists/illumination_train.txt
lists/noise_train.txt
```

## 采样比例

YAML的 `data.tiers` 声明可用分类，每个phase的 `ratios` 控制采样概率。
权重会自动归一化，不必严格相加为1。

```yaml
data:
  tiers: [easy, normal, hard, opensource, illumination, noise, teacher]

phases:
  - name: mixed_special
    epochs: 100
    ratios:
      easy: 0.20
      normal: 0.20
      hard: 0.15
      opensource: 0.15
      illumination: 0.10
      noise: 0.10
      teacher: 0.10
```

某分类 `ratio: 0` 时，对应list允许不存在。一旦权重大于0，
训练启动时会检查对应 `*_train.txt`，缺失时直接报出明确错误。
