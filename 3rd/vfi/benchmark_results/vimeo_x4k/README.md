# VFIMamba checkpoint comparison

All models were evaluated with standard flip TTA. PSNR and SSIM are averaged
per predicted frame. SSIM uses the official VFIMamba `ssim_matlab`
implementation.

| Model | Vimeo90K PSNR | Vimeo90K SSIM | X4K holdout PSNR | X4K holdout SSIM |
|---|---:|---:|---:|---:|
|  v2s3 | 35.6768 | 0.978574 | 31.0328 | 0.904190 |
|  v2s3 Official| **36.7068** | **0.982079** | 29.3304 | 0.859523 |
| Official VFIMamba | 36.6356 | 0.981949 | **31.3434** | **0.906913** |

## Protocols

- Vimeo90K: all 3,782 non-empty entries from the official
  `tri_testlist.txt`; `im1 + im3 -> im2`, evaluated at native 448x256.
- X4K1000FPS holdout: the repository's 18-scene `xtrain_val.txt`. Each scene
  is a 65-frame 768x768 training crop. Intervals `0 -> 32` and `32 -> 64` are
  recursively interpolated 8x, producing 14 evaluated frames per scene and
  252 frames in total. Predictions are rounded to uint8 before metrics, as in
  the official XTEST script.

The available X4K data is not the independent full-resolution XTEST-2K/4K
benchmark used in the paper. These numbers must be labelled
`X4K1000FPS holdout (768x768)` and must not be reported as XTEST results.

## Checkpoints

- `ckpt/0729_lc_v3s2/0729_lc_v3s2_800.pkl`
- `ckpt/0807_s2v3_official_tuesday/0807_s2v3_official_tuesday_320.pkl`
- `/home/zhenying/qhong/repo/VFIMamba_official_eval/ckpt/VFIMamba.pkl`

Raw accumulators and metadata are stored in the three adjacent JSON files.
