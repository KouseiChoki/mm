# 0729 LC hard-ROI causal ablation (2026-08-19)

## Question

Can pixel-level supervision on genuinely interpolation-hard regions improve
small foregrounds and dense/high-frequency motion without sacrificing the
stable texture behavior of `0729_lc_v3s2`?

This experiment does not add a new inference branch.  It only changes the
training objective, so inference latency and checkpoint architecture remain
unchanged.

## Hard-region definition

For timestamp `t`, form the endpoint-linear baseline

`linear = (1 - t) * img0 + t * img1`.

The detached RGB L1 residual `abs(gt - linear)` identifies pixels that cannot
be explained by a stationary linear blend.  Per image, select the largest 10%
of residuals, discard values below 0.04, then apply a 5x5 morphological
dilation so thin object and motion boundaries are covered.  The actual mask
area is logged because thresholding and dilation make it data-dependent.

## Strict A/B

Both arms restore
`ckpt/0729_lc_v3s2/0729_lc_v3s2_800.pkl` without `--resume`, use seed 1234,
and share every data, optimizer, phase and model setting.  Both compute the
same ROI diagnostics.  Only the following weights differ:

| Loss on hard ROI | Control | Weighted |
|---|---:|---:|
| final-output Charbonnier | 0.00 | 0.10 |
| last-warp Charbonnier | 0.00 | 0.10 |
| final-output Sobel Charbonnier | 0.00 | 0.05 |

Training budget is 4 epochs x 401 optimizer steps with batch composition
`hard:vimeo:xtrain = 1:1:2`, crop 384x704 and gradient accumulation 3.
Validation and checkpointing run every epoch.

## Diagnostics and decision rule

Training logs include `roi`, `roi_final`, `roi_warp` and `roi_edge`.
Validation reports:

- `roi_psnr`: final output PSNR inside the shared hard mask;
- `warp_roi_psnr`: last warp/merge PSNR inside the same mask;
- `roi_final_gain_db`: final output minus last-warp ROI PSNR;
- `roi_l1`, `roi_edge_error`, `roi_area`, and sample coverage.

Advance the method only if the weighted arm improves hard/xtrain ROI metrics
and visual small-object/detail stability while keeping full-frame Vimeo/X4K
PSNR and real-sequence temporal stability at least neutral.  If only final
ROI improves but warp ROI does not, the gain comes from the refiner rather
than better flow.  If neither improves, stop this direction rather than adding
more weight.

## Preflight evidence

On mms3 RTX 5090, a real batch of 4 at 384x704 completed forward/backward:

- peak allocated: 13.34 GiB; peak reserved: 16.21 GiB;
- measured ROI area over 12 real samples: mean 21.7%, range 5.9%-51.5%;
- control loss: 0.0675683;
- weighted loss: 0.0704398;
- the 0.0028715 difference exactly equals the three logged weighted ROI terms.

All 45 repository unit tests passed after the implementation.
