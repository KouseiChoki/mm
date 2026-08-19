# Explicit matching A/B result (2026-08-19)

## Verdict

The shared X4K-heavy repair recipe is effective for large-motion recursive
interpolation, but the current post-hoc GMFlow sparse-matching branch does not
pass the causal gate.  Keep the epoch-2 control checkpoint as the useful
candidate.  Do not extend the GMFlow arm in its current form.

Both arms restored `0729_lc_v3s2_800.pkl` without resume and ran the fixed v2
configs for ten epochs.  Their best X4K validation checkpoints were both saved
at epoch 2.  Later epochs generally regressed.

## Training validation

| Epoch | Control hard | GMFlow hard | Control Vimeo | GMFlow Vimeo | Control X4K | GMFlow X4K |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 39.8871 | 39.8855 | 36.4206 | 36.4189 | 41.9925 | 42.0093 |
| 4 | 39.8381 | 39.8541 | 36.4138 | 36.4249 | 41.9732 | 41.9775 |
| 6 | 39.8429 | 39.8466 | 36.4241 | 36.4200 | 41.9745 | 41.9638 |
| 8 | 39.8284 | 39.8306 | 36.4153 | 36.4207 | 41.9655 | 41.9684 |
| 10 | 39.8407 | 39.8316 | 36.4191 | 36.4201 | 41.9641 | 41.9569 |

At the best epoch, GMFlow gained only 0.0167 dB on X4K while losing about
0.002 dB on hard and Vimeo.  The sign changes at later epochs, so this is not
a robust validation improvement.

## Same-machine complete benchmark

All three checkpoints were evaluated on mms3 with TTA disabled.  Vimeo uses
all 3782 official triplets.  X4K uses 18 held-out scenes, recursive 8x
interpolation and 252 predicted frames.

| Model | Vimeo PSNR | Vimeo SSIM | X4K PSNR | X4K SSIM |
|---|---:|---:|---:|---:|
| 0729 LC | 35.5339 | 0.978047 | 31.0256 | 0.903634 |
| Epoch-2 control | **35.5608** | **0.978133** | 31.7352 | **0.910380** |
| Epoch-2 GMFlow | 35.5510 | 0.978081 | **31.7539** | 0.910227 |

Relative to 0729, the control gains 0.7096 dB and 0.006747 SSIM on recursive
X4K.  This proves the shared data/crop/edge-loss repair is useful.  Relative
to that control, GMFlow changes the result by:

- Vimeo: -0.0098 dB, -0.000051 SSIM;
- X4K: +0.0187 dB, -0.000154 SSIM.

The small X4K PSNR gain is paired with an SSIM loss and a Vimeo regression.
It is not enough to justify the branch.

Machine-readable results are committed under
`benchmark_results/explicit_matching_v2/`.

## Why the branch did not help

The matcher found candidates, but the learned adapter mostly rejected them.

| Training step | Selected-region correction | Global correction |
|---:|---:|---:|
| 0 | 0.0000 px | 0.0000 px |
| 800 | 0.0348 px | 0.0054 px |
| 1600 | 0.0258 px | 0.0043 px |
| 2400 | 0.0032 px | 0.0008 px |
| 3200 | 0.0091 px | 0.0022 px |
| 4000 | 0.0062 px | 0.0013 px |

The branch briefly crossed the 0.02 px activation gate near epoch 2, then was
suppressed.  At the end, only 47.85% of proposed matches improved feature
similarity over the current flow correspondence.  High mutual consistency
(90.23%) therefore did not imply that the proposal was useful to the VFI
objective, especially on repeated textures.

## Fixed real-scene gate

Sixteen fixed scenes were evaluated on the same first pair.  Between the
control and GMFlow outputs:

- mean absolute difference: 0.115 of one 8-bit level;
- mean pixels differing by more than two levels: 0.34%;
- mean gradient-magnitude ratio: 0.9989;
- typical inference overhead: 17.3 ms, about 5.6% in aggregate;
- ten-epoch training time: 47.0 minutes control vs 53.8 minutes GMFlow.

The outputs are visually almost identical.  The largest isolated difference
appeared at a moving foreground/window boundary in `Hunter_clip083`; it did
not restore the blurred contour and introduced a local color displacement.

Outputs:

- `/home/zhenying/qhong/sync/result/0729_lc_800_no_tta_gate`
- `/home/zhenying/qhong/sync/result/0819_explicit_control_v2_best_gate`
- `/home/zhenying/qhong/sync/result/0819_explicit_gmflow_v2_best_gate`

## Reusable conclusion

Do not continue by merely increasing the GMFlow arm's epochs, confidence or
residual scale.  The evidence indicates a structural limitation: a sparse
post-hoc correction after the main flow stages is easy for reconstruction
training to suppress and does not materially change the final interpolation.
If explicit correspondence is revisited, inject correlation/matching evidence
inside an IFBlock before flow refinement, with a region-level objective that
can verify correspondence quality.  Treat that as a new architecture test,
not a continuation of this checkpoint.
