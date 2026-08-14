# Teacher MV validation and cycle filtering (2026-08-13)

## Verified convention

- `mv1`: middle/current frame to the previous input frame.
- `mv0`: middle/current frame to the next input frame.
- Cache channel order: `mv1_x, mv1_y, mv0_x, mv0_y`.
- Stored units are normalized by source width/height; the dataset restores pixel units.
- Current signs `[1, 1]`, channel order, and scale outperform swapped, negated,
  half-scale, double-scale, and zero-flow controls on both clean and final renders.

On 501 deterministic triplets per domain, moving-region best-of-two MAE was:

| Domain | Zero flow | Current MV |
| --- | ---: | ---: |
| clean | 0.043442 | 0.011944 |
| final | 0.039019 | 0.012554 |

The sampled EXR-to-`mv_cache_f16` comparison was numerically exact, and sampled
clean/final MV copies were byte-identical.

## Confirmed bad tail supervision

The final render contains five frames not present in the corresponding clean
sequence. The old clean endpoint therefore becomes a final middle-frame
candidate although its `mv0` is zero:

- `finalUrbanMartialArtsWoman/24fps/1.0318`
- `finalUrbanMartialArtsWoman/48fps/1.0637`
- `finalWerewolf/12fps/1.0115`
- `finalWerewolf/24fps/1.0231`
- `finalWerewolf/48fps/1.0463`

The on-the-fly cycle mask gives these samples zero valid teacher pixels while
retaining them for image reconstruction. It conservatively does the same for
the second and penultimate frames of every scene because an adjacent endpoint
does not have a two-direction cache pair.

## Implemented filter

For current-to-adjacent flow `f` and sampled adjacent-to-current flow `b`, the
filter computes:

```text
residual2 = ||f + warp(b, f)||^2
threshold = 0.05 * (||f||^2 + ||warp(b, f)||^2) + 1.0
confidence = threshold / (threshold + residual2)
hard_valid = confidence >= 0.5
```

Training computes this after choosing the random crop. It mmap-reads only the
same crop from the previous and next MV caches. Pixels whose endpoints leave
the crop are already invalid for supervision, so crop-local remapping is
equivalent for all supervised pixels and avoids full-frame I/O.

Real-data smoke checks:

- `teacher/Spring/0001/left/frame_left_0011.png`: 97.90% valid in a 256 crop.
- `finalUrbanMartialArtsWoman/24fps/1.0318.png`: 0% valid as expected.
- Raw 256-crop MV read: 3.59 ms/sample.
- On-the-fly cycle-filtered read: 13.06 ms/sample.

## Controlled ablation

All variants use the same `0807_s2v3_official_tuesday_320.pkl`, model, seed,
data mixture, crop, schedule, and validation mask:

- `train_config_teacher_ablation_noflow.yaml`: teacher images, no flow loss.
- `train_config_teacher_ablation_raw.yaml`: raw teacher flow, weight 0.0005.
- `train_config_teacher_ablation_cycle.yaml`: hard-cycle flow, weight 0.0005.

Validation logs overall/moving/static EPE, valid-pixel ratio, sample coverage,
and clean/final-by-FPS metrics. Compare PSNR together with moving EPE; a lower
EPE that reduces reconstruction quality is not considered a successful teacher.
