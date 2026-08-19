# VFI experiment decisions

This file records conclusions that should survive individual configs, logs and
chat sessions.  Add a dated entry when an ablation changes the recommended
training policy.

## 2026-08-18: native teacher flow / FlyingThings3D

### Question

Can native ground-truth motion vectors repair the production behaviour of the
existing VFI models, especially foreground and small-object motion, without
damaging the stable high-frequency texture behaviour of `0729_lc_v3s2`?

### Evidence

The controlled full-model FlyingThings3D experiments used identical image
sampling and changed only direct flow supervision/filtering:

| Experiment | FlyingThings moving EPE at epoch 150 | Interpretation |
| --- | ---: | --- |
| no direct flow loss | 13.4181 | reconstruction-only control |
| raw native flow | 13.3004 | almost no useful gain |
| hard-cycle flow | **12.4658** | the only effective teacher-flow route |
| hard-cycle, local IFBlock only | 15.2831 | restricting supervision to local IFBlocks is harmful |

The `0729_lc_v3s2_800.pkl` flow-head fine-tune gave the following equal-budget
comparison at epoch 30:

| Metric | noflow | hard-cycle | Change |
| --- | ---: | ---: | ---: |
| all PSNR | 35.3089 | 35.3116 | +0.0027 |
| FlyingThings PSNR | 23.4381 | 23.4927 | +0.0546 |
| FlyingThings flow EPE | 12.9586 | 12.4973 | -3.56% |
| FlyingThings moving EPE | 13.6403 | 13.1560 | -3.55% |
| aggregate teacher moving EPE | 23.7995 | 23.0171 | -3.29% |

Long low-LR consolidation produced its best FlyingThings flow result at epoch
140 (`flow EPE=12.3919`, `moving EPE=13.0440`), but the checkpoint performed
poorly in real application footage.  Its all-set PSNR (`35.3071`) did not
improve over the 30-epoch control.  The real outputs remained globally close
to the original 0729 outputs while localized motion-boundary changes caused
visible regressions.  This demonstrates that lower teacher EPE is not a valid
production checkpoint-selection metric by itself.

Sintel, Spring and Unreal did not provide comparable positive evidence in the
completed source ablations.  FlyingThings3D was the only useful native flow
source, and only after forward/backward cycle filtering.

### Decision

FlyingThings3D plus hard-cycle confidence is an **early motion auxiliary**, not
a late production-model repair method and not a final-quality objective.

It may be included opportunistically in a new regular/full-model training run:

```yaml
model:
  flow_loss_weight: 0.0005
  flow_loss_warmup_steps: 2000

data:
  mv_cycle_confidence: hard
  mv_cycle_cache_required: true
  mv_cycle_on_the_fly: false
```

Recommended policy:

- Use a 5-10% FlyingThings3D sample share during only the first 20-30% of a
  from-scratch or broad full-model training schedule.
- Train the full motion/reconstruction system together during that stage.
- Set the FlyingThings3D share and direct flow loss to zero for the remaining
  70-80%, allowing target-domain reconstruction data to determine texture,
  mask, occlusion and refiner behaviour.
- Select checkpoints using real application clips plus ROI/edge/flicker
  measurements; use teacher EPE only as a diagnostic.

Do not repeat the following without a materially different architecture or
objective:

- late teacher-flow injection into a converged 0729 checkpoint;
- raw native flow without cycle filtering;
- teacher shares above 10%;
- direct teacher flow throughout the final convergence stage;
- local-IFBlock-only or existing-flow-head-only repair;
- checkpoint selection by aggregate PSNR or teacher EPE alone.

### Relevant artifacts

- Full four-way configs: `train_config_teacher_flyingthings_*.yaml`
- Full four-way launcher: `launch_teacher_flyingthings_ablation.sh`
- 0729 flow-head configs: `train_config_0729_lc_flowheads_{cycle,noflow}.yaml`
- 0729 launcher: `launch_0729_lc_flowheads_ablation.sh`
- Logs: `record/0814_teacher_flyingthings_*` and
  `record/0817_0729_lc_flowheads_*`
- Real-test failure checkpoint:
  `ckpt/0817_0729_lc_flowheads_cycle/0817_0729_lc_flowheads_cycle_140.pkl`

## Next-experiment gate

Long runs must no longer be launched from validation PSNR/EPE alone.  Every new
architecture first passes this sequence:

1. Verify exact checkpoint compatibility and identity/no-op initialization.
2. Run a short 5-10 epoch smoke test.
3. Render only the known difficult application clips (small foreground,
   fences/grids, blurred regions and occlusions).
4. Continue to a medium run only if those clips show a visible benefit without
   new cracks, flicker or texture loss.
5. Run a long experiment only after the medium checkpoint passes the same real
   test.

## Candidate backlog after teacher-flow closure

Priority is based on the observed production failures, implementation scope and
the ability to initialize a new branch as an exact no-op.

### P1: content-aware flow upsampling

The current `IFBlock` upsamples its five-channel flow/mask residual with
`torch.nn.functional.interpolate(..., mode="bilinear")`.  This is a likely
cause of small-object disappearance and smeared motion boundaries.  Add a
feature-guided residual-kernel/content-aware upsampler only to the last local
stage.  Its output projection must be initialized to zero so an old 0729
checkpoint is initially bit-identical while the new branch can learn from the
first update.

This is the smallest structural experiment that directly targets the known
failure.  BiM-VFI's Content-Aware Upsampling Network is the primary reference:
https://openaccess.thecvf.com/content/CVPR2025/html/Seo_BiM-VFI_Bidirectional_Motion_Field-Guided_Frame_Interpolation_for_Video_with_Non-uniform_CVPR_2025_paper.html

First gate: 5-10 epochs, regular reconstruction data only, then render the
known foreground/fence clips.  Do not start with another 100+ epoch run.

#### 2026-08-18 gate result: stop current P1 implementation

The zero-initialized 5-epoch `0818_0729_lc_caun_smoke` run was safe but
functionally inactive.  Its final mean flow correction was only `0.00050 px`;
the all-set PSNR gain over an exactly re-evaluated 0729 checkpoint was
`+0.00066 dB` and the hard-set gain was `+0.00183 dB`.  Across 15 real pairs,
the mean final-image change was `0.0000587` in normalized RGB and the mean
flow change was `0.00068 px`, both visually negligible.

Do not rerun or simply extend this CAUN configuration.  Revisit P1 only with
materially different boundary/ROI supervision that can directly reward flow
edge corrections; otherwise the ordinary LC reconstruction objective gives
the residual-kernel branch too little signal.

### P2: sparse explicit correspondence

Mamba supplies global context but does not itself guarantee explicit
cross-frame pixel correspondence.  Add a zero-initialized sparse matching
compensation branch at the 1/16 or 1/8 feature level and evaluate only
high-error/high-difference pixels.  This targets small objects and large
displacements while avoiding a full-resolution dense cost volume.

Primary reference: SGM-VFI:
https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Sparse_Global_Matching_for_Video_Frame_Interpolation_with_Large_Motion_CVPR_2024_paper.html

AMT's bidirectional all-pairs correlation is an alternative reference if the
sparse branch cannot establish reliable matches:
https://openaccess.thecvf.com/content/CVPR2023/html/Li_AMT_All-Pairs_Multi-Field_Transforms_for_Efficient_Frame_Interpolation_CVPR_2023_paper.html

The first repository implementation is deliberately lightweight: use the
existing 1/8 Mamba feature map, select about 2% high-disagreement target-grid
points, globally correlate their frame-0 features against frame 1, filter with
top-1 margin plus backward consistency, and feed the sparse midpoint-preserving
flow proposal into a zero-initialized adaptive merger.  It is SGM-inspired,
not an SGM-VFI reproduction: it does not import GMFlow or its pretrained
weights.  The first gate is `train_config_0729_lc_sparse_match_stage1.yaml`.

#### 2026-08-18 gate result: stop the lightweight frozen-feature P2

The 5-epoch run was stable and its best checkpoint was epoch 3, but hard PSNR
improved only `+0.00239 dB` over an exact 0729 re-evaluation.  Across 15 known
difficult real pairs, the mean final-image difference was `0.0000337` and the
mean flow difference was `0.00052 px`.  Corrections were concentrated around
the intended motion/texture boundaries, but the mean applied matching delta
plateaued near `0.0005 px`; isolated flow changes reached `0.76 px` without a
visible application-level gain.

Do not merely extend this configuration.  Five epochs do not disprove sparse
explicit correspondence in general, but 2005 optimizer updates are sufficient
to reject the current 13.7K-parameter merger over frozen, non-matching-trained
Mamba features.  A future P2 must change the feature supervision materially,
for example with pretrained GMFlow-style correspondence features and a
large-motion fine-tuning set.

### P3: multiple motion hypotheses for occlusion

One flow pair plus one soft mask is underdetermined at occlusions.  AMT derives
multiple fine-grained flow fields and blends their warped candidates.  A small
two-hypothesis local head can test this idea before attempting AMT's complete
architecture.  This is more invasive than P1/P2 and should be attempted only
after explicit matching is understood.

The first implementation keeps 0729 as hypothesis 0 and adds one
quarter-resolution bilateral flow/mask candidate.  Its final mixing channel is
zero initialized for exact checkpoint identity.  Unlike P1/P2, the alternate
candidate also receives a lightweight reconstruction objective restricted to
the primary warps' disagreement regions, so it gets a direct training signal
even before the final mixing channel opens.  The gate config is
`train_config_0729_lc_multi_hypothesis_stage1.yaml`: 10 epochs, 4010 optimizer
updates and validation every 2 epochs.

#### 2026-08-18 gate result: stop the frozen two-hypothesis P3

The best checkpoint was epoch 2 and changed all-set PSNR by only `+0.00004 dB`
and hard PSNR by `+0.00123 dB`.  On 15 difficult real pairs its mean final
RGB change was `0.00000262`.  The hidden alternate flow did move (`0.040 px`
mean at the best checkpoint), but its warped candidate differed from the
primary by only `0.000402`; the final mixer then reduced that already small
difference by another two orders of magnitude.  Epoch 10 remained inactive.

The direct alternate-candidate reconstruction objective therefore collapsed
the second candidate toward the same conditional mean as the first.  Do not
extend this frozen P3.  Multiple hypotheses remain plausible only inside a
jointly trained model with a winner-take-all/specialization objective or an
AMT-style multi-field architecture.

## 2026-08-18 root-quality joint-training decision

P1, P2 and P3 all exposed the same optimization failure: a converged frozen
0729 model plus a small reconstruction-trained branch learns an almost-zero
correction.  The next experiment must optimize the complete network, not add
another frozen adapter.

Inspection also found that `motion_aware_crop_prob` only affects samples with
teacher MV.  Most easy/normal/hard/opensource samples were still randomly
cropped, so the main reconstruction training did not actively expose the model
to small moving foregrounds.  The root-quality run makes two targeted changes:

- interpolation-aware crop selection for ordinary no-MV triples, using the
  residual between the real middle frame and the timestamp-weighted linear
  endpoint blend to find small hard regions;
- a lightweight luminance edge loss on both the final result and the last pure
  warp, weighted by endpoint motion, so boundary error reaches flow/mask rather
  than being hidden by the refiner.

The first config is `train_config_0729_lc_root_quality_joint.yaml`.  It restores
0729 only as initialization and trains all 70.1M parameters, including the
backbone, all flow stages, local IFBlocks, mask and refiner, for both phases.
It deliberately excludes teacher flow and all P1/P2/P3 branches so the result
can be attributed to sampling plus end-to-end boundary supervision.

### P4: lightweight PerVFI blending smoke test

The repository already contains a partial PerVFI-inspired quasi-binary blend
and `train_config_pervfi_ablation.yaml`, but the experiment was deferred.  It
does not include PerVFI's normalizing-flow generator and must not be presented
as a reproduction of the paper.  Because it is already implemented, only an
inference/very-short controlled smoke test is justified.  Stop immediately if
fence flicker, cracks or foreground switching increase.

Paper reference:
https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Perception-Oriented_Video_Frame_Interpolation_via_Asymmetric_Blending_CVPR_2024_paper.html

### Cross-cutting safety rule

Any new branch attached to 0729 must be additive and zero-initialized.  During
short fine-tuning, use the frozen original 0729 output as a preservation target
on regular replay samples.  A candidate is rejected if ROI edges or temporal
stability regress even when global PSNR remains unchanged.
