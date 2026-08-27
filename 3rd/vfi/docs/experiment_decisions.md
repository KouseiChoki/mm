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

## 2026-08-20: move correspondence into the motion core

The additive-adapter phase is closed. CAUN, lightweight sparse matching,
post-hoc GMFlow matching and the second-motion-hypothesis branch all converged
to negligible final corrections when attached to the mature 0729 model. The
shared failure is structural: endpoint features are concatenated and passed to
the flow heads, but explicit correspondence evidence is not a required input
to the coarse-to-fine motion updates.

The next root experiment therefore uses flow-aligned local cost volumes inside
all three learned-feature flow heads:

- 1/16 stage, radius 6: approximately 96 input-pixel search range;
- 1/8 stage, radius 4: approximately 32 input-pixel refinement range;
- 1/4 stage, radius 3: approximately 12 input-pixel refinement range;
- after the first stage, the current bilateral flow warps both endpoint
  features onto the target grid before each cost volume is constructed;
- correlation channels are concatenated into the predictor itself, before
  each residual flow/mask update. They are not merged after the final flow.

This is an intentional architecture replacement, so the old zero-initialized
adapter safety rule does not apply. Restoring 0729 drops exactly the first
convolution weight of each of the three flow heads because its input width has
changed. The backbone, remaining flow layers, local IFBlocks and refiner remain
compatible and are jointly trainable. The first experiment excludes teacher
flow so that any result is attributable to the new motion core.

Artifacts:

- config: `train_config_0820_lc_pyramid_correlation.yaml`;
- CUDA preflight: `smoke_pyramid_correlation.py`;
- preflight result:
  `record/pyramid_correlation_preflight/local_batch4_384x704_5steps.json`.

RTX 5090 preflight at batch 4, 384x704:

- peak allocated/reserved memory: 18.46/23.41 GiB;
- steady microbatch forward/backward: 0.263 seconds after compilation;
- five-step same-batch loss: 0.19036 -> 0.17455;
- all three correlation encoders receive non-zero gradients immediately;
- initial normalized correlation entropy is high (0.91/0.97/0.89 from coarse
  to fine), as expected before the new predictor layers have learned.

The first ten epochs retain the existing control's exact 4010-step LR horizon;
epochs 11-20 are low-LR consolidation. Compare epochs 2/4/6/8/10 against the
already completed `0819_0729_lc_explicit_match_control_v2` run. Continue or add
FlyingThings hard-cycle only if all of the following hold:

1. by epoch 4 the initially reinitialized motion heads have recovered the
   control's hard/Vimeo validation range;
2. by epoch 10 X4K improves by at least 0.05 dB over the same-recipe control,
   while Vimeo and hard each regress by less than 0.02 dB;
3. correlation entropy/peak diagnostics show learned, non-degenerate matching;
4. known real clips show a visible small-object or large-motion improvement
   without fence flicker, cracks, blur or temporal instability.

If this fails, do not tune only the radii or train the same module for hundreds
of epochs. The next materially different architecture would be an AMT-style
all-pairs recurrent correlation lookup plus jointly trained multi-field motion,
which should be treated as a new model rather than another 0729 repair.

## 2026-08-21: pretrained correspondence integration and Tuesday run

The raw Mamba cost-volume replacement failed, so correspondence was first
trained independently on FlyingThings hard-cycle supervision. Its best
checkpoint reached 87.13%/91.31% top-1 matching accuracy at 1/8 and 1/16.
The frozen representation is now injected through a separate residual
cost-volume branch after the historical first convolution of each flow head.
This preserves every 0729 VFI parameter and starts with only a 1.5e-4 mean
output perturbation.

Four-epoch controlled results (same 0729 initialization, seed, samples and LR):

| Injection | hard | Vimeo | xtrain | FlyingThings EPE |
|---|---:|---:|---:|---:|
| 1/16 only | 39.8517 | 36.4118 | 42.0057 | 11.7486 |
| 1/8 only | 39.8711 | 36.4071 | 42.0033 | 11.8142 |
| 1/16 + 1/8 | 39.8640 | 36.4154 | 41.9985 | 11.7732 |
| no-correlation control | 39.8381 | 36.4138 | 41.9732 | n/a |

The differences between correlation arms are small, but all improve hard and
xtrain over the matched control. The dual-scale arm is selected because it is
the only arm that also preserves/improves Vimeo, and its two adapters both
receive non-zero gradients without destabilizing the inherited model.

The long configuration remains
`train_config_0821_lc_pretrained_corr_weekend.yaml`, with experiment name
`0821_0729_lc_pretrained_corr_tuesday`. It runs 1050 epochs: 120 epochs of
10%-FlyingThings adapter anchoring, 630 epochs of target-only reconstruction,
and 300 epochs of low-LR hard-scene consolidation. At the measured 0.745
seconds per step with 401 steps per epoch, a Friday 16:00 launch is expected to
finish around Tuesday 08:00-10:00. Best xtrain checkpoints are evaluated every
10 epochs, so a late final-phase regression does not erase the best model.

### 2026-08-25 final result: frozen correspondence is not a production fix

The 1050-epoch run completed at hard 39.9563 dB, Vimeo 36.4501 dB and xtrain
41.8019 dB, with FlyingThings flow EPE 11.2392.  These aggregate values are
not a production pass: real clips still lose small foregrounds, corrupt fence
and grid texture, and blur detail.  The modest validation movement after a
very long run confirms that fixed descriptors plus residual cost-volume
adapters do not make correspondence a required part of final synthesis.

Do not extend or retune this frozen-adapter experiment.  It is retained only
as evidence that a correspondence prior can be numerically stable.

## 2026-08-25: PQMax all-pairs multi-field model

`train_config_0825_pqmax_amt_mamba_max.yaml` is the first implementation of
the materially different architecture called for above.  It is AMT/RAFT-
inspired, not a reproduction of either paper.  The model uses 0729 weights and
the FlyingThings correspondence checkpoint only as initialization; all 81.8M
parameters train jointly.

The required path is:

1. a full 1/8 all-pairs matrix produces three distinct global modes alongside
   the inherited Mamba flow;
2. all four bilateral fields undergo six shared ConvGRU updates, each using a
   newly evaluated bidirectional radius-4 local cost volume;
3. a 128-channel, ten-block 1/4 boundary network emits full-resolution flow,
   mask and field-selection corrections;
4. a low-temperature per-pixel selector combines the four warped candidates;
5. an eight-block full-resolution module restores high frequencies using the
   selected multi-field warp as a gated detail source.

The trainer directly supervises the best candidate (oracle), the winning-field
selector, selector entropy, field diversity, multi-scale high frequencies,
hard interpolation ROIs and edge-aligned warps.  These objectives are not
optional decoration: they prevent the exact collapse/inactivity observed in
P1-P3.  Diagnostics must show non-zero field spread, falling selector entropy,
an active detail gate and a visible ROI gain before this model is accepted.

The maximum schedule is 1560 epochs / 625,560 optimizer steps / 5,004,480
microbatches.  Phase 1 includes FlyingThings hard-cycle motion; phases 2-3 are
target-only at 512x896.  Use `launch_pqmax_max.sh`; do not pass `--resume` when
initializing from 0729.

### 32 GiB memory correction

The first real CUDA preflight initially failed at 384x704 batch 2 while
entering the legacy UNet: 30.49 GiB was already allocated and only 70 MiB
remained.  This was a genuine capacity limit, not allocator fragmentation.
The maximum-quality configuration now uses activation checkpointing for all
six PQ recurrent updates, the quarter-resolution boundary stack, the
full-resolution detail stack and the legacy UNet.  Phase 1 changes from batch
2 / accumulation 4 to batch 1 / accumulation 8, preserving effective batch 8
without removing fields, channels, blocks or the later 512x896 crop.  The
launcher preflight consequently checks 384x704x1 and 512x896x1 before starting.

## Deferred experiment: directional single-side hard-cycle supervision

Status on 2026-08-25: **accepted as a credible follow-up, deliberately deferred
to the next teacher-flow experiment**.  Do not modify the current PQMax run or
its configuration for this idea.

### Motivation

The current cache computes independent cycle confidence for the middle-to-
previous and middle-to-next flows, then stores only their minimum.  The
Dataset consequently exposes one shared valid channel for all four flow
components.  A pixel is therefore removed from both flow losses when only one
endpoint direction fails its cycle check.  This is conservative but discards
useful one-sided supervision around occlusions, image borders and small moving
objects.

Here, "single-side confirmed" has one precise meaning: one endpoint direction
has a complete forward/backward cycle check above threshold while the other
endpoint direction does not.  A forward MV with no reverse MV/check is raw,
unconfirmed supervision and is not included by this proposal.

### Required implementation

1. Generate a new two-channel cycle sidecar instead of overwriting the current
   one-channel `.cycle.npy` files:
   `[..., 0] = confidence_previous`,
   `[..., 1] = confidence_next`.
2. Change teacher flow representation from
   `[mv1_x, mv1_y, mv0_x, mv0_y, valid_shared]` to
   `[mv1_x, mv1_y, mv0_x, mv0_y, valid_previous, valid_next]` (or return an
   equivalent separate directional-valid tensor).
3. Split the endpoint-inside-image test by direction.  The current logical AND
   between both endpoints would otherwise erase a valid single-side sample
   before cycle confidence is applied.
4. Compute the two directional EPE terms with their own valid/confidence masks
   in training and validation.  Moving/static statistics and valid coverage
   must also be reported per direction.
5. Keep the existing both-side hard-cycle cache and training path unchanged so
   the historical baseline remains reproducible.

Never implement this by replacing `min(conf_previous, conf_next)` with `max`.
That would incorrectly supervise the failed direction whenever the other side
passes.

### Initial weighting and controlled ablation

Use identical initialization, seed, samples, crop schedule, optimizer steps
and image losses.  Only directional flow validity/weight may differ:

| Arm | Both directions pass | Only one direction passes | Failed direction |
|---|---:|---:|---:|
| A: current baseline | 1.0 / 1.0 | 0 / 0 | 0 |
| B: single-side 0.25 | 1.0 / 1.0 | 0.25 on confirmed side | 0 |
| C: single-side 0.50 | 1.0 / 1.0 | 0.50 on confirmed side | 0 |

Start with B.  C is justified only if B improves directional moving EPE and
real boundary behavior without degrading reconstruction.  Do not begin with
weight 1.0 because one-sided pixels are concentrated in a harder, shifted
occlusion/border distribution.

### Pre-training audit and acceptance criteria

Before training, report the pixel/sample ratios for both-pass, previous-only,
next-only and neither-pass, split by teacher dataset and moving/static region.
On sampled scenes, compare the confirmed side's warp error with raw MV and
zero flow to ensure the extra labels are genuinely informative.

Accept the strategy only when it provides all of the following against arm A:

- lower per-direction moving EPE on single-side and both-side validation;
- no meaningful regression in Hard ROI PSNR or edge error;
- visible improvement on small foregrounds/occlusion boundaries;
- no increase in fence/grid flicker, cracks or temporal instability.

This remains an early, low-weight motion auxiliary.  As established by the
previous teacher experiments, a lower synthetic EPE alone is not sufficient
evidence for production VFI quality.

## 2026-08-25: replace PQMax image-candidate fusion with single-flow matching

The first live PQMax run was stopped after its selector diagnostics showed a
clear collapse: normalized fusion entropy rose towards roughly 0.92 and the
four-way selection loss stayed near `ln(4)`, while the motion fields differed
by only about 1--2 pixels.  In that state the model was averaging nearly
equivalent warped candidates, recreating the soft-fusion blur it was intended
to solve.  More selector weight or lower fusion temperature would force a
choice without making the underlying correspondence more correct, so the run
was not continued.

The replacement experiment is `train_config_0825_single_match_flow.yaml`
(`exp_name=0825_single_match`).
Its structural boundary is deliberately narrower:

1. the trainable 1/8 correspondence descriptors build one dense all-pairs
   endpoint match;
2. match uniqueness, forward/backward cycle consistency and improvement over
   the current feature similarity form an analytic confidence;
3. a conservatively initialized learned gate applies one midpoint-preserving
   bilateral flow residual, followed by four shared local-correlation ConvGRU
   updates;
4. the corrected single field is directly included in multi-stage flow/warp
   supervision and passed through both inherited local IFBlocks;
5. image synthesis remains the inherited two-warp blend plus 0729 refiner.

There are no alternative images, per-pixel candidate selector, fusion entropy
objective, oracle candidate loss, PQMax detail restorer or multi-field boundary
head in this experiment.  Architecture-independent high-pass losses remain at
moderate weight because removing the failed selector should not also remove
direct supervision for fences and fine texture.

Acceptance requires more than final PSNR.  During bootstrap, the log fields
`single_match_proposal`, `single_match_applied`, `single_match_recurrent`,
`single_match_confidence`, `single_match_gate`, `single_match_mutual_error` and
`single_match_similarity_gain` must remain finite and non-degenerate.  Reject
the branch if the applied correction falls back towards zero, if it grows
without positive similarity gain, or if macro-average PSNR improves while
small objects/fences visibly become less stable.

On the 32 GiB GPU, the exact BF16 training graph (including teacher-flow loss)
passed at 320x576 and 336x608 with batch 4.  The larger phase peaked at 24.80
GiB allocated / 29.16 GiB reserved.  `train.py` defines every `crop_sizes`
entry as `[height, width, actual_batch]`; the first attempted live run retained
`2` in that third field and therefore really used batch 2 despite the top-level
`data.batch_size: 4`.  That run was stopped during epoch 0 and retained only as
an invalid launch record.  The corrected `0825_single_match` experiment sets
the third field to 4 in every phase and uses two micro-batches per optimizer
step (effective batch 8), replacing PQMax's slower batch 2 / accumulation 4
arrangement.
