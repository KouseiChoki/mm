# Explicit matching preflight (2026-08-19)

## Experiment contract

Both arms restore `0729_lc_v3s2_800.pkl` without `--resume`.  They use the
same seed, data, crop, optimizer, phase and validation lists.  The only model
difference is the explicit-matching structure.

- Control: `train_config_0729_lc_explicit_match_control.yaml`
- Matching: `train_config_0729_lc_explicit_match_gmflow.yaml`
- Fixed real gate: `eval_lists/explicit_matching_real_scenes.txt`
- Common evaluator: `run_explicit_matching_eval.sh`

Teacher flow, CAUN, PerVFI and multi-hypothesis are disabled in both arms.

## Matching implementation

- Frozen GMFlow Sintel backbone + six-layer feature transformer at 1/8 scale.
- Select 5% high-disagreement target positions, bounded to 64-256 queries.
- Match each query globally against all frame-1 keys.
- Filter with top-1 margin and backward consistency.
- Feed a midpoint-preserving proposal to a zero-initialized adaptive merger.
- Train the original VFI backbone, flow heads, local IFBlocks, mask and refiner
  jointly; the GMFlow representation remains frozen.

The GMFlow source subset is vendored from:

- upstream: <https://github.com/haofeixu/gmflow>
- commit: `b5123431164d01ec14526a1c3d22218aecb62024`
- license: Apache-2.0

The SGM-VFI design reference is:

- upstream: <https://github.com/MCG-NJU/SGM-VFI>
- inspected commit: `40b5e74c7d0cc44a21683d05298e3e0b70d09d76`
- license: Apache-2.0

## Fixed artifacts

| Artifact | SHA256 |
|---|---|
| `gmflow_sintel-0c07dcb3.pth` | `0c07dcb35770464f38a5ff4de18c04177b242dc5de8cd2068adf46f3d4fe193a` |
| `0729_lc_v3s2_800.pkl` | `345b34025661232273f44efbb76816b55e60739fefd78603a5ac69efc162d5ab` |
| `0729_lc_v3s2/model.yaml` | `4ce85a24eac5b28a33b33c178ee884cc9cd9f5fea76949d8de83aa2ed1bc1d50` |
| `xtrain_train.txt` | `e63ad2e9eaaad3a64056be184e5caf1ae2d2c6567965c90519ac3898cb259d39` |
| `xtrain_val.txt` | `9c0cae4c9f2c4a8ab2057830bce1f917d87fb6d73d09faebbcd6205e694e16af` |
| `vimeo_train.txt` | `95dadc14a75468eca58d9e9722dd324ef9eaf79c2157d357d404a4f5b2a7aaae` |
| `vimeo_val.txt` | `91a6af06005039e3f204452c5c54cc85446df3d37fa12824de5f0160e55c3ab6` |
| fixed real-scene list | `ce86ffd9594af830d2e4078496776ba351dd660395712ffe8fbcd6a8fc2c518e` |

List sizes: X4K train/val = 4287/18 scenes; Vimeo train/val = 72822/369
scenes.  The real gate contains 16 curated sequences and evaluates the first
adjacent pair of each sequence for the short checkpoint gate.

## mms3 smoke result

Environment: Python 3.10.20, PyTorch 2.11.0+cu130, CUDA 13.0, RTX 5090.

- shape: batch 4, `384x704`, BF16;
- one training step: 7.69 seconds;
- peak allocated/reserved: 13.41/16.25 GiB;
- zero-initialized matching residual before training: exactly `0.0 px`;
- normal Mamba Triton repeat noise: mean `2.84e-5` RGB;
- matching-enabled mean difference before training: `3.15e-5` RGB, with
  exactly zero matching residual, so it is within the backend's run-to-run
  numerical variation rather than a structural checkpoint change.

Five fixed X4K validation scenes at endpoint gap 64 produced:

- mean selected-query confidence: 0.1313;
- mean backward-consistent ratio: 38.52%;
- mean queries above confidence 0.25: 21.61%;
- mean top-1 feature-similarity gain: 0.0937;
- top-1 match improved the current correspondence on 87.03% of queries;
- mean proposed displacement: 19.65 px.

The confidence scale was calibrated to `0.01`.  The old `0.05` retained only
2.75% of selected queries above confidence 0.25 and would risk repeating the
inactive frozen-feature P2.

The full machine-readable report is written to
`record/explicit_matching_preflight/mms3_batch4_384x704.json` and is not
required in Git.

## mms2 synchronization and smoke result

The experiment branch is installed as an isolated worktree at
`/home/zhenying/qhong/repo/mm-explicit`, so the existing
`pervfi-local-ablation` worktree and its uncommitted files are untouched.
The baseline checkpoint, GMFlow checkpoint and four train/validation lists
match the hashes above.  The fixed real test set also matches mms3 at 1659
files and 19,698,369,552 bytes; an rsync dry run reports no differences.

Environment: Python 3.10.20, PyTorch 2.11.0+cu130, CUDA 13.0, RTX 5090.

- all 33 unit/config tests pass;
- shape: batch 4, `384x704`, BF16;
- one training step: 8.09 seconds;
- peak allocated/reserved: 13.37/14.17 GiB;
- zero-initialized matching residual before training: exactly `0.0 px`;
- mean selected-query confidence on the same five X4K scenes: 0.1336;
- mean backward-consistent ratio: 38.87%;
- mean queries above confidence 0.25: 22.18%;
- mean feature-similarity gain: 0.0938;
- improved-query ratio: 86.72%;
- mean proposed displacement: 20.08 px.

The full report is written to
`record/explicit_matching_preflight/mms2_batch4_384x704.json` in the mms2
worktree.  The fixed real-evaluation dry run resolves all 16 sequences and
1447 adjacent pairs.

Recommended assignment: run the control on mms2 and the matching arm on mms3.
Both use the same GPU model, software versions, checkpoint and data hashes.

## Gate criteria

Stop the matching arm instead of extending it if any condition holds:

1. selected-region applied correction remains below `0.02 px` after epoch 2;
2. matching confidence or backward-consistent support collapses toward zero;
3. X4K PSNR/SSIM does not beat the paired control by epoch 5;
4. any fixed real clip regresses in fence/grid stability, foreground boundary
   continuity or temporal flicker;
5. latency or memory exceeds the production budget.

Only after the 10-epoch gate passes should both arms be extended with an
identical schedule.  FlyingThings teacher flow must remain disabled for this
first causal test.

## Commands after both machines are ready

Control:

```bash
python train.py \
  --config train_config_0729_lc_explicit_match_control.yaml \
  --restore_ckpt ckpt/0729_lc_v3s2/0729_lc_v3s2_800.pkl
```

Matching:

```bash
python train.py \
  --config train_config_0729_lc_explicit_match_gmflow.yaml \
  --restore_ckpt ckpt/0729_lc_v3s2/0729_lc_v3s2_800.pkl
```
