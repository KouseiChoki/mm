# Ground-truth MV fine-tuning for IFBlock x2

This branch supports direct ground-truth motion-vector supervision for the two
local IFBlocks used after the coarse learned-feature flow heads.

## Relevant implementation

- `model/flow_estimation.py`
  - `LOCAL_CFG` defines two local IFBlocks: a half-resolution correction block
    followed by a full-resolution correction block.
  - Both refined flow predictions are appended to `flow_list`, allowing direct
    supervision at both local stages.
- `Trainer.py`
  - `set_trainable_scope("local_ifblock")` freezes every parameter except
    `local_block.*`, so both local IFBlocks are fine-tuned while the backbone,
    coarse flow heads and reconstruction refiner remain fixed.
  - The auxiliary flow objective supervises current-to-previous (`mv1`) and
    current-to-next (`mv0`) predictions at every configured flow stage.
- `kousei_dataset.py`
  - Loads crop-addressable `float16` MV caches and restores normalized vectors
    to pixel units.
  - Supports hard/soft forward-backward cycle confidence for excluding
    occlusion, out-of-frame and inconsistent teacher pixels.
- `data_prepare/build_mv_cache.py`
  - Converts legacy EXR MV into fast mmap NPY caches and generates cycle masks.
- `data_prepare/convert_native_teacher.py`
  - Converts native Sintel and FlyingThings3D optical flow into the same cache
    convention without an intermediate EXR copy.

## Flow convention

The teacher tensor is `[mv1_x, mv1_y, mv0_x, mv0_y, valid]` in pixel units:

- `mv1`: middle/current frame to the previous input frame.
- `mv0`: middle/current frame to the next input frame.
- `valid`: confidence/validity weight used by the auxiliary flow loss.

The stored cache is normalized by source width/height and is converted back to
pixel units by the dataset.

## IFBlock x2 fine-tuning command

Edit `data.root`, `data.lists_dir` and the checkpoint path for the target
machine, then run:

```bash
python -u train.py \
  --config train_config_teacher_flyingthings_cycle_ifblock.yaml \
  --restore_ckpt ckpt/0807_s2v3_official_tuesday/0807_s2v3_official_tuesday_320.pkl
```

The exact configuration used in the current experiment is
`train_config_teacher_flyingthings_cycle_ifblock.yaml`. It uses hard-cycle
filtered FlyingThings3D ground-truth flow with `flow_loss_weight: 0.0005` and
`trainable: local_ifblock`.

## Controlled experiments

The accompanying four-way experiment changes one factor at a time:

- `train_config_teacher_flyingthings_noflow.yaml`: same images/crops, no direct
  flow loss.
- `train_config_teacher_flyingthings_raw.yaml`: native GT flow without cycle
  filtering.
- `train_config_teacher_flyingthings_cycle.yaml`: hard-cycle GT flow, all model
  parameters trainable.
- `train_config_teacher_flyingthings_cycle_ifblock.yaml`: hard-cycle GT flow,
  only IFBlock x2 trainable.

Run all four sequentially with:

```bash
./launch_teacher_flyingthings_ablation.sh
```

Validation reports reconstruction PSNR plus overall, moving-region and
static-region flow EPE. Predicted and ground-truth MV visualizations are saved
under `record/<experiment>/mv_comparisons/` at the configured interval.
