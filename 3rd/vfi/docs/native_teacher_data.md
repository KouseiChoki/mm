# Native Sintel / FlyingThings3D teacher data

## Source and target

- Source: `/run/media/zhenying/Spica/optical_flow`
- Target: `/data/vfi_database/teacher`
- Sintel: all 23 official training sequences, `clean` and `final`
- FlyingThings3D: official `TRAIN` only, `clean` and `final`, left and right
- FlyingThings3D `TEST` stays on the source disk and is not mixed into training.

The conversion command is:

```bash
cd /home/zhenying/qhong/repo/mm/3rd/vfi
/home/zhenying/qhong/envs/anaconda3/envs/vfi/bin/python \
  data_prepare/convert_native_teacher.py \
  --source /run/media/zhenying/Spica/optical_flow \
  --root /data/vfi_database \
  --datasets sintel flyingthings \
  --passes clean final \
  --flying_split TRAIN \
  --workers 8
```

The command is restart-safe. Existing complete image/cache files are skipped;
use `--overwrite` only when the conversion logic has intentionally changed.

## Flow convention

The cache is `float16 [H,W,4]` in this order:

1. `mv1_x`: middle to previous, divided by image width
2. `mv1_y`: middle to previous, divided by image height
3. `mv0_x`: middle to next, divided by image width
4. `mv0_y`: middle to next, divided by image height

FlyingThings3D maps native `into_past` to `mv1` and `into_future` to `mv0`.
The PFM reader performs the required vertical file flip; the vector Y sign is
not negated. This was verified by remapping the neighbouring RGB frames.

Sintel's native `.flo` is current-to-next and maps directly to `mv0`. Sintel
does not publish the reverse direction, so `mv1` is conservatively inverted
from the preceding native forward flow. Invalid/occluded source pixels,
disocclusion holes, ambiguous many-to-one splats and failed cycle pixels are
stored as NaN and therefore excluded from flow supervision.

Each cache also has:

- `*.motion.npy`: low-resolution motion residual for motion-aware crop choice
- `*.cycle.npy`: compact forward/backward confidence used by hard/soft cycle
  filtering

Clean/final render passes share cache files with hard links, so flow storage is
not duplicated.

## List generation

After conversion:

```bash
cd /home/zhenying/qhong/repo/mm/3rd/vfi
/home/zhenying/qhong/envs/anaconda3/envs/vfi/bin/python \
  data_prepare/build_lists.py \
  --root /data/vfi_database \
  --max_framestep 2
```

`build_lists.py` recognizes both legacy EXR teacher scenes and native cache
scenes. It groups Sintel clean/final together, and groups FlyingThings3D
clean/final/left/right by source sequence, preventing train/validation leakage.
Teacher validation is stratified by source dataset, so Unreal, Spring, Sintel
and FlyingThings3D each retain validation content instead of small sources being
lost in a single global random draw.

The aggregate `teacher_train.txt` / `teacher_val.txt` remain available for old
configs. The same leak-free split is also exposed as four independently
sampleable tiers:

- `teacher_unreal_{train,val}.txt`
- `teacher_spring_{train,val}.txt`
- `teacher_sintel_{train,val}.txt`
- `teacher_flyingthings_{train,val}.txt`

Training configs that use these native scenes should set:

```yaml
data:
  mv_cache_dirname: mv_cache_f16
  mv_cache_required: true
  mv_cache_preview_stride: 4
```
