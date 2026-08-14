import importlib.util
from pathlib import Path

import numpy as np


VFI_ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


converter = _load(
    'convert_native_teacher',
    VFI_ROOT / 'data_prepare' / 'convert_native_teacher.py')
build_lists = _load(
    'native_teacher_build_lists',
    VFI_ROOT / 'data_prepare' / 'build_lists.py')


def test_invert_forward_translation():
    flow = np.zeros((12, 16, 2), dtype=np.float32)
    flow[..., 0] = 2.0
    backward = converter.invert_forward_flow(flow)

    valid = np.isfinite(backward).all(axis=-1)
    assert valid[:, 2:].mean() > 0.95
    np.testing.assert_allclose(backward[valid, 0], -2.0, atol=1e-4)
    np.testing.assert_allclose(backward[valid, 1], 0.0, atol=1e-4)
    assert not valid[:, :2].any()


def test_normalize_pair_keeps_direction_validity_independent():
    past = np.zeros((4, 6, 2), dtype=np.float32)
    future = np.ones((4, 6, 2), dtype=np.float32)
    past[0] = np.nan
    cache = converter._normalize_pair(past, future)

    assert cache.dtype == np.float16
    assert np.isnan(cache[0, :, :2]).all()
    assert np.isfinite(cache[0, :, 2:]).all()


def test_native_cache_scene_scan_and_grouping(tmp_path):
    scene = tmp_path / 'teacher' / 'Sintel' / 'clean' / 'alley_1'
    (scene / 'image').mkdir(parents=True)
    (scene / 'mv_cache_f16').mkdir()
    for frame in range(1, 6):
        (scene / 'image' / f'frame_{frame:04d}.png').write_bytes(b'png')
        np.save(
            scene / 'mv_cache_f16' / f'frame_{frame:04d}.npy',
            np.zeros((2, 2, 4), dtype=np.float16))
        np.save(
            scene / 'mv_cache_f16' / f'frame_{frame:04d}.motion.npy',
            np.zeros((1, 1), dtype=np.float16))

    broken = []
    rows = build_lists.scan_teacher(
        tmp_path, min_frames=5, allow_gaps=False, pair_ratio=0.9,
        broken=broken, teacher_fps=None)
    assert not broken
    assert len(rows) == 1
    assert rows[0]['has_mv'] == 1
    assert rows[0]['n'] == 5

    clean = {'rel': 'teacher/Sintel/clean/alley_1'}
    final = {'rel': 'teacher/Sintel/final/alley_1'}
    assert build_lists.teacher_group_key(clean) == build_lists.teacher_group_key(final)

    flying = [
        {'rel': 'teacher/FlyingThings3D/TRAIN/clean/A/0000/left'},
        {'rel': 'teacher/FlyingThings3D/TRAIN/final/A/0000/right'},
    ]
    assert (build_lists.teacher_group_key(flying[0])
            == build_lists.teacher_group_key(flying[1]))


def test_teacher_split_is_stratified_by_source():
    rows = []
    for scene in ('alley_1', 'alley_2'):
        for render_pass in ('clean', 'final'):
            rows.append({
                'rel': f'teacher/Sintel/{render_pass}/{scene}',
                'n': 10,
            })
    for sequence in ('0000', '0001'):
        for render_pass in ('clean', 'final'):
            for view in ('left', 'right'):
                rows.append({
                    'rel': (
                        'teacher/FlyingThings3D/TRAIN/'
                        f'{render_pass}/A/{sequence}/{view}'),
                    'n': 10,
                })

    train, val = build_lists.split_teacher_rows(rows, 0.005, seed=123)
    val_sources = {build_lists.teacher_source_key(row) for row in val}
    assert val_sources == {
        'teacher/Sintel', 'teacher/FlyingThings3D'}
    train_groups = {build_lists.teacher_group_key(row) for row in train}
    val_groups = {build_lists.teacher_group_key(row) for row in val}
    assert not train_groups & val_groups


def test_teacher_source_tier_names():
    expected = {
        'teacher/Unreal/finalScene/24fps': 'teacher_unreal',
        'teacher/Spring/0001/left': 'teacher_spring',
        'teacher/Sintel/clean/alley_1': 'teacher_sintel',
        ('teacher/FlyingThings3D/TRAIN/clean/A/0000/left'):
            'teacher_flyingthings',
    }
    for rel, tier in expected.items():
        assert build_lists.teacher_tier_name({'rel': rel}) == tier
