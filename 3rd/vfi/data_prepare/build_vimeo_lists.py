#!/usr/bin/env python3
"""将官方 Vimeo90K-Triplet split 转成当前 TierDataset 清单。

标准输入结构::

    vimeo_triplet/
      tri_trainlist.txt
      tri_testlist.txt
      sequences/00001/0001/{im1.png,im2.png,im3.png}

脚本只生成 txt/json 索引，不复制、不改名、不创建数万个软链接。

用法::

    python data_prepare/build_vimeo_lists.py \
        --root /data/vimeo_triplet
"""

import argparse
import json
from pathlib import Path, PurePosixPath


FRAME_NAMES = ('im1.png', 'im2.png', 'im3.png')


def normalized_scene(line: str) -> str:
    """规范官方split中的相对目录，并拒绝越过数据根目录的路径。"""
    value = line.strip().replace('\\', '/')
    if not value or value.startswith('#'):
        return ''
    path = PurePosixPath(value)
    parts = list(path.parts)
    if parts and parts[0] == 'sequences':
        parts = parts[1:]
    if (not parts or path.is_absolute()
            or any(part in ('', '.', '..') for part in parts)):
        raise ValueError(f'非法Vimeo场景路径: {line!r}')
    return PurePosixPath(*parts).as_posix()


def read_split(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Vimeo split不存在: {path}')
    scenes = []
    seen = set()
    for line_number, line in enumerate(
            path.read_text(encoding='utf-8-sig').splitlines(), 1):
        scene = normalized_scene(line)
        if not scene:
            continue
        if scene in seen:
            raise ValueError(
                f'{path}:{line_number} 出现重复场景: {scene}')
        seen.add(scene)
        scenes.append(scene)
    if not scenes:
        raise ValueError(f'Vimeo split为空: {path}')
    return scenes


def validate_scenes(root: Path, scenes, split_name: str):
    """校验三帧齐全，返回当前数据集清单使用的scene相对路径。"""
    rows = []
    missing = []
    for scene in scenes:
        scene_dir = root / 'sequences' / scene
        absent = [
            name for name in FRAME_NAMES
            if not (scene_dir / name).is_file()]
        if absent:
            missing.append(f'{scene}: {", ".join(absent)}')
            continue
        rows.append({
            'rel': (PurePosixPath('sequences') / scene).as_posix(),
            'frames': list(FRAME_NAMES),
        })
    if missing:
        preview = '\n  '.join(missing[:10])
        suffix = (
            f'\n  ...另有{len(missing) - 10}项'
            if len(missing) > 10 else '')
        raise FileNotFoundError(
            f'{split_name} 中 {len(missing)} 个场景缺帧；'
            f'请确认--root指向包含sequences的vimeo_triplet目录:\n'
            f'  {preview}{suffix}')
    return rows


def write_tier_list(output: Path, filename: str, rows, tier: str):
    output.mkdir(parents=True, exist_ok=True)
    txt_path = output / filename
    txt_path.write_text(
        ''.join(
            f'{row["rel"]}\t3\t{tier}\t0\t0\n'
            for row in rows),
        encoding='utf-8')
    json_path = txt_path.with_suffix('.frames.json')
    json_path.write_text(
        json.dumps(
            {row['rel']: row['frames'] for row in rows},
            ensure_ascii=False),
        encoding='utf-8')
    print(f'[vimeo] {txt_path}: {len(rows)} triplets')
    print(f'[vimeo] {json_path}: frame index ready')


def main():
    parser = argparse.ArgumentParser(
        description='Vimeo90K-Triplet → Kousei TierDataset清单')
    parser.add_argument(
        '--root', required=True, type=Path,
        help='包含sequences/和tri_*list.txt的vimeo_triplet目录')
    parser.add_argument(
        '--output', type=Path, default=None,
        help='输出目录，默认<root>/lists')
    parser.add_argument(
        '--train-list', default='tri_trainlist.txt',
        help='相对--root的官方训练split文件名')
    parser.add_argument(
        '--test-list', default='tri_testlist.txt',
        help='相对--root的官方测试split文件名')
    parser.add_argument(
        '--tier', default='vimeo',
        help='训练tier名称，必须与YAML phases.ratios一致')
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not (root / 'sequences').is_dir():
        raise FileNotFoundError(
            f'未找到 {root / "sequences"}；'
            '--root应直接指向vimeo_triplet目录')
    output = (
        args.output.expanduser().resolve()
        if args.output is not None else root / 'lists')

    train_scenes = read_split(root / args.train_list)
    test_scenes = read_split(root / args.test_list)
    overlap = set(train_scenes) & set(test_scenes)
    if overlap:
        preview = ', '.join(sorted(overlap)[:10])
        raise ValueError(
            f'官方train/test split存在{len(overlap)}个重叠场景: {preview}')

    train_rows = validate_scenes(root, train_scenes, 'train')
    test_rows = validate_scenes(root, test_scenes, 'test')
    write_tier_list(output, f'{args.tier}_train.txt', train_rows, args.tier)
    write_tier_list(output, f'{args.tier}_val.txt', test_rows, args.tier)
    print(
        f'[vimeo] 完成: train={len(train_rows)}, val={len(test_rows)}, '
        '未移动任何图像')


if __name__ == '__main__':
    main()
