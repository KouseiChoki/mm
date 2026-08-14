from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from file_utils import read as read_file
from train import MVComparisonDumper, build_record_paths


class TrainingRecordTest(unittest.TestCase):
    def test_record_paths_are_grouped_by_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = build_record_paths(
                'experiment_a', {
                    'record_root': 'record',
                    'dump_dir': 'anomalies',
                }, project_dir=directory)
            root = Path(directory) / 'record' / 'experiment_a'
            self.assertEqual(Path(paths['root']), root)
            self.assertEqual(Path(paths['tensorboard']), root / 'tensorboard')
            self.assertEqual(Path(paths['anomalies']), root / 'anomalies')
            self.assertEqual(Path(paths['mv_comparisons']), root / 'mv_comparisons')

    def test_record_subdir_cannot_escape_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                build_record_paths(
                    'experiment_a', {'dump_dir': '../outside'},
                    project_dir=directory)

    def test_mv_dump_schedule(self):
        dumper = MVComparisonDumper('/tmp/not-used', every_epochs=5, max_samples=2)
        self.assertFalse(dumper.due(4))
        self.assertTrue(dumper.due(5))
        self.assertTrue(dumper.due(10))

    def test_mv_dump_contains_exact_arrays_and_visuals(self):
        height, width = 4, 6
        with tempfile.TemporaryDirectory() as directory:
            dumper = MVComparisonDumper(directory, every_epochs=1, max_samples=1)
            frames = torch.zeros(9, height, width)
            pred_frame = torch.full((3, height, width), 0.5)
            pred_mv = torch.zeros(4, height, width)
            gt_mv = torch.zeros(5, height, width)
            pred_mv[0] = 2.0
            gt_mv[0] = 1.0
            pred_mv[2] = -1.0
            gt_mv[2] = -1.0
            gt_mv[4] = 1.0
            gt_mv[4, 0] = 0.0

            output = Path(dumper.dump_sample(
                epoch=5, sample_index=3, frames=frames,
                pred_frame=pred_frame, flow_pred=pred_mv, flow_gt=gt_mv,
                timestep=0.5, metadata={'rel': 'teacher/scene'}))

            arrays = np.load(output / 'mv.npz')
            np.testing.assert_array_equal(arrays['pred_mv1'], pred_mv[:2].numpy())
            np.testing.assert_array_equal(arrays['gt_mv0'], gt_mv[2:4].numpy())
            self.assertTrue((output / 'pred_mv1_to_previous.png').is_file())
            self.assertTrue((output / 'gt_mv0_to_next.png').is_file())
            self.assertTrue((output / 'error_mv1_to_previous.png').is_file())
            self.assertTrue((output / 'valid.png').is_file())
            pred_mv1_exr = read_file(str(output / 'pred_mv1.exr'), type='flo')
            gt_mv0_exr = read_file(str(output / 'gt_mv0.exr'), type='flo')
            np.testing.assert_allclose(
                pred_mv1_exr[..., 0], pred_mv[0].numpy() / width, atol=1e-3)
            np.testing.assert_allclose(
                gt_mv0_exr[..., 0], gt_mv[2].numpy() / width, atol=1e-3)
            np.testing.assert_array_equal(pred_mv1_exr[..., 2:], 0)
            with open(output / 'meta.yaml') as handle:
                metadata = yaml.safe_load(handle)
            self.assertAlmostEqual(metadata['mv1_epe'], 1.0)
            self.assertAlmostEqual(metadata['mv0_epe'], 0.0)
            self.assertEqual(metadata['metadata']['rel'], 'teacher/scene')


if __name__ == '__main__':
    unittest.main()
