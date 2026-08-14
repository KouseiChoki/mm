import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kousei_dataset import TierDataset


MODULE_PATH = Path(__file__).parents[1] / 'data_prepare' / 'build_mv_cache.py'
SPEC = importlib.util.spec_from_file_location('build_mv_cache', MODULE_PATH)
build_mv_cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_mv_cache)


def _normalized_constant(height, width, dx, dy):
    result = np.empty((height, width, 2), dtype=np.float32)
    result[..., 0] = dx / width
    result[..., 1] = dy / height
    return result


class TeacherMVCycleTest(unittest.TestCase):
    def test_accepts_inverse_translation(self):
        height, width = 12, 16
        forward = _normalized_constant(height, width, 2, 0)
        backward = _normalized_constant(height, width, -2, 0)

        confidence = build_mv_cache._direction_cycle_confidence(
            forward, backward, alpha=0.05, beta=1.0)

        self.assertTrue(np.allclose(confidence[:, :-2], 1.0))
        self.assertTrue(np.all(confidence[:, -2:] == 0.0))

    def test_rejects_wrong_reverse_flow(self):
        height, width = 12, 16
        forward = _normalized_constant(height, width, 3, 0)
        wrong_backward = _normalized_constant(height, width, 0, 0)

        confidence = build_mv_cache._direction_cycle_confidence(
            forward, wrong_backward, alpha=0.05, beta=1.0)

        self.assertTrue(np.all(confidence[:, :-3] < 0.5))

    def test_rejects_missing_adjacent_mv(self):
        forward = _normalized_constant(8, 10, 1, 0)
        confidence = build_mv_cache._direction_cycle_confidence(
            forward, None, alpha=0.05, beta=1.0)
        self.assertEqual(np.count_nonzero(confidence), 0)

    @staticmethod
    def _dataset_for_mode(mode):
        dataset = TierDataset.__new__(TierDataset)
        dataset.mv_sign = (1, 1)
        dataset.mv_cycle_confidence = mode
        dataset.mv_symmetry_confidence = False
        dataset.occ_alpha = 0.05
        dataset.occ_beta = 1.0
        return dataset

    def test_compact_cycle_mask_hard_threshold(self):
        normalized = np.zeros((2, 3, 4), dtype=np.float16)
        cycle = np.array([[255, 128, 127], [0, 255, 255]], dtype=np.uint8)
        flow = self._dataset_for_mode('hard')._normalized_mv_to_flow(
            normalized, source_h=2, source_w=3, cycle_confidence=cycle)
        expected = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
        self.assertTrue(np.array_equal(flow[..., 4], expected))

    def test_compact_cycle_mask_soft_weight(self):
        normalized = np.zeros((1, 2, 4), dtype=np.float16)
        cycle = np.array([[255, 64]], dtype=np.uint8)
        flow = self._dataset_for_mode('soft')._normalized_mv_to_flow(
            normalized, source_h=1, source_w=2, cycle_confidence=cycle)
        self.assertAlmostEqual(float(flow[0, 0, 4]), 1.0, places=6)
        self.assertAlmostEqual(float(flow[0, 1, 4]), 64 / 255, places=6)

    def test_on_the_fly_neighbor_crop(self):
        height, width = 6, 8
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / 'scene' / 'mv_cache_f16'
            cache_dir.mkdir(parents=True)
            previous = np.zeros((height, width, 4), dtype=np.float16)
            following = np.zeros((height, width, 4), dtype=np.float16)
            previous[..., 2] = -2 / width   # previous mv0 -> current
            following[..., 0] = -2 / width  # next mv1 -> current
            np.save(cache_dir / 'previous.npy', previous)
            np.save(cache_dir / 'next.npy', following)

            dataset = self._dataset_for_mode('hard')
            dataset.root = root
            dataset.mv_cache_dirname = 'mv_cache_f16'
            dataset.mv_cycle_cache_root = None
            normalized = np.zeros((height, width, 4), dtype=np.float16)
            normalized[..., 0] = 2 / width
            normalized[..., 2] = 2 / width
            confidence = dataset._cycle_from_neighbor_cache(
                {'rel': 'scene'}, normalized,
                Path('previous.png'), Path('next.png'),
                height, width, (0, 0, height, width))

            self.assertTrue(np.allclose(confidence[:, :-2], 1.0))
            self.assertTrue(np.all(confidence[:, -2:] == 0.0))


if __name__ == '__main__':
    unittest.main()
