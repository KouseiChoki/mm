import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from kousei_dataset import TierDataset
from model.loss import EdgeCharbonnierLoss


class InterpolationAwareCropTest(unittest.TestCase):
    def test_small_interpolation_residual_is_kept_in_crop(self):
        dataset = TierDataset.__new__(TierDataset)
        dataset.interpolation_residual_threshold = 0.04
        dataset.small_motion_min_pixels = 8
        dataset.small_motion_max_ratio = 0.1
        dataset.motion_crop_jitter = 0.0

        height, width = 64, 96
        img0 = np.zeros((height, width, 3), dtype=np.uint8)
        img1 = np.zeros_like(img0)
        gt = np.zeros_like(img0)
        gt[26:30, 72:76] = 255

        random.seed(1234)
        top, left = dataset._interpolation_crop_origin(
            img0, gt, img1, 0.5, 32, 32, height, width)
        self.assertLessEqual(top, 26)
        self.assertGreaterEqual(top + 32, 30)
        self.assertLessEqual(left, 72)
        self.assertGreaterEqual(left + 32, 76)


class EdgeCharbonnierLossTest(unittest.TestCase):
    def test_identical_image_is_epsilon_floor(self):
        loss = EdgeCharbonnierLoss(eps=1e-3)
        image = torch.rand(2, 3, 32, 48)
        value = loss(image, image)
        self.assertAlmostEqual(value.item(), 1e-3, places=6)

    def test_missing_edge_has_larger_loss_and_accepts_weight(self):
        loss = EdgeCharbonnierLoss(eps=1e-3)
        target = torch.zeros(1, 3, 32, 48)
        target[..., :, 24:] = 1.0
        prediction = torch.zeros_like(target)
        weight = torch.ones(1, 1, 32, 48)
        value = loss(prediction, target, weight)
        self.assertGreater(value.item(), 0.02)


if __name__ == '__main__':
    unittest.main()
