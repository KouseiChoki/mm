import sys
import unittest
from pathlib import Path

import torch


VFI_ROOT = Path(__file__).resolve().parents[1]
if str(VFI_ROOT) not in sys.path:
    sys.path.insert(0, str(VFI_ROOT))

from Trainer import Model
from model.loss import EdgeCharbonnierLoss
import train as train_module


class HardRoiMaskTest(unittest.TestCase):
    def test_mask_uses_timestamp_linear_residual_and_exact_top_ratio(self):
        img0 = torch.zeros(1, 3, 4, 4)
        img1 = torch.ones_like(img0)
        gt = torch.full_like(img0, 0.25)
        gt[:, :, 1:3, 2:4] += 0.5
        imgs = torch.cat((img0, img1), dim=1)

        mask = Model.hard_roi_mask(
            imgs, gt, torch.tensor([0.25]), ratio=0.25,
            min_residual=0.04, dilation=0)

        expected = torch.zeros(1, 1, 4, 4)
        expected[:, :, 1:3, 2:4] = 1.0
        self.assertTrue(torch.equal(mask, expected))
        self.assertFalse(mask.requires_grad)

    def test_threshold_can_return_empty_mask(self):
        imgs = torch.zeros(2, 6, 5, 7)
        gt = torch.full((2, 3, 5, 7), 0.01)
        mask = Model.hard_roi_mask(
            imgs, gt, 0.5, ratio=0.2,
            min_residual=0.04, dilation=0)
        self.assertEqual(mask.sum().item(), 0.0)

    def test_dilation_expands_one_pixel_by_configured_radius(self):
        imgs = torch.zeros(1, 6, 7, 7)
        gt = torch.zeros(1, 3, 7, 7)
        gt[:, :, 3, 3] = 1.0
        mask = Model.hard_roi_mask(
            imgs, gt, 0.5, ratio=1.0 / 49.0,
            min_residual=0.04, dilation=3)
        self.assertEqual(mask.sum().item(), 9.0)
        self.assertTrue(torch.all(mask[:, :, 2:5, 2:5] == 1))

    def test_even_dilation_kernel_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'must be 0 or odd'):
            Model.hard_roi_mask(
                torch.zeros(1, 6, 4, 4), torch.zeros(1, 3, 4, 4),
                0.5, ratio=0.1, min_residual=0.0, dilation=2)


class MaskedCharbonnierTest(unittest.TestCase):
    def test_gradient_is_confined_to_selected_pixels(self):
        prediction = torch.zeros(1, 3, 3, 4, requires_grad=True)
        target = torch.ones_like(prediction)
        mask = torch.zeros(1, 1, 3, 4)
        mask[:, :, 1, 2] = 1.0

        loss = Model._masked_charbonnier(
            prediction, target, mask, eps=1e-3)
        loss.backward()

        selected = prediction.grad[:, :, 1, 2]
        outside = prediction.grad.clone()
        outside[:, :, 1, 2] = 0.0
        self.assertGreater(selected.abs().sum().item(), 0.0)
        self.assertEqual(outside.abs().sum().item(), 0.0)


class _DummyWriter:
    def __init__(self):
        self.tags = {}

    def add_scalar(self, tag, value, step):
        self.tags[tag] = (value, step)


class _DummyEvaluationModel:
    def __init__(self):
        self.net = self
        self.local = True
        self.hard_roi_ratio = 0.25
        self.hard_roi_min_residual = 0.04
        self.hard_roi_dilation = 0
        self.edge_loss = EdgeCharbonnierLoss()

    def eval(self):
        return None

    def train(self):
        return None

    @staticmethod
    def pad_to_multiple(images, _multiple):
        return images, (0, 0)

    @staticmethod
    def unpad(value, _right, _bottom):
        return value

    def __call__(self, images, **_kwargs):
        prediction = images[:, :3]
        return [], None, None, None, None, [prediction], prediction

    hard_roi_mask = staticmethod(Model.hard_roi_mask)


class HardRoiEvaluationTest(unittest.TestCase):
    def test_evaluate_publishes_final_and_warp_roi_metrics(self):
        frames = torch.zeros(1, 9, 8, 8, dtype=torch.uint8)
        frames[:, 6:9, 2:6, 2:6] = 255
        sample = (
            frames, torch.full((1, 1, 1, 1), 0.5),
            torch.zeros(1, 5, 8, 8), torch.zeros(1))
        writer = _DummyWriter()
        old_device = train_module.device
        train_module.device = torch.device('cpu')
        try:
            metrics = train_module.evaluate(
                _DummyEvaluationModel(), {'xtrain': [sample]}, 1,
                writer, use_amp=False, amp_dtype=torch.bfloat16)
        finally:
            train_module.device = old_device

        for key in (
                'xtrain/roi_psnr', 'xtrain/warp_roi_psnr',
                'xtrain/roi_final_gain_db', 'xtrain/roi_l1',
                'xtrain/roi_edge_error', 'xtrain/roi_area'):
            self.assertIn(key, metrics)
        self.assertAlmostEqual(metrics['xtrain/roi_area'], 0.25)
        self.assertAlmostEqual(metrics['xtrain/roi_final_gain_db'], 0.0)
        self.assertIn('val/xtrain_roi_psnr', writer.tags)


if __name__ == '__main__':
    unittest.main()
