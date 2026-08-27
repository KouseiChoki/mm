import unittest
from pathlib import Path

import torch
import yaml

from model.flow_estimation import Head, LocalCorrelationVolume


VFI_ROOT = Path(__file__).resolve().parents[1]
CONFIG = VFI_ROOT / 'train_config_0820_lc_pyramid_correlation.yaml'


class LocalCorrelationVolumeTest(unittest.TestCase):
    def test_known_horizontal_match_has_expected_peak(self):
        torch.manual_seed(23)
        module = LocalCorrelationVolume(
            feature_channels=8, radius=2, output_channels=4,
            temperature=0.03)
        feature0 = torch.randn(1, 8, 7, 9)
        feature1 = torch.roll(feature0, shifts=1, dims=3)
        captured = {}

        def capture_input(_module, args):
            captured['volume'] = args[0].detach()

        handle = module.encoder[0].register_forward_pre_hook(capture_input)
        module(feature0, feature1, flow=None)
        handle.remove()

        volume = captured['volume']
        kernel = 2 * module.radius + 1
        expected_channel = module.radius * kernel + module.radius + 1
        interior_peak = volume[:, :, 2:-2, 2:-2].argmax(dim=1)
        expected = torch.full_like(interior_peak, expected_channel)
        self.assertTrue(torch.equal(interior_peak, expected))

    def test_flow_aligned_volume_is_finite_and_differentiable(self):
        torch.manual_seed(29)
        module = LocalCorrelationVolume(
            feature_channels=8, radius=2, output_channels=6)
        feature0 = torch.randn(2, 8, 5, 7, requires_grad=True)
        feature1 = torch.randn(2, 8, 5, 7, requires_grad=True)
        flow = torch.randn(2, 4, 20, 28, requires_grad=True)

        output = module(feature0, feature1, flow)
        output.square().mean().backward()

        self.assertEqual(output.shape, (2, 6, 5, 7))
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(feature0.grad)
        self.assertIsNotNone(feature1.grad)
        self.assertIsNotNone(flow.grad)
        self.assertGreater(float(module.last_peak_probability), 0.0)
        self.assertGreaterEqual(float(module.last_normalized_entropy), 0.0)
        self.assertLessEqual(float(module.last_normalized_entropy), 1.0)

    def test_integrated_head_preserves_prediction_shapes(self):
        torch.manual_seed(31)
        head = Head(
            in_planes=64, scale=4, c=16, in_else=7,
            correlation_radius=2, correlation_channels=8)
        endpoint_features = torch.randn(1, 128, 8, 12)
        image_inputs = torch.randn(1, 7, 32, 48)

        flow, mask = head(endpoint_features, image_inputs, flow=None)

        self.assertEqual(flow.shape, (1, 4, 32, 48))
        self.assertEqual(mask.shape, (1, 1, 32, 48))
        self.assertIsNotNone(head.correlation.last_feature_abs)

    def test_disabled_head_keeps_legacy_input_width(self):
        head = Head(in_planes=64, scale=4, c=16, in_else=7)
        first_conv = head.conv[0][0]

        self.assertIsNone(head.correlation)
        self.assertEqual(first_conv.in_channels, 64 * 2 // 16 + 7)


class PyramidCorrelationConfigTest(unittest.TestCase):
    def test_config_has_one_coarse_to_fine_radius_per_stage(self):
        config = yaml.safe_load(CONFIG.read_text())
        model = config['model']

        self.assertTrue(model['pyramid_correlation'])
        self.assertEqual(
            len(model['pyramid_correlation_radii']),
            model['flow_num_stages'])
        self.assertEqual(model['pyramid_correlation_radii'], [6, 4, 3])
        self.assertFalse(model['sparse_matching'])
        self.assertEqual(model['flow_loss_weight'], 0.0)

    def test_first_ten_epochs_keep_the_existing_control_lr_horizon(self):
        config = yaml.safe_load(CONFIG.read_text())
        phase = config['phases'][0]

        self.assertEqual(phase['steps_per_epoch'], 401)
        self.assertEqual(phase['lr_total_steps'], 10 * 401)
        self.assertGreaterEqual(phase['epochs'], 10)


if __name__ == '__main__':
    unittest.main()
