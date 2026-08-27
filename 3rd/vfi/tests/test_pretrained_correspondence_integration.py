import unittest
from pathlib import Path

import torch
import yaml

from model.flow_estimation import Head


ROOT = Path(__file__).resolve().parents[1]


class PretrainedCorrespondenceHeadTest(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(7)
        motion = torch.randn(1, 64, 4, 4)
        images = torch.randn(1, 7, 64, 64)
        external = (torch.randn(1, 128, 4, 4),
                    torch.randn(1, 128, 4, 4))
        return motion, images, external

    def test_adapter_does_not_change_legacy_input_width(self):
        legacy = Head(32, 16, 8, in_else=7)
        injected = Head(
            32, 16, 8, in_else=7, external_correlation_radius=1,
            external_correlation_init_scale=0.0)
        self.assertEqual(
            legacy.conv[0][0].weight.shape,
            injected.conv[0][0].weight.shape)

        injected.feature_transform.load_state_dict(
            legacy.feature_transform.state_dict())
        injected.conv.load_state_dict(legacy.conv.state_dict())
        motion, images, external = self._inputs()
        legacy_flow, legacy_mask = legacy(motion, images, None)
        flow, mask = injected(
            motion, images, None, external_features=external)
        torch.testing.assert_close(flow, legacy_flow, rtol=0, atol=0)
        torch.testing.assert_close(mask, legacy_mask, rtol=0, atol=0)

    def test_adapter_and_scale_receive_gradient(self):
        head = Head(
            32, 16, 8, in_else=7, external_correlation_radius=1,
            external_correlation_init_scale=0.01)
        motion, images, external = self._inputs()
        flow, mask = head(
            motion, images, None, external_features=external)
        (flow.square().mean() + mask.square().mean()).backward()
        self.assertGreater(
            float(head.external_correlation_scale.grad.abs()), 0.0)
        gradients = [
            parameter.grad for parameter in head.external_correlation.parameters()
            if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertGreater(
            float(sum(gradient.abs().sum() for gradient in gradients)), 0.0)


class WeekendConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load((
            ROOT / 'train_config_0821_lc_pretrained_corr_weekend.yaml'
        ).read_text())

    def test_uses_best_pretrained_checkpoint_and_frozen_encoder(self):
        model = self.config['model']
        self.assertTrue(model['pretrained_correspondence'])
        self.assertTrue(model['pretrained_correspondence_frozen'])
        self.assertEqual(
            model['pretrained_correspondence_path'],
            'ckpt/0821_flyingthings_correspondence_pretrain/best.pkl')
        self.assertEqual(model['pretrained_correspondence_radii'], [6, 4])

    def test_weekend_phase_budget_and_teacher_isolation(self):
        phases = self.config['phases']
        self.assertEqual(sum(phase['epochs'] for phase in phases), 1050)
        self.assertEqual(phases[0]['ratios']['teacher_flyingthings'], 0.10)
        self.assertEqual(phases[1]['ratios']['teacher_flyingthings'], 0.0)
        self.assertEqual(phases[2]['ratios']['teacher_flyingthings'], 0.0)


if __name__ == '__main__':
    unittest.main()
