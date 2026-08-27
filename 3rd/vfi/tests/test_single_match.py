import sys
import unittest
from pathlib import Path

import torch
import yaml


VFI_ROOT = Path(__file__).resolve().parents[1]
if str(VFI_ROOT) not in sys.path:
    sys.path.insert(0, str(VFI_ROOT))

from model.single_match import ConfidenceGatedMatchRefiner  # noqa: E402
from train import (  # noqa: E402
    TENSORBOARD_CORE_LOSS_KEYS, route_tensorboard_components,
    validate_crop_sizes,
)


class ConfidenceGatedMatchRefinerTest(unittest.TestCase):
    def _module(self):
        return ConfidenceGatedMatchRefiner(
            recurrent_iterations=2,
            recurrent_hidden_channels=32,
            correlation_channels=16,
            local_radius=1,
            max_global_delta=16.0,
            recurrent_delta=1.0,
            gate_hidden_channels=16)

    def test_single_flow_shape_projection_and_gradients(self):
        torch.manual_seed(11)
        module = self._module().train()
        batch, height, width = 1, 64, 96
        image0 = torch.rand(batch, 3, height, width)
        image1 = torch.rand_like(image0)
        base_flow = torch.zeros(batch, 4, height, width)
        timestep = torch.full((batch, 1, 1, 1), 0.5)
        feature0 = torch.rand(batch, 128, height // 8, width // 8)
        feature1 = torch.rand_like(feature0)

        output = module(
            image0, image1, base_flow, timestep, (feature0, feature1))
        self.assertEqual(output.shape, base_flow.shape)
        self.assertTrue(torch.isfinite(output).all())
        # Every new update preserves the target-time location at t=0.5.
        midpoint_shift = 0.5 * output[:, :2] + 0.5 * output[:, 2:4]
        self.assertLess(midpoint_shift.abs().max().item(), 2e-4)
        self.assertGreater(module.last_proposal_abs.item(), 0.0)
        self.assertGreaterEqual(module.last_confidence.item(), 0.0)
        self.assertLessEqual(module.last_confidence.item(), 1.0)

        output.abs().mean().backward()
        self.assertIsNotNone(module.gate[-1].weight.grad)
        self.assertGreater(module.gate[-1].weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(module.recurrent_head[-1].weight.grad)
        self.assertGreater(
            module.recurrent_head[-1].weight.grad.abs().sum().item(), 0.0)

    def test_config_removes_image_candidate_selection(self):
        path = VFI_ROOT / 'train_config_0825_single_match_flow.yaml'
        config = yaml.safe_load(path.read_text())
        model = config['model']
        self.assertTrue(model['single_match_enabled'])
        self.assertFalse(model['pqmax_enabled'])
        self.assertFalse(model['multi_hypothesis'])
        self.assertFalse(model['sparse_matching'])
        self.assertTrue(model['pretrained_correspondence'])
        self.assertFalse(model['pretrained_correspondence_frozen'])
        self.assertEqual(model['pqmax_selection_weight'], 0.0)
        self.assertEqual(model['pqmax_entropy_weight'], 0.0)
        self.assertGreater(model['frequency_loss_weight'], 0.0)
        self.assertEqual(
            config['monitor']['best_metric'], 'average/psnr')
        self.assertNotIn('batch_size', config['data'])
        self.assertTrue(all(
            crop[2] == 4
            for phase in config['phases']
            for crop in phase['crop_sizes']))
        self.assertTrue(all(
            phase['grad_accum_steps'] == 2
            for phase in config['phases']))

    def test_crop_tuple_is_the_only_batch_source(self):
        self.assertEqual(
            validate_crop_sizes([[320, 576, 4]]), [(320, 576, 4)])
        with self.assertRaisesRegex(ValueError, 'height,width,batch'):
            validate_crop_sizes([[320, 576]])
        with self.assertRaisesRegex(ValueError, '16'):
            validate_crop_sizes([[321, 576, 4]])

    def test_tensorboard_routes_only_active_nonduplicate_metrics(self):
        value = torch.tensor(1.0)
        components = {
            name: value for name in TENSORBOARD_CORE_LOSS_KEYS
        }
        components.update({
            'total': value,
            'flow_raw': value,
            'flow_weight': value,
            'merge_0': value,
            'merge_weight_0': value,
            'lc_warp_stage_0': value,
            'pre_corr_stage_1_peak': value,
            'pre_corr_stage_1_entropy': value,
            'pre_corr_stage_1_feature_abs': value,
            'pre_corr_stage_1_scale': value,
            'single_match_applied': value,
            'single_match_mutual_error': value,
            'hard_roi_area': value,
            'residual': value,
            'pervfi_mask_binary': value,
            'multi_hypothesis_oracle': value,
            'pqmax_selection': value,
            'pqmax_fusion_entropy': value,
        })
        routed = route_tensorboard_components(components)
        loss_tags = {
            tag for tag in routed if tag.startswith('loss_component/')}
        self.assertEqual(len(loss_tags), len(TENSORBOARD_CORE_LOSS_KEYS))
        self.assertIn('warp_stage/loss_0', routed)
        self.assertIn('matching/pre_corr/stage_1/peak', routed)
        self.assertIn('matching/single/applied', routed)
        self.assertIn('roi/train_area', routed)
        self.assertIn('synthesis/residual_abs', routed)
        self.assertNotIn('loss_component/total', routed)
        self.assertFalse(any('merge_' in tag for tag in routed))
        self.assertFalse(any('feature_abs' in tag for tag in routed))
        self.assertFalse(any('pqmax' in tag for tag in routed))
        self.assertFalse(any('multi_hypothesis' in tag for tag in routed))
        self.assertFalse(any(tag.startswith('mask/') for tag in routed))

        pq_routed = route_tensorboard_components(
            components, pqmax_enabled=True)
        self.assertIn('pqmax_loss/selection', pq_routed)
        self.assertIn('synthesis/pqmax/fusion_entropy', pq_routed)


if __name__ == '__main__':
    unittest.main()
