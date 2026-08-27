import sys
import unittest
from pathlib import Path

import torch
import yaml


VFI_ROOT = Path(__file__).resolve().parents[1]
if str(VFI_ROOT) not in sys.path:
    sys.path.insert(0, str(VFI_ROOT))

from model.pqmax import PQMaxMotionSynthesizer  # noqa: E402
from Trainer import Model  # noqa: E402


class PQMaxMotionSynthesizerTest(unittest.TestCase):
    def _module(self):
        return PQMaxMotionSynthesizer(
            num_fields=3, recurrent_iterations=2,
            recurrent_hidden_channels=32, correlation_channels=16,
            local_radius=1, boundary_hidden_channels=32,
            boundary_blocks=2, detail_hidden_channels=16,
            detail_blocks=2)

    def test_shapes_probabilities_and_gradients(self):
        torch.manual_seed(3)
        module = self._module().train()
        batch, height, width = 1, 64, 96
        image0 = torch.rand(batch, 3, height, width)
        image1 = torch.rand_like(image0)
        base_flow = torch.zeros(batch, 4, height, width)
        base_mask = torch.zeros(batch, 1, height, width)
        timestep = torch.full((batch, 1, height, width), 0.5)
        feature0 = torch.rand(batch, 128, height // 8, width // 8)
        feature1 = torch.rand_like(feature0)

        output = module(
            image0, image1, base_flow, base_mask, timestep,
            (feature0, feature1))
        prediction = module.restore_detail(
            image0, image1, output['merged'], output['merged'],
            output['warp0'], output['warp1'], output['fusion_entropy'])
        self.assertEqual(output['flow'].shape, base_flow.shape)
        self.assertEqual(output['merged'].shape, image0.shape)
        self.assertEqual(prediction.shape, image0.shape)
        self.assertEqual(
            module.last_candidate_merges.shape,
            (batch, 3, 3, height, width))
        probability_sum = module.last_fusion_weights.sum(dim=1)
        self.assertTrue(torch.allclose(
            probability_sum, torch.ones_like(probability_sum), atol=1e-5))

        loss = prediction.mean() + module.last_candidate_merges.mean()
        loss.backward()
        self.assertIsNotNone(module.boundary_output.weight.grad)
        self.assertGreater(
            module.boundary_output.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(module.recurrent_head[-1].weight.grad)

    def test_full_config_enables_joint_nonfrozen_model(self):
        path = VFI_ROOT / 'train_config_0825_pqmax_amt_mamba_max.yaml'
        config = yaml.safe_load(path.read_text())
        model = config['model']
        self.assertTrue(model['pqmax_enabled'])
        self.assertTrue(model['pretrained_correspondence'])
        self.assertFalse(model['pretrained_correspondence_frozen'])
        self.assertGreaterEqual(model['pqmax_num_fields'], 4)
        self.assertGreater(model['pqmax_oracle_weight'], 0.0)
        self.assertGreater(model['pqmax_selection_weight'], 0.0)
        self.assertGreater(model['pqmax_frequency_weight'], 0.0)
        self.assertGreaterEqual(sum(
            phase['epochs'] for phase in config['phases']), 1500)

    def test_direct_field_losses_are_differentiable(self):
        class State:
            pass

        trainer = Model.__new__(Model)
        trainer.net = State()
        trainer.net.pqmax = State()
        trainer.pqmax_charbonnier_eps = 1e-3
        trainer.pqmax_diversity_margin = 1.0
        trainer.pqmax_frequency_levels = 2
        batch, fields, height, width = 1, 3, 16, 24
        candidates = torch.rand(
            batch, fields, 3, height, width, requires_grad=True)
        flow = torch.rand(
            batch, fields, 4, height, width, requires_grad=True)
        logits = torch.rand(
            batch, fields, 1, height, width, requires_grad=True)
        trainer.net.pqmax.last_candidate_merges = candidates
        trainer.net.pqmax.last_candidate_flows = flow
        trainer.net.pqmax.last_fusion_weights = torch.softmax(logits, dim=1)
        target = torch.rand(batch, 3, height, width)
        prediction = torch.rand(
            batch, 3, height, width, requires_grad=True)
        roi = torch.zeros(batch, 1, height, width)
        roi[:, :, 4:10, 6:14] = 1.0

        losses = trainer._pqmax_losses(
            target, prediction, prediction, roi, 0, 0)
        sum(losses.values()).backward()
        self.assertGreater(candidates.grad.abs().sum().item(), 0.0)
        self.assertGreater(flow.grad.abs().sum().item(), 0.0)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)
        self.assertGreater(prediction.grad.abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
