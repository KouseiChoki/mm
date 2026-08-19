import unittest

import torch

from model.flow_estimation import MultiHypothesisBranch
from model.warplayer import warp


class MultiHypothesisBranchTest(unittest.TestCase):
    @staticmethod
    def _inputs(batch=1, height=32, width=48):
        torch.manual_seed(29)
        img0 = torch.rand(batch, 3, height, width)
        img1 = torch.rand(batch, 3, height, width)
        flow = torch.randn(batch, 4, height, width) * 0.2
        mask = torch.randn(batch, 1, height, width) * 0.1
        timestep = torch.full((batch, 1, height, width), 0.5)
        warp0 = warp(img0, flow[:, :2])
        warp1 = warp(img1, flow[:, 2:4])
        primary = warp0 * torch.sigmoid(mask) + warp1 * (
            1.0 - torch.sigmoid(mask))
        return img0, img1, warp0, warp1, flow, mask, timestep, primary

    @staticmethod
    def _forward(module, inputs):
        img0, img1, warp0, warp1, flow, mask, timestep, primary = inputs
        alt_flow, alt_mask, mix_logits = module.predict_alternative(
            img0, img1, warp0, warp1, flow, mask, timestep)
        alt_warp0 = warp(img0, alt_flow[:, :2])
        alt_warp1 = warp(img1, alt_flow[:, 2:4])
        alt_probability = torch.sigmoid(alt_mask)
        alternative = (
            alt_warp0 * alt_probability
            + alt_warp1 * (1.0 - alt_probability))
        combined = module.combine(
            primary, alternative, warp0, warp1, mix_logits, flow,
            alt_flow)
        return combined, alternative, alt_flow

    def test_zero_mix_is_exact_primary_identity(self):
        module = MultiHypothesisBranch(
            hidden_channels=8, work_scale=4, candidate_init_std=0.005)
        inputs = self._inputs()

        combined, alternative, alt_flow = self._forward(module, inputs)

        self.assertTrue(torch.equal(combined, inputs[-1]))
        self.assertGreater(float(
            (alternative - inputs[-1]).detach().abs().sum()), 0.0)
        self.assertGreater(float(
            (alt_flow - inputs[4]).detach().abs().sum()), 0.0)
        self.assertEqual(float(module.last_mix_abs), 0.0)
        self.assertEqual(float(module.last_output_delta_abs), 0.0)

    def test_direct_candidate_loss_trains_alternative_head(self):
        module = MultiHypothesisBranch(
            hidden_channels=8, work_scale=4, candidate_init_std=0.005)
        inputs = self._inputs()
        _, alternative, _ = self._forward(module, inputs)
        target = torch.rand_like(alternative)

        (alternative - target).square().mean().backward()

        candidate_gradient = module.output.weight.grad[:5]
        self.assertGreater(float(candidate_gradient.abs().sum()), 0.0)
        self.assertGreater(
            float(module.body[0][0].weight.grad.abs().sum()), 0.0)

    def test_mix_channel_receives_gradient_from_first_step(self):
        module = MultiHypothesisBranch(
            hidden_channels=8, work_scale=4, candidate_init_std=0.005)
        inputs = self._inputs()
        combined, _, _ = self._forward(module, inputs)
        target = inputs[-1] + 0.05

        (combined - target).square().mean().backward()

        mix_gradient = module.output.weight.grad[5:6]
        self.assertGreater(float(mix_gradient.abs().sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
