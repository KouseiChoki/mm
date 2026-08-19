import unittest

import torch

from model.flow_estimation import SparseGlobalMatcher


class SparseGlobalMatcherTest(unittest.TestCase):
    @staticmethod
    def _inputs(batch=1, channels=8, height=4, width=6, scale=2):
        torch.manual_seed(7)
        feature0 = torch.randn(batch, channels, height, width)
        feature1 = torch.randn(batch, channels, height, width)
        img0 = torch.rand(batch, 3, height * scale, width * scale)
        img1 = torch.rand(batch, 3, height * scale, width * scale)
        flow = torch.randn(batch, 4, height * scale, width * scale)
        timestep = torch.full(
            (batch, 1, height * scale, width * scale), 0.5)
        return feature0, feature1, img0, img1, flow, timestep

    def test_zero_initialized_adapter_is_exact_noop(self):
        module = SparseGlobalMatcher(
            feature_channels=8, feature_scale=2, hidden_channels=8,
            topk_ratio=0.25, min_points=2, max_points=8)
        inputs = self._inputs()

        actual = module(*inputs)

        self.assertTrue(torch.equal(actual, inputs[4]))
        self.assertEqual(float(module.last_residual_abs), 0.0)
        self.assertGreater(float(module.last_selected_ratio), 0.0)

    def test_output_adapter_receives_gradient_on_first_step(self):
        module = SparseGlobalMatcher(
            feature_channels=8, feature_scale=2, hidden_channels=8,
            topk_ratio=0.25, min_points=2, max_points=8)
        inputs = self._inputs()

        target = inputs[4] + 0.1
        (module(*inputs) - target).square().mean().backward()

        gradient = module.adapter[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_sparse_point_budget_is_bounded(self):
        module = SparseGlobalMatcher(
            feature_channels=8, feature_scale=2, hidden_channels=8,
            topk_ratio=0.5, min_points=2, max_points=5)
        inputs = self._inputs(height=5, width=7)

        module(*inputs)

        self.assertAlmostEqual(
            float(module.last_selected_ratio), 5.0 / 35.0, places=6)

    def test_midpoint_proposal_preserves_target_coordinate(self):
        module = SparseGlobalMatcher(
            feature_channels=8, feature_scale=2, hidden_channels=8,
            topk_ratio=1.0, min_points=24, max_points=24,
            confidence_scale=1.0, mutual_sigma=100.0)
        torch.manual_seed(11)
        feature0 = torch.randn(1, 8, 4, 6)
        # A cyclic shift creates a known frame-1 correspondence for interior
        # points; random features make the global top-1 match unambiguous.
        feature1 = torch.roll(feature0, shifts=1, dims=3)
        flow_low = torch.zeros(1, 4, 4, 6)
        score = torch.ones(1, 4, 6)
        timestep = torch.full((1, 1, 4, 6), 0.25)

        proposal, support = module._sparse_proposal(
            feature0, feature0, feature1, flow_low, score, timestep)
        weighted_midpoint = (
            (1.0 - timestep) * proposal[:, :2]
            + timestep * proposal[:, 2:4])

        self.assertGreater(float(support.sum()), 0.0)
        self.assertTrue(torch.allclose(
            weighted_midpoint, torch.zeros_like(weighted_midpoint),
            atol=1e-6))


if __name__ == '__main__':
    unittest.main()
