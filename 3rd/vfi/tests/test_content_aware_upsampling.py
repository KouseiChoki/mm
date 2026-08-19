import unittest

import torch
import torch.nn.functional as F

from model.flow_estimation import ContentAwareFlowUpsampler, IFBlock


class ContentAwareFlowUpsamplerTest(unittest.TestCase):
    def test_zero_kernel_is_exact_bilinear_identity(self):
        torch.manual_seed(1)
        module = ContentAwareFlowUpsampler(
            guidance_channels=18, factor=2, hidden_channels=8)
        low_flow = torch.randn(2, 4, 5, 7)
        guidance = torch.randn(2, 18, 10, 14)

        actual = module(low_flow, guidance)
        expected = F.interpolate(
            low_flow, scale_factor=2, mode='bilinear',
            align_corners=False) * 2

        self.assertTrue(torch.equal(actual, expected))

    def test_output_kernel_receives_gradient_from_first_step(self):
        torch.manual_seed(2)
        module = ContentAwareFlowUpsampler(
            guidance_channels=18, factor=2, hidden_channels=8)
        low_flow = torch.randn(1, 4, 4, 6)
        guidance = torch.randn(1, 18, 8, 12)

        module(low_flow, guidance).square().mean().backward()
        kernel_grad = module.kernel[-1].weight.grad
        self.assertIsNotNone(kernel_grad)
        self.assertGreater(float(kernel_grad.abs().sum()), 0.0)

    def test_zero_sum_kernel_preserves_constant_flow(self):
        module = ContentAwareFlowUpsampler(
            guidance_channels=18, factor=2, hidden_channels=8)
        with torch.no_grad():
            module.kernel[-1].bias.normal_()
        low_flow = torch.ones(1, 4, 4, 6)
        guidance = torch.randn(1, 18, 8, 12)

        actual = module(low_flow, guidance)
        expected = F.interpolate(
            low_flow, scale_factor=2, mode='bilinear',
            align_corners=False) * 2

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_ifblock_preserves_full_resolution_shapes(self):
        block = IFBlock(
            in_planes=18, c=16, scale=1, down=4, blocks=1,
            content_aware_upsampling=True,
            content_aware_hidden_channels=8)
        inputs = torch.randn(1, 14, 32, 48)
        flow = torch.randn(1, 4, 32, 48)

        flow_delta, mask_delta = block(inputs, flow)

        self.assertEqual(flow_delta.shape, (1, 4, 32, 48))
        self.assertEqual(mask_delta.shape, (1, 1, 32, 48))


if __name__ == '__main__':
    unittest.main()
