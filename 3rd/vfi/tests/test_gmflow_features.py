import unittest
from pathlib import Path

import torch

from model.gmflow_pretrained import GMFlowFeatureEncoder


VFI_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = VFI_ROOT / 'pretrained' / 'gmflow_sintel-0c07dcb3.pth'


class GMFlowFeatureEncoderTest(unittest.TestCase):
    def test_large_images_are_bounded_to_attention_budget(self):
        encoder = GMFlowFeatureEncoder(
            checkpoint_path=None, max_feature_tokens=4224)
        height, width = encoder._bounded_image_size(2160, 3840)
        self.assertEqual(height % 8, 0)
        self.assertEqual(width % 8, 0)
        self.assertLessEqual((height // 8) * (width // 8), 4224)
        self.assertAlmostEqual(width / height, 3840 / 2160, delta=0.08)

    @unittest.skipUnless(CHECKPOINT.is_file(), 'GMFlow checkpoint unavailable')
    def test_official_checkpoint_loads_and_returns_eighth_scale_features(self):
        encoder = GMFlowFeatureEncoder(CHECKPOINT).eval()
        image0 = torch.rand(1, 3, 64, 96)
        image1 = torch.rand(1, 3, 64, 96)

        with torch.no_grad():
            feature0, feature1 = encoder(image0, image1)

        self.assertEqual(feature0.shape, (1, 128, 8, 12))
        self.assertEqual(feature1.shape, feature0.shape)
        self.assertTrue(torch.isfinite(feature0).all())
        self.assertTrue(torch.isfinite(feature1).all())
        self.assertGreater(float(feature0.std()), 0.01)


if __name__ == '__main__':
    unittest.main()
