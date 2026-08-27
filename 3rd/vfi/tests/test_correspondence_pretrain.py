import unittest
from pathlib import Path

import torch
import yaml

from train_correspondence import bilateral_info_nce, finalize_stats


VFI_ROOT = Path(__file__).resolve().parents[1]
CONFIG = VFI_ROOT / 'train_config_0821_correspondence_pretrain.yaml'


class BilateralInfoNCETest(unittest.TestCase):
    def test_identical_unique_features_match_exactly(self):
        torch.manual_seed(41)
        feature = torch.randn(1, 16, 4, 6)
        flow_gt = torch.zeros(1, 5, 16, 24)
        flow_gt[:, 4] = 1.0

        loss, raw = bilateral_info_nce(
            feature, feature.clone(), flow_gt, torch.ones(1),
            temperature=0.03, max_queries=24, random_queries=False)
        metrics = finalize_stats(raw)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics['top1_accuracy'], 1.0)
        self.assertEqual(metrics['matching_epe_px'], 0.0)
        self.assertEqual(metrics['positive_rank'], 1.0)

    def test_gt_flow_aligns_shifted_endpoint_features(self):
        torch.manual_seed(43)
        feature0 = torch.randn(1, 16, 5, 7)
        feature1 = torch.roll(feature0, shifts=1, dims=3)
        flow_gt = torch.zeros(1, 5, 20, 28)
        # Endpoint-1 feature at x+1 contains endpoint-0 feature at x.
        flow_gt[:, 2] = 4.0
        flow_gt[:, 4] = 1.0

        _, raw = bilateral_info_nce(
            feature0, feature1, flow_gt, torch.ones(1),
            temperature=0.03, max_queries=35, random_queries=False)
        metrics = finalize_stats(raw)

        self.assertGreater(metrics['top1_accuracy'], 0.99)
        self.assertLess(metrics['matching_epe_px'], 0.01)


class CorrespondenceConfigTest(unittest.TestCase):
    def test_config_uses_only_native_flyingthings_hard_cycle(self):
        config = yaml.safe_load(CONFIG.read_text())

        self.assertIn('teacher_flyingthings_train.txt',
                      config['data']['train_list'])
        self.assertEqual(config['data']['mv_cache_dirname'], 'mv_cache_f16')
        self.assertEqual(set(config['loss']['scale_weights']), {'1/8', '1/16'})

    def test_token_budget_matches_training_crop(self):
        config = yaml.safe_load(CONFIG.read_text())
        height, width = config['data']['crop_size']

        self.assertEqual(
            config['model']['max_feature_tokens'],
            (height // 8) * (width // 8))


if __name__ == '__main__':
    unittest.main()
