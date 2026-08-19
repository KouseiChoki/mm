import unittest
from pathlib import Path

import yaml


VFI_ROOT = Path(__file__).resolve().parents[1]
CONTROL = VFI_ROOT / 'train_config_0729_lc_hard_roi_control.yaml'
WEIGHTED = VFI_ROOT / 'train_config_0729_lc_hard_roi_weighted.yaml'


class HardRoiConfigTest(unittest.TestCase):
    def test_arms_differ_only_in_names_and_three_roi_weights(self):
        control = yaml.safe_load(CONTROL.read_text())
        weighted = yaml.safe_load(WEIGHTED.read_text())
        control['exp_name'] = weighted['exp_name']
        control['phases'][0]['name'] = weighted['phases'][0]['name']
        for suffix in ('final', 'warp', 'edge'):
            weighted['model'][f'hard_roi_{suffix}_weight'] = 0.0
        self.assertEqual(control, weighted)

    def test_control_still_enables_identical_roi_diagnostics(self):
        control = yaml.safe_load(CONTROL.read_text())
        model = control['model']
        self.assertGreater(model['hard_roi_ratio'], 0.0)
        self.assertEqual(model['hard_roi_final_weight'], 0.0)
        self.assertEqual(model['hard_roi_warp_weight'], 0.0)
        self.assertEqual(model['hard_roi_edge_weight'], 0.0)
        self.assertEqual(control['monitor']['eval_every_epochs'], 1)
        self.assertEqual(control['monitor']['best_metric'], 'xtrain/roi_psnr')

    def test_batch_composition_and_training_budget_are_fixed(self):
        for path in (CONTROL, WEIGHTED):
            config = yaml.safe_load(path.read_text())
            phase = config['phases'][0]
            self.assertEqual(sum(phase['batch_counts'].values()), 4)
            self.assertEqual(phase['epochs'], 4)
            self.assertEqual(phase['steps_per_epoch'], 401)
            self.assertEqual(phase['grad_accum_steps'], 3)


if __name__ == '__main__':
    unittest.main()
