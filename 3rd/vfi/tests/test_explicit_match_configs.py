import unittest
from pathlib import Path

import yaml


VFI_ROOT = Path(__file__).resolve().parents[1]
CONTROL = VFI_ROOT / 'train_config_0729_lc_explicit_match_control.yaml'
MATCH = VFI_ROOT / 'train_config_0729_lc_explicit_match_gmflow.yaml'


class ExplicitMatchConfigTest(unittest.TestCase):
    def test_control_and_match_differ_only_in_matching_structure(self):
        control = yaml.safe_load(CONTROL.read_text())
        match = yaml.safe_load(MATCH.read_text())
        control['exp_name'] = match['exp_name']
        control['phases'][0]['name'] = match['phases'][0]['name']
        matching_keys = {
            key for key in match['model']
            if key.startswith('sparse_matching_')
        }
        self.assertTrue(matching_keys)
        for key in matching_keys:
            match['model'].pop(key)
        match['model']['sparse_matching'] = False
        self.assertEqual(control, match)

    def test_batch_composition_matches_crop_batch(self):
        for path in (CONTROL, MATCH):
            config = yaml.safe_load(path.read_text())
            phase = config['phases'][0]
            self.assertEqual(
                sum(phase['batch_counts'].values()),
                phase['crop_sizes'][0][2])


if __name__ == '__main__':
    unittest.main()
