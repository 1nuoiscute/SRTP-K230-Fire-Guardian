"""A semantic target gain cannot conceal old-scene loss or excess false positives."""
from copy import deepcopy
import importlib.util
import unittest

AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('torch', 'ultralytics', 'cv2', 'numpy', 'yaml'))
if AVAILABLE:
    from screen_visible_flame_pilot_v15 import candidate_gate


@unittest.skipUnless(AVAILABLE, 'Optional local ML dependencies unavailable')
class SemanticGateTests(unittest.TestCase):
    def fixture(self):
        v3 = dict(legacy={'map50': .973}, blue={'primary_localized_iou50': 0},
                  sources={'nofire_real_indoor': {'fp': 1}})
        v13 = dict(blue={'primary_localized_iou50': 1}, micro={'micro_f1': .87},
                   visible_flame={'mean_best_prediction_iou': .31,
                                  'supplemental': {'0.3': {'tp': 6, 'micro_f1': .4}, '0.5': {'tp': 1, 'micro_f1': .067}}})
        candidate = dict(legacy={'map50': .97}, blue={'primary_localized_iou50': 2}, micro={'micro_f1': .89},
                         sources={'nofire_real_indoor': {'fp': 1}, 'ks_flame': {'tp': 57}, 'kitchen_stove_fire': {'tp': 18}},
                         visible_flame={'mean_best_prediction_iou': .5,
                                        'supplemental': {'0.3': {'tp': 9, 'micro_f1': .6}, '0.5': {'tp': 5, 'micro_f1': .333}}})
        return candidate, v3, v13

    def test_actual_semantic_gain_with_preserved_regression_passes(self):
        row, v3, v13 = self.fixture()
        self.assertTrue(all(candidate_gate(row, v3, v13).values()))

    def test_more_tight_flame_tp_with_lower_f1_does_not_pass(self):
        row, v3, v13 = self.fixture()
        row['visible_flame']['supplemental']['0.3']['micro_f1'] = .3
        row['visible_flame']['supplemental']['0.5']['micro_f1'] = .06
        gate = candidate_gate(row, v3, v13)
        self.assertTrue(gate['visible_flame_tp_gain_iou03'])
        self.assertFalse(gate['visible_flame_f1_gain_iou03'])
        self.assertFalse(gate['visible_flame_f1_gain_iou05'])

    def test_tight_flame_improvement_does_not_excuse_old_map_or_nofire_loss(self):
        row, v3, v13 = self.fixture()
        row['legacy']['map50'] = .95
        row['sources']['nofire_real_indoor']['fp'] = 3
        gate = candidate_gate(row, v3, v13)
        self.assertTrue(gate['visible_flame_tp_gain_iou05'])
        self.assertFalse(gate['legacy_map50'])
        self.assertFalse(gate['indoor_false_positive'])

    def test_blue_or_paired_f1_loss_against_v13_does_not_pass(self):
        row, v3, v13 = self.fixture()
        row['blue']['primary_localized_iou50'] = 0
        row['micro']['micro_f1'] = .86
        gate = candidate_gate(row, v3, v13)
        self.assertFalse(gate['blue_not_less_than_v13'])
        self.assertFalse(gate['development_micro_f1_not_less_than_v13'])


if __name__ == '__main__':
    unittest.main()
