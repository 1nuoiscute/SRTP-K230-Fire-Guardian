import unittest
from copy import deepcopy
from compare_fire_training_routes import candidate_gate, micro_score


class RouteComparisonTests(unittest.TestCase):
    def fixtures(self):
        source = {k: dict(tp=0, fp=0, fn=0) for k in ('nofire_real_indoor', 'ks_flame', 'kitchen_stove_fire')}
        source['nofire_real_indoor']['fp'] = 1
        source['ks_flame']['tp'] = 57
        source['kitchen_stove_fire']['tp'] = 18
        v3 = dict(legacy=dict(map50=.973), blue=dict(primary_localized_iou50=0), sources=source)
        v13 = dict(legacy=dict(map50=.971), blue=dict(primary_localized_iou50=1), micro=dict(micro_f1=.872))
        candidate = dict(legacy=dict(map50=.97), blue=dict(primary_localized_iou50=2), micro=dict(micro_f1=.89))
        return source, v3, v13, candidate

    def test_count_micro_f1_is_paired_and_not_primary_photo_metric(self):
        result = micro_score({'old': dict(tp=75, fp=7, fn=2)}, {'new': dict(tp=27, fp=7, fn=14)})
        self.assertEqual((result['tp'], result['fp'], result['fn']), (102, 14, 16))
        self.assertAlmostEqual(result['micro_f1'], .8717948717948718)

    def test_candidate_needs_real_gain_over_incumbent(self):
        source, v3, v13, candidate = self.fixtures()
        self.assertTrue(all(candidate_gate(candidate, source, v3, v13).values()))
        candidate['micro']['micro_f1'] = v13['micro']['micro_f1']
        self.assertFalse(candidate_gate(candidate, source, v3, v13)['development_f1_gain_over_v13'])

    def test_improving_new_cases_cannot_mask_legacy_failure(self):
        source, v3, v13, candidate = self.fixtures()
        candidate['legacy']['map50'] = .90
        self.assertFalse(candidate_gate(candidate, source, v3, v13)['legacy_map50'])
        candidate['legacy']['map50'] = .97
        source = deepcopy(source)
        source['nofire_real_indoor']['fp'] = 5
        self.assertFalse(candidate_gate(candidate, source, v3, v13)['indoor_false_positive'])


if __name__ == '__main__':
    unittest.main()
