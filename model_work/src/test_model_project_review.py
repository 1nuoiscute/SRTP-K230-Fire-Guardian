"""Project review must keep a useful candidate even when the old regression gate fails."""
from copy import deepcopy
import unittest
from model_project_review import video_review_plan


class ProjectReviewTests(unittest.TestCase):
    def fixture(self):
        ref = dict(blue={'primary_localized_iou50': 1, 'per_image': [
            {'slug': 'blue_cardoner', 'primary_localized_iou50': False},
            {'slug': 'blue_ka23', 'primary_localized_iou50': False}]},
            visible_flame={'supplemental': {'0.5': {'micro_f1': .07}}, 'mean_best_prediction_iou': .31},
            legacy={'map50': .97}, micro={'micro_f1': .87}, sources={'nofire_real_indoor': {'fp': 1}})
        candidate = deepcopy(ref)
        candidate['blue']['per_image'][1]['primary_localized_iou50'] = True
        candidate['blue']['primary_localized_iou50'] = 4
        candidate['legacy']['map50'] = .79
        candidate['micro']['micro_f1'] = .82
        candidate['sources']['nofire_real_indoor']['fp'] = 19
        candidate['gate_passed'] = False
        return ref, candidate

    def test_blue_gain_and_old_loss_still_receive_review(self):
        ref, candidate = self.fixture()
        plan = video_review_plan({'v13': ref, 'scratch': candidate}, 'v13', ['scratch'])
        self.assertEqual(plan['candidates_in_review_order'], ['scratch'])
        self.assertEqual(plan['signals']['scratch']['under_pot_blue_gain'], 1)
        self.assertEqual(plan['signals']['scratch']['legacy_indoor_fp_change'], 18)
        self.assertFalse(plan['historical_gate_used_to_exclude'])
        self.assertFalse(plan['automatic_default_replacement'])

    def test_ordering_does_not_discard_geometry_or_no_gain_candidates(self):
        ref, blue = self.fixture()
        geometry = deepcopy(ref)
        geometry['visible_flame']['supplemental']['0.5']['micro_f1'] = .5
        plan = video_review_plan({'v13': ref, 'blue': blue, 'geometry': geometry, 'unchanged': ref},
                                 'v13', ['geometry', 'unchanged', 'blue'])
        self.assertEqual(plan['candidates_in_review_order'], ['blue', 'geometry', 'unchanged'])
        self.assertTrue(plan['all_declared_candidates_retained'])

    def test_missing_under_pot_identity_is_rejected(self):
        ref, candidate = self.fixture()
        candidate['blue']['per_image'].pop()
        with self.assertRaises(ValueError):
            video_review_plan({'v13': ref, 'scratch': candidate}, 'v13', ['scratch'])

    def test_duplicate_candidates_are_rejected(self):
        ref, candidate = self.fixture()
        with self.assertRaises(ValueError):
            video_review_plan({'v13': ref, 'scratch': candidate}, 'v13', ['scratch', 'scratch'])


if __name__ == '__main__':
    unittest.main()
