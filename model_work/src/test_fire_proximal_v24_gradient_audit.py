import unittest
from audit_fire_proximal_v24_gradients import summarize


class ProximalGradientAuditTests(unittest.TestCase):
    def test_uses_current_distance_and_excludes_skipped_step(self):
        rows = [
            dict(actual_optimizer_step=True, detection_grad_norm=4.0, current_anchor_distance_norm=2.0, anchor_to_detection_grad_ratio=5.0, combined_grad_norm_preclip=11.0),
            dict(actual_optimizer_step=False),
        ]
        result = summarize(dict(anchor_lambda=0.5, records=rows, actual_optimizer_steps=1, target_anchor_gradient_ratio=0.2))
        self.assertEqual(result["actual_preclip_ratio_median"], 0.25)
        self.assertEqual(result["historical_initial_distance_ratio_median"], 5.0)
        self.assertEqual(result["steps_with_combined_preclip_norm_above_10"], 1)

    def test_rejects_invalid_or_missing_step(self):
        with self.assertRaises(ValueError):
            summarize(dict(anchor_lambda=1.0, records=[], actual_optimizer_steps=1, target_anchor_gradient_ratio=0.2))
        with self.assertRaises(ValueError):
            summarize(dict(anchor_lambda=1.0, records=[dict(actual_optimizer_step=True, detection_grad_norm=0.0, current_anchor_distance_norm=1.0)], actual_optimizer_steps=1, target_anchor_gradient_ratio=0.2))


if __name__ == "__main__":
    unittest.main()
