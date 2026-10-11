import unittest

from verify_onnx_export import check_detections


class DetectionParityTests(unittest.TestCase):
    def test_order_independence_and_tolerance(self):
        a = [[0, 0, 10, 10, .9, 0], [30, 30, 40, 40, .7, 0]]
        b = [[30.01, 30, 40, 40, .70001, 0], [0, 0, 10, 10, .9, 0]]
        self.assertTrue(check_detections(a, b))

    def test_missing_duplicate_wrong_class_and_large_drift(self):
        a = [[0, 0, 10, 10, .9, 0], [30, 30, 40, 40, .7, 0]]
        for b in [a[:1], [a[0], a[0]], [a[0], [30, 30, 40, 40, .7, 1]],
                  [a[0], [30, 30, 41, 40, .7, 0]], [a[0], [30, 30, 40, 40, .72, 0]]]:
            self.assertFalse(check_detections(a, b))

    def test_empty_pair(self):
        self.assertTrue(check_detections([], []))
