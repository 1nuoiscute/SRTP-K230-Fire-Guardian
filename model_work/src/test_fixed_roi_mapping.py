import unittest

from eval_fixed_roi_rescue_20261029 import map_boxes


class _Array:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value

    def item(self):
        return self.value


def box(coords, confidence=0.7):
    class Box:
        xyxy = [_Array(coords)]
        conf = _Array(confidence)
    return Box()


class FixedRoiMappingTests(unittest.TestCase):
    def test_maps_normal_and_partially_clipped_boxes(self):
        results = map_boxes([box([1, 2, 8, 9]), box([-0.05, -0.05, 10.05, 20.05])], (100, 200, 110, 220))
        self.assertEqual(results[0]["xyxy"], [101, 202, 108, 209])
        self.assertEqual(results[1]["xyxy"], [100, 200, 110, 220])
        self.assertEqual(results[1]["confidence"], 0.7)

    def test_fully_outside_within_tolerance_is_rejected(self):
        for coords in ([-0.08, 1, -0.02, 2], [10.02, 1, 10.08, 2],
                       [1, -0.08, 2, -0.02], [1, 20.02, 2, 20.08]):
            with self.subTest(coords=coords), self.assertRaisesRegex(ValueError, "zero area"):
                map_boxes([box(coords)], (100, 200, 110, 220))

    def test_rejects_invalid_coordinates_and_confidence(self):
        for coords, confidence in (([-0.11, 1, 2, 3], 0.7), ([1, 2, 1, 3], 0.7),
                                   ([1, 2, float("nan"), 3], 0.7), ([1, 2, 3, 4], float("inf"))):
            with self.subTest(coords=coords, confidence=confidence), self.assertRaises(ValueError):
                map_boxes([box(coords, confidence)], (100, 200, 110, 220))

    def test_rejects_invalid_roi_even_when_no_boxes(self):
        for roi in ((0, 0, 0, 20), (0, 0, 10, 0), (-1, 0, 10, 20), (0, 0, float("inf"), 20)):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                map_boxes([], roi)

    def test_empty_predictions_remain_empty(self):
        self.assertEqual(map_boxes([], (100, 200, 110, 220)), [])


if __name__ == "__main__":
    unittest.main()
