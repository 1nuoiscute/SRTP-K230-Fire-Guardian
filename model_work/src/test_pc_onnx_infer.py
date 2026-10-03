import importlib.util
import unittest

from pc_onnx_infer import postprocess, preprocess, restore_boxes, validate_input_policy


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("cv2"), "Optional NumPy/OpenCV unavailable")
class OnnxPostprocessTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.raw = np.zeros((1, 5, 8400), dtype=np.float32)

    def test_nms_score_only_and_duplicate_suppression(self):
        self.raw[0, :, 0] = [200, 100, 80, 40, .9]
        self.raw[0, :, 1] = [201, 100, 80, 40, .8]
        self.raw[0, :, 2] = [400, 300, 20, 20, .7]
        original = self.raw.copy()
        boxes = postprocess(self.raw)
        self.assertEqual(boxes.shape, (2, 6))
        self.np.testing.assert_allclose(boxes[0, :4], [160, 80, 240, 120])
        self.np.testing.assert_array_equal(original, self.raw)
        self.assertEqual(postprocess(self.raw, max_det=1).shape, (1, 6))

    def test_empty_and_strict_confidence_boundary(self):
        self.raw[0, :, 0] = [20, 20, 10, 10, .25]
        self.assertEqual(postprocess(self.raw).shape, (0, 6))

    def test_reject_output_contract_and_nonfinite(self):
        for raw in [self.raw[:, :, :6], self.raw.astype(self.np.float64), self.raw + self.np.nan]:
            with self.assertRaises(ValueError):
                postprocess(raw)
        self.raw[0, 4, 0] = 1.1
        with self.assertRaises(ValueError):
            postprocess(self.raw)

    def test_letterbox_rgb_and_original_coordinate_restoration(self):
        image = self.np.zeros((321, 641, 3), dtype=self.np.uint8)
        image[:, :, 2] = 255
        tensor, geometry = preprocess(image)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor[0, 0, 320, 320], 1.)
        self.assertEqual(tensor[0, 2, 320, 320], 0.)
        original = self.np.array([[10, 20, 100, 200, .9, 0]], dtype=self.np.float32)
        padded = original.copy()
        padded[:, :4] *= geometry["gain"]
        padded[:, [0, 2]] += geometry["left"]
        padded[:, [1, 3]] += geometry["top"]
        self.np.testing.assert_allclose(restore_boxes(padded, geometry), original, atol=1e-4)

    def test_dynamic_rectangular_grid_contract(self):
        image = self.np.zeros((321, 641, 3), dtype=self.np.uint8)
        tensor, geometry = preprocess(image, rectangular=True)
        self.assertEqual(tensor.shape, (1, 3, 320, 640))
        self.assertEqual(geometry["top"], 0)
        raw = self.np.zeros((1, 5, 4200), dtype=self.np.float32)
        self.assertEqual(postprocess(raw, input_hw=(320, 640)).shape, (0, 6))
        with self.assertRaises(ValueError):
            postprocess(raw)
        with self.assertRaises(ValueError):
            postprocess(raw, input_hw=(321, 640))

    def test_rectangular_policy_rejects_static_session(self):
        class Input:
            shape = [1, 3, 640, 640]
        class Session:
            def get_inputs(self):
                return [Input()]
        validate_input_policy(Session(), False)
        with self.assertRaises(ValueError):
            validate_input_policy(Session(), True)
