import importlib.util
import unittest

from checkpoint_blend import blend_states


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional PyTorch unavailable")
class BlendTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.base = {"weight": torch.tensor([0., 2.], dtype=torch.float16),
                     "running_var": torch.tensor([1., 3.], dtype=torch.float16),
                     "num_batches_tracked": torch.tensor(4)}
        self.adapted = {"weight": torch.tensor([4., 6.], dtype=torch.float16),
                        "running_var": torch.tensor([5., 7.], dtype=torch.float16),
                        "num_batches_tracked": torch.tensor(99)}

    def test_float_parameters_and_bn_buffers_integer_base_and_no_mutation(self):
        result, counts = blend_states(self.base, self.adapted, .25)
        self.assertEqual(result["weight"].tolist(), [1., 3.])
        self.assertEqual(result["running_var"].tolist(), [2., 4.])
        self.assertEqual(result["weight"].dtype, self.torch.float32)
        self.assertEqual(result["num_batches_tracked"].item(), 4)
        self.assertEqual(self.base["weight"].tolist(), [0., 2.])
        self.assertEqual(counts, {"floating_tensors": 2, "base_integer_buffers": 1})

    def test_endpoints(self):
        for alpha, expected in [(0., self.base), (1., self.adapted)]:
            result, _ = blend_states(self.base, self.adapted, alpha)
            self.assertEqual(result["weight"].tolist(), expected["weight"].tolist())

    def test_reject_invalid_alpha(self):
        for alpha in [-.1, 1.1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                blend_states(self.base, self.adapted, alpha)

    def test_reject_keys_shape_dtype_nonfinite(self):
        t = self.torch
        for other in [{"weight": self.adapted["weight"]},
                      {**self.adapted, "weight": t.zeros(3, dtype=t.float16)},
                      {**self.adapted, "weight": t.zeros(2, dtype=t.float32)},
                      {**self.adapted, "weight": t.tensor([float("nan"), 0.], dtype=t.float16)}]:
            with self.assertRaises(ValueError):
                blend_states(self.base, other, .5)
