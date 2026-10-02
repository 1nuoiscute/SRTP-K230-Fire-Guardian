import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('ultralytics'),'Optional Torch/Ultralytics unavailable')
class IndependentHeadTests(unittest.TestCase):
    def test_box_geometry_and_class_scores_are_independent(self):
        import torch
        from independent_smoke_head import combine_single_class_outputs
        fire=torch.arange(10,dtype=torch.float32).reshape(1,5,2)
        smoke=torch.arange(15,dtype=torch.float32).reshape(1,5,3)+20
        result=combine_single_class_outputs(fire,smoke)
        self.assertEqual(tuple(result.shape),(1,6,5))
        self.assertTrue(torch.equal(result[:,:5,:2],fire))
        self.assertTrue(torch.equal(result[:,:4,2:],smoke[:,:4]))
        self.assertTrue(torch.equal(result[:,5:,2:],smoke[:,4:5]))
        self.assertTrue(torch.all(result[:,5:,:2]==0)); self.assertTrue(torch.all(result[:,4:5,2:]==0))

    def make_head(self):
        import torch
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.nc=1; self.end2end=False; self.nl=3; self.reg_max=16
                self.stride=torch.tensor([8.,16.,32.]); self.xyxy=False; self.i=23; self.f=[16,19,22]
            def forward(self,x): return torch.ones(1,5,2),{}
        return Head()

    def test_wrapper_rejects_training_and_allows_eval(self):
        from independent_smoke_head import IndependentSmokeDetect
        head=IndependentSmokeDetect(self.make_head(),self.make_head())
        with self.assertRaisesRegex(RuntimeError,'inference-only'): head([])
        head.eval(); self.assertEqual(tuple(head([])[0].shape),(1,6,4))

    def test_class_or_stride_mismatch_rejected(self):
        from independent_smoke_head import IndependentSmokeDetect
        fire=self.make_head(); smoke=self.make_head(); smoke.nc=2
        with self.assertRaises(ValueError): IndependentSmokeDetect(fire,smoke)
        smoke=self.make_head(); smoke.stride*=2
        with self.assertRaises(ValueError): IndependentSmokeDetect(fire,smoke)

    def test_bad_output_layout_rejected(self):
        import torch
        from independent_smoke_head import combine_single_class_outputs
        with self.assertRaises(ValueError): combine_single_class_outputs(torch.zeros(1,6,2),torch.zeros(1,5,2))

    def test_nested_stride_and_cache_follow_dtype_moves(self):
        import torch
        from independent_smoke_head import IndependentSmokeDetect
        head=IndependentSmokeDetect(self.make_head(),self.make_head())
        for branch in (head.fire_head,head.smoke_head):
            branch.anchors=torch.ones(2,2); branch.strides=torch.ones(1,2); branch.shape=(1,4,8,8)
        head.half()
        for branch in (head.fire_head,head.smoke_head):
            self.assertEqual(branch.stride.dtype,torch.float16)
            self.assertEqual(branch.anchors.dtype,torch.float16)
            self.assertEqual(branch.strides.dtype,torch.float16)
            self.assertIsNone(branch.shape)


if __name__=='__main__':unittest.main()
