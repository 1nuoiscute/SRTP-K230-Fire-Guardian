import importlib.util
import unittest
from pc_dual_head_onnx import postprocess_dual


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Optional NumPy unavailable')
class DualRuntimeTests(unittest.TestCase):
    def data(self):
        import numpy as np
        return np.zeros((1,6,16800),dtype=np.float32)

    def test_same_geometry_across_classes_keeps_both_and_within_class_suppresses_duplicates(self):
        import numpy as np
        raw=self.data(); raw[0,:4,0]=[100,100,40,40]; raw[0,4,0]=.8
        raw[0,:4,1]=[100,100,40,40]; raw[0,4,1]=.7
        raw[0,:4,8400]=[100,100,40,40]; raw[0,5,8400]=.9
        result=postprocess_dual(raw)
        self.assertEqual(result[:,5].tolist(),[1.,0.]); np.testing.assert_equal(result[:,:4],[[80,80,120,120]]*2)

    def test_bad_layout_and_cross_class_scores_rejected(self):
        import numpy as np
        with self.assertRaises(ValueError): postprocess_dual(np.zeros((1,6,8400),dtype=np.float32))
        raw=self.data(); raw[0,5,0]=.5
        with self.assertRaisesRegex(ValueError,'isolation'): postprocess_dual(raw)
        raw=self.data(); raw[0,4,8400]=.5
        with self.assertRaisesRegex(ValueError,'isolation'): postprocess_dual(raw)

    def test_global_limit_and_empty_output(self):
        raw=self.data(); self.assertEqual(postprocess_dual(raw).shape,(0,6))
        raw[0,:4,0]=[100,100,40,40]; raw[0,4,0]=.8
        raw[0,:4,8400]=[100,100,40,40]; raw[0,5,8400]=.9
        self.assertEqual(postprocess_dual(raw,max_det=1)[:,5].tolist(),[1.])


if __name__=='__main__':unittest.main()
