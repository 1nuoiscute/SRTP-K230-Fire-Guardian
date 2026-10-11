import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional PyTorch unavailable')
class JointHeadTests(unittest.TestCase):
    def make_model(self,nc,channels=4):
        import torch
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.nc=nc; self.end2end=False
                self.cv3=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(channels,nc,1)) for _ in range(3)])
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.model=torch.nn.ModuleList([torch.nn.Conv2d(3,channels,1),Head()])
        return Model()

    def test_exact_fire_copy_new_smoke_low_and_source_unchanged(self):
        import torch
        from joint_head_init import transfer_fire_and_add_smoke
        old=self.make_model(1); new=self.make_model(2)
        before={k:v.clone() for k,v in old.state_dict().items()}
        result=transfer_fire_and_add_smoke(old,new)
        self.assertEqual(result['expanded_class_tensors'],6)
        for key,value in before.items(): self.assertTrue(torch.equal(value,old.state_dict()[key]))
        image=torch.randn(1,3,8,8)
        features=old.model[0](image); candidate=new.model[0](image)
        self.assertTrue(torch.equal(features,candidate))
        for a,b in zip(old.model[1].cv3,new.model[1].cv3):
            expected=a(features); actual=b(candidate)
            torch.testing.assert_close(actual[:,0:1],expected)
            self.assertTrue(torch.all(actual[:,1]==-6))

    def test_wrong_class_count_rejected(self):
        from joint_head_init import transfer_fire_and_add_smoke
        with self.assertRaisesRegex(ValueError,'one-class'):
            transfer_fire_and_add_smoke(self.make_model(1),self.make_model(3))

    def test_nonclass_shape_change_rejected(self):
        from joint_head_init import transfer_fire_and_add_smoke
        with self.assertRaises(ValueError):
            transfer_fire_and_add_smoke(self.make_model(1),self.make_model(2,5))


if __name__=='__main__':unittest.main()
