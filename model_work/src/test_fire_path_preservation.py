import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from fire_path_preservation import FirePathPreservation


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional PyTorch unavailable')
class PreservationTests(unittest.TestCase):
    def make_model(self):
        import torch
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.nc=2; self.end2end=False
                self.cv3=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(4,2,1)) for _ in range(3)])
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.model=torch.nn.ModuleList([
                    torch.nn.Sequential(torch.nn.Conv2d(3,4,1),torch.nn.BatchNorm2d(4)),Head()])
        return Model()

    def test_adamw_and_mixed_gradients_preserve_fire_update_smoke(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            model=self.make_model(); observer=FirePathPreservation(model,Path(directory)/'audit.json')
            optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
            observer.configure(model,optimizer); before=copy.deepcopy(model.state_dict())
            model.train(); observer.set_fixed_bn(model)
            features=model.model[0](torch.randn(2,3,8,8))
            sum(branch(features).sum() for branch in model.model[-1].cv3).backward(); optimizer.step()
            observer.verify(model,'after_step')
            self.assertFalse(torch.equal(model.state_dict()['model.1.cv3.0.0.weight'][1],before['model.1.cv3.0.0.weight'][1]))
            self.assertEqual(observer.batch_checks,1); observer.close()

    def test_protected_weight_buffer_and_trainable_mask_changes_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            for target in ('weight','buffer','mask'):
                model=self.make_model(); observer=FirePathPreservation(model,Path(directory)/(target+'.json'))
                optimizer=torch.optim.AdamW(model.parameters()); observer.configure(model,optimizer)
                with torch.no_grad():
                    if target=='weight': model.model[-1].cv3[0][-1].weight[0].add_(1)
                    if target=='buffer': model.model[0][1].running_mean.add_(1)
                if target=='mask': model.model[0][0].weight.requires_grad_(True)
                with self.assertRaises(RuntimeError): observer.verify(model,'corrupt')
                observer.close()

    def test_ema_restores_protected_state_and_keeps_smoke_row(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            model=self.make_model(); observer=FirePathPreservation(model,Path(directory)/'audit.json')
            ema=copy.deepcopy(model)
            with torch.no_grad():
                for value in ema.parameters(): value.add_(1)
                ema.model[0][1].running_mean.add_(1)
            smoke=ema.model[-1].cv3[0][-1].weight[1].clone()
            observer.sync_ema(ema)
            self.assertTrue(torch.equal(smoke,ema.model[-1].cv3[0][-1].weight[1]))
            observer.close()


if __name__=='__main__':unittest.main()
