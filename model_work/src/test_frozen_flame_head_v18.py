"""Reject false v18 completion and feature drift, including non-parameter buffers."""
import importlib.util
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
AVAILABLE=all(importlib.util.find_spec(n) is not None for n in ['cv2','numpy','yaml','torch','ultralytics'])
if AVAILABLE:
    import torch
    from train_frozen_flame_head_v18 import HeadAudit,validate_completion,START_SHA


@unittest.skipUnless(AVAILABLE,'Full local ML dependencies needed')
class FrozenFlameTests(unittest.TestCase):
    def complete(self):
        meta=dict(candidate_training=True,source_sha256=START_SHA,config=dict(epochs=12,freeze=2),fixed_feature_blocks=2,
            checkpoint_screen=['best.pt','last.pt'],checkpoints={'best.pt':'a','last.pt':'b'},actual_optimizer_steps=24,actual_batches=24)
        epochs=[dict(epoch=i,feature_unchanged=True,changed_head_weight_tensors=1,ema_fixed_features_restored=True) for i in range(1,13)]
        return meta,epochs,dict(actual_optimizer_steps=24,batches=24)

    def test_unfinished_schedule_or_no_actual_learning_rejected(self):
        m,e,c=self.complete();validate_completion(m,e,c)
        with self.assertRaises(ValueError):validate_completion(m,e[:-1],c)
        e[4]['changed_head_weight_tensors']=0
        with self.assertRaises(ValueError):validate_completion(m,e,c)

    def test_changed_features_or_unrestored_ema_rejected(self):
        for key in ['feature_unchanged','ema_fixed_features_restored']:
            m,e,c=self.complete();e[7][key]=False
            with self.assertRaises(ValueError):validate_completion(m,e,c)

    def fixture(self,folder):
        core=torch.nn.Module();core.model=torch.nn.ModuleList([torch.nn.BatchNorm1d(2),torch.nn.Linear(2,1)])
        for p in core.model[0].parameters():p.requires_grad_(False)
        core.model[0].eval()
        opt=torch.optim.AdamW(core.model[1].parameters())
        return core,HeadAudit(core,folder),SimpleNamespace(model=core,optimizer=opt,accumulate=1,epoch=0,ema=None)

    def test_live_feature_buffer_drift_fails_audit(self):
        with tempfile.TemporaryDirectory() as d:
            core,audit,trainer=self.fixture(Path(d))
            try:
                audit.on_start(trainer);audit.batch_start(trainer)
                core.model[0].running_mean[0]+=1
                with self.assertRaisesRegex(RuntimeError,'Frozen backbone tensors changed'):audit.epoch_end(trainer)
            finally:audit.cadence.close()

    def test_actual_source_change_rejected_before_step(self):
        with tempfile.TemporaryDirectory() as d:
            core,audit,trainer=self.fixture(Path(d))
            try:
                with torch.no_grad():core.model[1].weight.add_(1)
                with self.assertRaisesRegex(RuntimeError,'Actual v18 start differs'):audit.on_start(trainer)
            finally:audit.cadence.close()


if __name__=='__main__':unittest.main()
