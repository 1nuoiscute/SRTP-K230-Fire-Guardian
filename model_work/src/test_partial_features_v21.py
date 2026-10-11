"""Acceptance-boundary regressions for partial-feature training completion."""
from copy import deepcopy
import unittest
from train_kitchen_partial_features_v21 import validate_completion,START_SHA,DATA_SHA


def fixture():
    meta=dict(dataset={'manifest_sha256':DATA_SHA},source_sha256=START_SHA,
        candidate_training=True,config={'epochs':12,'freeze':10},frozen_prefix_blocks=10,
        feature_blocks=23,feature_bn_statistics_fixed=True,checkpoint_screen=['best.pt','last.pt'],
        checkpoints={'best.pt':'a','last.pt':'b'},serialized_frozen_state_equal=True,
        serialized_adaptation={n:{'feature_weight_tensors':60,'head_weight_tensors':42} for n in ['best.pt','last.pt']},
        actual_optimizer_steps=3618,actual_batches=3624)
    epochs=[dict(epoch=i,frozen_prefix_unchanged=True,feature_bn_unchanged=True,
        changed_head_weight_tensors=42,changed_feature_weight_tensors=60,ema_fixed_state_restored=True) for i in range(1,13)]
    cadence=dict(actual_optimizer_steps=3618,batches=3624)
    return meta,epochs,cadence


class PartialFeatureCompletionTests(unittest.TestCase):
    def test_complete_route_accepted(self):validate_completion(*fixture())

    def test_head_only_updates_do_not_prove_partial_adaptation(self):
        meta,epochs,cadence=fixture();epochs[-1]['changed_feature_weight_tensors']=0
        with self.assertRaises(ValueError):validate_completion(meta,epochs,cadence)

    def test_feature_statistics_changes_or_ema_failure_rejected(self):
        for key in ['feature_bn_unchanged','frozen_prefix_unchanged','ema_fixed_state_restored']:
            with self.subTest(key=key):
                meta,epochs,cadence=fixture();epochs[4][key]=False
                with self.assertRaises(ValueError):validate_completion(meta,epochs,cadence)

    def test_saved_checkpoint_must_contain_feature_adaptation(self):
        meta,epochs,cadence=fixture();meta['serialized_adaptation']['last.pt']['feature_weight_tensors']=0
        with self.assertRaises(ValueError):validate_completion(meta,epochs,cadence)


if __name__=='__main__':unittest.main()
