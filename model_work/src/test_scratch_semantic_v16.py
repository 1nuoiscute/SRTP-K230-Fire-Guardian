import copy
import unittest
from pathlib import Path
try:
    from screen_scratch_semantic_v16 import validate_completion
    from train_scratch_semantic_v16 import PRIMARY, START_SHA, schedule
    FULL_ENVIRONMENT = True
except ImportError:
    FULL_ENVIRONMENT = False


@unittest.skipUnless(FULL_ENVIRONMENT, 'Optional model/data dependencies unavailable in source-only CI')
class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.meta=dict(candidate_training=True,source_sha256=START_SHA,config={'epochs':12},
            checkpoint_screen=PRIMARY,checkpoints={n:n for n in PRIMARY},teacher_loss=False,
            actual_optimizer_steps=2837,actual_batches=2844)
        self.epochs=[dict(epoch=i,changed_feature_weight_tensors=126,changed_head_weight_tensors=42) for i in range(1,13)]
        self.cadence=dict(actual_optimizer_steps=2837,batches=2844)

    def test_incomplete_or_wrong_epoch_sequence_rejected(self):
        for epochs in [self.epochs[:-1],self.epochs[1:]+[self.epochs[-1]]]:
            with self.assertRaises(ValueError):
                validate_completion(self.meta,epochs,self.cadence)

    def test_actual_updates_not_inferred_from_batches(self):
        validate_completion(self.meta,self.epochs,self.cadence)
        with self.assertRaises(ValueError):
            validate_completion(self.meta,self.epochs,dict(self.cadence,actual_optimizer_steps=2844))

    def test_rehearsal_cannot_be_evaluated_as_candidate(self):
        with self.assertRaises(ValueError):
            validate_completion(dict(self.meta,candidate_training=False),self.epochs,self.cadence)

    def test_missing_feature_learning_rejected(self):
        bad=copy.deepcopy(self.epochs)
        bad[0]['changed_feature_weight_tensors']=0
        with self.assertRaises(ValueError):
            validate_completion(self.meta,bad,self.cadence)

    def test_finetuning_explicitly_retains_local_checkpoint(self):
        start=Path('local-v14.pt')
        cfg=schedule(Path('data.yaml'),Path('new-run'),start)
        self.assertEqual(cfg['pretrained'],str(start.resolve()))
        self.assertFalse(cfg['resume'])


if __name__=='__main__':
    unittest.main()
