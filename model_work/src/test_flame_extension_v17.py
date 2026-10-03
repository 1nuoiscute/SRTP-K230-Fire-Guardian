"""Reject unsafe train supervision and false completion claims in the v17 route."""
import importlib.util
from pathlib import Path
from copy import deepcopy
import tempfile
import unittest
AVAILABLE=all(importlib.util.find_spec(n) is not None for n in ('cv2','numpy','yaml'))
if AVAILABLE:
    from extend_flame_review_v17 import validate_decisions, quantiles
    from data_integrity import sha256


@unittest.skipUnless(AVAILABLE,'OpenCV/NumPy/PyYAML needed for reviewed data tools')
class ExtensionReviewTests(unittest.TestCase):
    def fixture(self,root):
        image=root/'image.bin'; image.write_bytes(b'fixed train pixels')
        label=root/'label.txt'; label.write_text('',encoding='utf-8')
        sources=[dict(index=i,kind='negative',split='train',image=str(image),label=str(label),
            image_sha256=sha256(image),label_sha256=sha256(label),size=[1920,1080],original_boxes_xyxy=[]) for i in range(1,49)]
        decisions=[dict(index=i,status='held',boxes_xyxy=[],note='uncertain visible evidence') for i in range(1,49)]
        return dict(images=sources),decisions

    def test_validation_image_cannot_be_approved_as_training_negative(self):
        with tempfile.TemporaryDirectory() as d:
            inputs,decisions=self.fixture(Path(d)); inputs['images'][0]['split']='val'; decisions[0]['status']='negative'
            with self.assertRaisesRegex(ValueError,'Source identity or split'):
                validate_decisions(inputs,decisions)

    def test_uncertain_positive_cannot_be_made_empty_negative(self):
        with tempfile.TemporaryDirectory() as d:
            inputs,decisions=self.fixture(Path(d)); inputs['images'][0].update(kind='flame',original_boxes_xyxy=[[1,1,2,2]])
            decisions[0]['status']='negative'
            with self.assertRaisesRegex(ValueError,'uncertain flame treated as negative'):
                validate_decisions(inputs,decisions)

    def test_held_supervised_boxes_and_out_of_bounds_flames_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            inputs,decisions=self.fixture(Path(d)); decisions[0]['boxes_xyxy']=[[1,1,2,2]]
            with self.assertRaises(ValueError): validate_decisions(inputs,decisions)
            inputs['images'][0].update(kind='flame',original_boxes_xyxy=[[1,1,2,2]])
            decisions[0].update(status='flame',boxes_xyxy=[[1,1,641,20]])
            with self.assertRaisesRegex(ValueError,'Invalid flame envelope'):
                validate_decisions(inputs,decisions)

    def test_changed_original_and_incomplete_review_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            inputs,decisions=self.fixture(Path(d)); Path(inputs['images'][0]['image']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Source identity'):
                validate_decisions(inputs,decisions)
            with self.assertRaisesRegex(ValueError,'Incomplete ordered'):
                validate_decisions(inputs,decisions[:-1])

    def test_selection_is_input_order_independent(self):
        rows=[dict(group_id=str(i),sha256=str(i)) for i in range(30)]
        self.assertEqual(quantiles(rows,8),quantiles(list(reversed(rows)),8))


ML_AVAILABLE=all(importlib.util.find_spec(n) is not None for n in ('torch','ultralytics','cv2','numpy','yaml'))
@unittest.skipUnless(ML_AVAILABLE,'Full ML dependencies needed for completion audit')
class CompletionTests(unittest.TestCase):
    def test_incomplete_epochs_or_fake_updates_cannot_be_completed(self):
        from train_flame_extension_v17 import validate_completion, START_SHA, PRIMARY
        meta=dict(candidate_training=True,source_sha256=START_SHA,config={'epochs':12},checkpoint_screen=PRIMARY,
                  checkpoints={'best.pt':'a','last.pt':'b'},actual_optimizer_steps=4,actual_batches=5)
        epochs=[dict(epoch=i,changed_feature_weight_tensors=1,changed_head_weight_tensors=1) for i in range(1,13)]
        cadence=dict(actual_optimizer_steps=4,batches=5)
        validate_completion(meta,epochs,cadence)
        with self.assertRaises(ValueError): validate_completion(meta,epochs[:-1],cadence)
        with self.assertRaises(ValueError): validate_completion(meta,epochs,dict(actual_optimizer_steps=3,batches=5))
        fake=deepcopy(meta); fake['candidate_training']=False
        with self.assertRaises(ValueError): validate_completion(fake,epochs,cadence)

if __name__=='__main__': unittest.main()
