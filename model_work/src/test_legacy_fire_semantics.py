"""Meaningful scope and identity checks for supplementary flame diagnostics."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

AVAILABLE = importlib.util.find_spec('cv2') is not None and importlib.util.find_spec('numpy') is not None
if AVAILABLE:
    import cv2
    import numpy as np
    from data_integrity import matches, sha256
    from legacy_fire_semantic_review import finalize, render, write
    from score_legacy_fire_semantics import score


@unittest.skipUnless(AVAILABLE, 'OpenCV/NumPy are required for semantic review fixtures')
class SemanticReviewTests(unittest.TestCase):
    def fixture(self, root):
        review = root/'review'; review.mkdir()
        cache = root/'cache'; cache.mkdir()
        rows, decisions, identities, predictions = [], [], [], []
        original = [[100,100,300,300]]
        proposed = [[200,200,240,240]]
        for index in range(1,15):
            image = root/f'样本_{index}.png'
            cv2.imwrite(str(image),np.zeros((640,640,3),np.uint8))
            label = root/f'label_{index}.txt'; label.write_text('0 .3125 .3125 .3125 .3125\n')
            rows.append(dict(index=index,image=str(image),image_sha256=sha256(image),label=str(label),
                             label_sha256=sha256(label),size=[640,640],original_boxes_xyxy=original,
                             status='pending_visible_flame_review'))
            decisions.append(dict(index=index,status='proposed' if index==14 else 'held',
                                  boxes_xyxy=proposed if index==14 else [],note='Visible extent uncertain' if index<14 else 'Approximate visible envelope'))
            identities.append(dict(file=image.name,sha256=sha256(image),label_sha256=sha256(label),gt=original))
            predictions.append(dict(image=image.name,predictions=[dict(confidence=.8,xyxy=proposed[0])],**matches(original,proposed)))
        source_sheet = review/'original.jpg'; source_sheet.write_bytes(b'fixture original view')
        write(review/'review_inputs.json',dict(images=rows,predictions_shown=False,previous_development_exposure_acknowledged=True,
              sheets=[dict(file=source_sheet.name,sha256=sha256(source_sheet))],
              script_sha256=sha256(Path(__file__).with_name('prepare_legacy_fire_semantic_audit.py'))))
        write(review/'manual_proposal.json',dict(role='Supplemental',policy='Hold uncertainty',images=decisions))
        render(review)
        manifest = json.loads((review/'proposal_render_manifest.json').read_text(encoding='utf-8'))
        finalize(review,[r['file'] for r in manifest['sheets']])
        write(cache/'summary.json',dict(configuration=dict(imgsz=640,conf=.25,nms_iou=.6,match_iou=.5),models=dict(test_model=dict(weights_sha256='a'*64))))
        (cache/'local_inputs.json').write_text(json.dumps(identities,ensure_ascii=False),encoding='utf-8')
        write(cache/'test_model_local_predictions.json',predictions)
        return review, cache

    def test_held_images_are_excluded_not_negative_and_utf8_inventory_works(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); review, cache = self.fixture(root)
            score(review,cache,['test_model'],root/'score')
            result = json.loads((root/'score/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(result['scored_images'],1)
            self.assertEqual(result['held_indices'],list(range(1,14)))
            self.assertEqual(result['models']['test_model']['supplemental']['0.5'],dict(tp=1,fp=0,fn=0,micro_f1=1.0))
            self.assertEqual(result['models']['test_model']['original_same_subset']['fp'],1)

    def test_cached_annotation_change_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); review, cache = self.fixture(root)
            inputs = json.loads((cache/'local_inputs.json').read_text(encoding='utf-8'))
            inputs[0]['gt'] = [[1,1,2,2]]
            write(cache/'local_inputs.json',inputs)
            with self.assertRaisesRegex(ValueError,'Cached original annotation differs'):
                score(review,cache,['test_model'],root/'score')
            self.assertFalse((root/'score').exists())

    def test_changed_review_cannot_silently_turn_held_image_into_negative(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); review, cache = self.fixture(root)
            proposal = json.loads((review/'manual_proposal.json').read_text(encoding='utf-8'))
            proposal['images'][0]['status'] = 'proposed'
            write(review/'manual_proposal.json',proposal)
            with self.assertRaisesRegex(ValueError,'Expected complete reviewed burner coverage'):
                score(review,cache,['test_model'],root/'score')
            self.assertFalse((root/'score').exists())


if __name__ == '__main__':
    unittest.main()
