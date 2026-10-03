"""Reject unsafe train-context approvals and validation pixel leakage."""
import csv
from pathlib import Path
import tempfile
import unittest
try:
    from curate_train_context_v19 import validate_decisions, check_inventory, label_text
    from data_integrity import sha256
    AVAILABLE=True
except ImportError:
    AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'Requires local image/dataset dependencies')
class ContextApprovalTest(unittest.TestCase):
    def inputs(self):
        rows=[dict(index=i,kind='flame' if i==40 else 'negative',size=[640,640]) for i in range(1,41)]
        decisions=[dict(index=i,status='flame' if i==40 else 'negative',boxes_xyxy=[[10,20,30,40]] if i==40 else [],note='Actually reviewed original full frame') for i in range(1,41)]
        return dict(images=rows),decisions

    def test_schedule_controls_match_v18_and_wrong_data_is_rejected(self):
        from train_context_flame_head_v19 import frozen_schedule, validate_completion
        from train_frozen_flame_head_v18 import frozen_schedule as previous_schedule
        previous=previous_schedule(Path('v17/data.yaml'),Path('runs/v18'),Path('v16/best.pt'),23)
        current=frozen_schedule(Path('v19/data.yaml'),Path('runs/v19'),Path('v16/best.pt'),23)
        self.assertEqual({k:(previous[k],v) for k,v in current.items() if previous[k]!=v}, {'data':(str(Path('v17/data.yaml').resolve()),str(Path('v19/data.yaml').resolve())),'name':('v18','v19')})
        with self.assertRaisesRegex(ValueError,'Wrong completed v19 data'):
            validate_completion(dict(dataset=dict(manifest_sha256='stale')),[],dict())

    def test_nonfinite_or_outside_geometry_rejected(self):
        inp,d=self.inputs()
        for b in ([10,20,float('nan'),40],[10,20,641,40],[30,20,10,40]):
            d[-1]['boxes_xyxy']=[b]
            with self.assertRaises(ValueError): validate_decisions(inp,d)

    def test_cannot_turn_positive_source_into_negative(self):
        inp,d=self.inputs(); d[-1].update(status='negative',boxes_xyxy=[])
        with self.assertRaises(ValueError): validate_decisions(inp,d)

    def test_missing_identity_or_review_rejected(self):
        inp,d=self.inputs(); d[3]['note']=''
        with self.assertRaises(ValueError): validate_decisions(inp,d)
        inp,d=self.inputs(); inp['images'][3]['index']=5
        with self.assertRaises(ValueError): validate_decisions(inp,d)

    def test_held_has_no_synthetic_flame_and_coordinates_use_native_size(self):
        inp,d=self.inputs(); d[-1].update(status='held',boxes_xyxy=[])
        self.assertEqual(validate_decisions(inp,d)[-1]['status'],'held')
        d[-1]['boxes_xyxy']=[[10,20,30,40]]
        with self.assertRaises(ValueError): validate_decisions(inp,d)
        self.assertEqual(label_text(dict(size=[100,200],boxes_xyxy=[[10,20,30,60]])), '0 0.200000000 0.200000000 0.200000000 0.200000000\n')

    def test_nontrain_same_pixels_and_mutated_labels_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); selected=[]; raw=[]
            for i in range(40):
                im=root/f'{i}.jpg'; lp=root/f'{i}.txt'; im.write_bytes(str(i).encode()); lp.write_text('',encoding='utf-8')
                selected.append(dict(image=str(im),label=str(lp),image_sha256=sha256(im),label_sha256=sha256(lp),split='train',kind='negative',provenance='object_pan'))
                raw.append(dict(file=im.name,split='train',sha256=sha256(im),fire_boxes='0',provenance='object_pan'))
            def manifest():
                with (root/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(raw[0])); writer.writeheader(); writer.writerows(raw)
            manifest(); check_inventory(dict(base=str(root)),selected)
            raw.append(dict(raw[0],file='validation_copy.jpg',split='val')); manifest()
            with self.assertRaisesRegex(ValueError,'Non-train or overlapping'): check_inventory(dict(base=str(root)),selected)
            raw.pop(); manifest(); Path(selected[0]['label']).write_text('0 .5 .5 .2 .2\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'label changed'): check_inventory(dict(base=str(root)),selected)


if __name__=='__main__': unittest.main()
