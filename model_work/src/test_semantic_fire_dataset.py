"""Prevent approved-label loss and unintended teacher/sampling changes in semantic pilots."""
import importlib.util
import unittest
AVAILABLE=all(importlib.util.find_spec(n) is not None for n in ('cv2','numpy','yaml'))
if AVAILABLE:
    from semantic_fire_dataset import membership,yolo_label
    from data_integrity import validate_yolo


@unittest.skipUnless(AVAILABLE,'OpenCV/NumPy/PyYAML needed for dataset tools')
class SemanticDatasetTests(unittest.TestCase):
    def rows(self):
        return [dict(image='new.jpg',origin='legacy_train',sha256='a',label_sha256='old-a',weight=1,teacher_preserve=True,label_revised=False),
                dict(image='prior.jpg',origin='legacy_train',sha256='b',label_sha256='reviewed-b',weight=1,teacher_preserve=False,label_revised=True),
                dict(image='held.jpg',origin='legacy_train',sha256='c',label_sha256='old-c',weight=1,teacher_preserve=True,label_revised=False),
                dict(image='other.jpg',origin='legacy_train',sha256='d',label_sha256='old-d',weight=3,teacher_preserve=True,label_revised=False)]

    def approved(self):
        return {'new.jpg':dict(split='train',image_sha256='a',label_sha256='old-a',boxes_xyxy=[[10,20,30,40]])}

    def test_existing_approved_revision_survives_quarantine_with_exact_weight(self):
        rows=self.rows(); raw={r['image']:dict(sha256=r['sha256']) for r in rows[:3]}
        kept,excluded=membership(rows,raw,self.approved(),8)
        self.assertEqual([r['image'] for r in kept],['new.jpg','prior.jpg','other.jpg'])
        self.assertEqual([r['image'] for r in excluded],['held.jpg'])
        self.assertEqual(kept[0]['weight'],8)
        self.assertFalse(kept[0]['teacher_preserve'])
        self.assertTrue(kept[1]['retained_parent_revision'])
        self.assertEqual(kept[1]['label_sha256'],'reviewed-b')
        self.assertEqual(kept[1]['weight'],1)
        self.assertEqual(kept[2]['weight'],3)
        self.assertTrue(kept[2]['teacher_preserve'])

    def test_validation_approval_or_changed_label_is_rejected(self):
        rows=self.rows(); raw={r['image']:dict(sha256=r['sha256']) for r in rows[:3]}
        approval=self.approved(); approval['new.jpg']['split']='val'
        with self.assertRaisesRegex(ValueError,'Approved image/label differs'):
            membership(rows,raw,approval,8)
        approval=self.approved(); approval['new.jpg']['label_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'Approved image/label differs'):
            membership(rows,raw,approval,8)

    def test_clipped_visible_flame_label_stays_inside_image_and_not_empty(self):
        text=yolo_label([[603,338,640,396]])
        validate_yolo(text)
        _,x,y,w,h=map(float,text.split())
        self.assertAlmostEqual((x+w/2)*640,640,places=5)
        with self.assertRaisesRegex(ValueError,'positive must have boxes'):
            yolo_label([])


if __name__=='__main__': unittest.main()
