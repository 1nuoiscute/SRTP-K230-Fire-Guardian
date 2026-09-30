import json
import tempfile
import unittest
from pathlib import Path
from data_integrity import matches, reject_exposed, sha256, validate_yolo

class IntegrityTests(unittest.TestCase):
    def test_duplicate_detection_is_false_positive(self):
        self.assertEqual(matches([[0,0,10,10]],[[0,0,10,10],[0,0,10,10]]),{'tp':1,'fp':1,'fn':0})
    def test_matching_reassigns_ambiguous_box(self):
        self.assertEqual(matches([[0,0,10,10],[3,0,13,10]],[[2,0,12,10],[0,0,7,10]]),{'tp':2,'fp':0,'fn':0})
    def test_empty_truth_is_negative(self):
        self.assertEqual(matches([],[[0,0,10,10]]),{'tp':0,'fp':1,'fn':0})
    def test_label_bounds_rejected(self):
        with self.assertRaises(ValueError): validate_yolo('0 0.1 0.5 0.6 0.2')
        validate_yolo('0 0.5 0.5 1 1\n')
        validate_yolo('')
    def test_exposed_file_rejected_even_if_renamed(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'renamed.mp4';p.write_bytes(b'known media')
            registry=Path(folder)/'registry.json'
            registry.write_text(json.dumps({'videos':[{'sha256':sha256(p)}]}))
            with self.assertRaisesRegex(ValueError,'Development-exposed'): reject_exposed(p,registry)
            p.write_bytes(b'new media')
            self.assertEqual(reject_exposed(p,registry),sha256(p))
    def test_missing_exposure_registry_rejected(self):
        with self.assertRaisesRegex(ValueError,'registry missing'):
            reject_exposed(Path('not-needed'),Path('nonexistent-registry.json'))

if __name__=='__main__': unittest.main()
