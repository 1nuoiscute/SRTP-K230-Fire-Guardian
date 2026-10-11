import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml
from data_integrity import sha256
from fire_label_revision_dataset import build, verify


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base, self.parent, self.review_dir = [self.root / n for n in ('base', 'parent', 'review')]
        for root in (self.base, self.parent):
            (root / 'images/train').mkdir(parents=True)
            (root / 'labels/train').mkdir(parents=True)
        (self.base / 'images/val').mkdir(parents=True)
        self.review_dir.mkdir()
        lineage = []
        for name, root, origin in (('old.jpg', self.base, 'legacy_train'), ('blue.jpg', self.parent, 'commons_reviewed_blue')):
            image = root / 'images/train' / name
            label = root / 'labels/train' / (image.stem + '.txt')
            image.write_bytes(name.encode())
            label.write_text('0 .5 .5 .4 .4\n')
            lineage.append(dict(image=name, sha256=sha256(image), label_sha256=sha256(label),
                                origin=origin, source_group=name, weight=2))
        (self.parent / 'build_manifest.json').write_text(json.dumps(dict(base=str(self.base), lineage=lineage)))
        (self.parent / 'data.yaml').write_text(yaml.safe_dump({}))
        proposed = self.review_dir / 'new.txt'
        proposed.write_text('0 .5 .5 .2 .2\n')
        image, label = self.base / 'images/train/old.jpg', self.base / 'labels/train/old.txt'
        self.review = self.review_dir / 'review.json'
        self.review.write_text(json.dumps(dict(sheets=[], images=[dict(image=str(image), image_sha256=sha256(image),
            original_label=str(label), original_label_sha256=sha256(label), proposed_label=str(proposed),
            proposed_label_sha256=sha256(proposed), approved_for_training=True)])))
        self.identity = {'manifest_sha256': 'fixed-parent'}
        self.mock = patch('fire_label_revision_dataset.verify_reviewed_dataset', return_value=self.identity)
        self.mock.start()
        self.addCleanup(self.mock.stop)
        self.out = self.root / 'child'

    def built(self):
        result = build(self.parent, self.review, self.out)
        return result['manifest_sha256']

    def test_copy_preserves_parent_and_exempts_revised_and_new(self):
        digest = self.built()
        self.assertEqual((self.base / 'labels/train/old.txt').read_text(), '0 .5 .5 .4 .4\n')
        self.assertEqual((self.out / 'labels/train/old.txt').read_text(), '0 .5 .5 .2 .2\n')
        result = verify(self.out, digest)
        self.assertEqual(result['entries_verified'], 4)
        self.assertEqual(result['teacher_preserved_images'], 0)

    def test_unapproved_changed_label_is_rejected(self):
        digest = self.built()
        (self.out / 'labels/train/blue.txt').write_text('')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            verify(self.out, digest)

    def test_sampling_repetition_change_is_rejected(self):
        digest = self.built()
        with (self.out / 'train.txt').open('a') as stream:
            stream.write(str(self.out / 'images/train/old.jpg') + '\n')
        with self.assertRaisesRegex(ValueError, 'count or weight'):
            verify(self.out, digest)

    def test_revision_instance_count_change_is_rejected(self):
        meta = json.loads(self.review.read_text())
        new = Path(meta['images'][0]['proposed_label'])
        new.write_text('0 .5 .5 .2 .2\n0 .8 .8 .1 .1\n')
        meta['images'][0]['proposed_label_sha256'] = sha256(new)
        self.review.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'instance count'):
            build(self.parent, self.review, self.out)


if __name__ == '__main__':
    unittest.main()
