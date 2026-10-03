import json
import unittest
from pathlib import Path
import yaml
from data_integrity import sha256
from smoke_head_dataset import smoke_only_labels, build_dataset, verify_smoke_dataset
import test_joint_pilot_dataset as joint_fixture


class SmokeHeadDatasetTests(unittest.TestCase):
    def setUp(self):
        self.fixture=joint_fixture.JointDatasetTests(methodName='runTest'); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        (self.fixture.data/'data.yaml').write_text(yaml.safe_dump(self.fixture.config),encoding='utf-8')
        self.child=self.fixture.root/'smoke_child'

    def test_only_smoke_truth_converted_and_same_groups_kept(self):
        self.assertEqual(smoke_only_labels('0 .3 .3 .2 .2\n1 .5 .5 .4 .4\n'),'0 0.50000000 0.50000000 0.40000000 0.40000000\n')
        result=build_dataset(self.fixture.data,sha256(self.fixture.manifest),self.child)
        self.assertEqual(result['verified_images'],2)
        for split in ('train','val'):
            self.assertEqual(len(list((self.child/'labels'/split).glob('*.txt'))),1)

    def test_wrong_labels_rejected_even_with_rebound_hash(self):
        build_dataset(self.fixture.data,sha256(self.fixture.manifest),self.child)
        path=self.child/'labels/train/image1.txt'; path.write_text('0 .3 .3 .2 .2\n',encoding='utf-8')
        mpath=self.child/'build_manifest.json'; manifest=json.loads(mpath.read_text()); manifest['images'][0]['label_sha256']=sha256(path)
        mpath.write_text(json.dumps(manifest),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'conversion'): verify_smoke_dataset(self.child,sha256(mpath))

    def test_changed_parent_and_child_class_config_rejected(self):
        result=build_dataset(self.fixture.data,sha256(self.fixture.manifest),self.child)
        cfg=self.child/'data.yaml'; original=cfg.read_text(encoding='utf-8')
        cfg.write_text(yaml.safe_dump({**self.fixture.config,'path':str(self.child)}),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'config'): verify_smoke_dataset(self.child,result['manifest_sha256'])
        cfg.write_text(original,encoding='utf-8'); (self.fixture.root/'source1.txt').write_text('0 .4 .4 .3 .3\n')
        with self.assertRaisesRegex(ValueError,'Original label'): verify_smoke_dataset(self.child,result['manifest_sha256'])


if __name__=='__main__':unittest.main()
