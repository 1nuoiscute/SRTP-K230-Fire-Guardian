import json
import tempfile
import unittest
from pathlib import Path
from data_integrity import sha256
from joint_pilot_dataset import converted_source_labels,verify_dataset


class JointDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.data=self.root/'data'; self.review=self.root/'review'
        self.review.mkdir(); rows=[]; images=[]
        pending=self.review/'review_pending.json'; decisions=self.review/'decisions.json'; sheet=self.review/'sheet.jpg'
        pending.write_text('{}'); decisions.write_text('{}'); sheet.write_bytes(b'reviewed-sheet-identity')
        for i,split in enumerate(('train','val'),1):
            (self.data/'images'/split).mkdir(parents=True); (self.data/'labels'/split).mkdir(parents=True)
            source=self.root/f'source{i}.jpg'; raw=self.root/f'source{i}.txt'
            source.write_bytes(f'original-image-{i}'.encode()); raw.write_text('0 .5 .5 .4 .4\n1 .3 .3 .2 .2\n')
            image=self.data/'images'/split/f'image{i}.jpg'; label=self.data/'labels'/split/f'image{i}.txt'
            image.write_bytes(source.read_bytes()); label.write_text(converted_source_labels(raw.read_text()),encoding='utf-8')
            rows.append({'index':i,'approved_for_training':True,'manual_review':{'decision':'approved_source_labels'},'errors':[],
                         'reviewed_development_group':f'group{i}','image':str(source),'sha256':sha256(source),
                         'label':str(raw),'label_sha256':sha256(raw)})
            images.append({'review_index':i,'group':f'group{i}','split':split,'file':image.name,
                           'image_sha256':sha256(image),'label_sha256':sha256(label)})
        self.complete=self.review/'review_complete.json'
        self.dump(self.complete,{'images':rows,'review_pending_sha256':sha256(pending),'decisions_sha256':sha256(decisions),
                                  'sheets':[{'file':sheet.name,'sha256':sha256(sheet)}]})
        self.manifest=self.data/'build_manifest.json'
        self.dump(self.manifest,{'source_review':str(self.complete),'source_review_sha256':sha256(self.complete),'images':images})
        self.config={'path':str(self.data),'train':'images/train','val':'images/val','nc':2,'names':{0:'fire',1:'smoke'}}

    def dump(self,path,value):path.write_text(json.dumps(value),encoding='utf-8')
    def load(self,path):return json.loads(path.read_text(encoding='utf-8'))

    def test_explicit_mapping_and_verified_dataset(self):
        text=converted_source_labels('0 .5 .5 .4 .4\n1 .3 .3 .2 .2\n')
        self.assertEqual([line.split()[0] for line in text.splitlines()],['1','0'])
        self.assertEqual(verify_dataset(self.data,self.config)['verified_images'],2)
        for text in ('2 .5 .5 .2 .2','0 nan .5 .2 .2','0 .99 .5 .4 .4'):
            with self.assertRaises(ValueError): converted_source_labels(text)

    def test_wrong_conversion_rejected_even_with_updated_label_hash(self):
        label=self.data/'labels/train/image1.txt'; label.write_text('0 .5 .5 .4 .4\n0 .3 .3 .2 .2\n',encoding='utf-8')
        m=self.load(self.manifest); m['images'][0]['label_sha256']=sha256(label); self.dump(self.manifest,m)
        with self.assertRaisesRegex(ValueError,'Class conversion'):verify_dataset(self.data,self.config)

    def test_reviewed_original_label_edit_rejected(self):
        (self.root/'source1.txt').write_text('0 .4 .4 .3 .3\n')
        with self.assertRaisesRegex(ValueError,'Original label'):verify_dataset(self.data,self.config)

    def test_review_group_cannot_cross_splits(self):
        review=self.load(self.complete); review['images'][1]['reviewed_development_group']='group1'; self.dump(self.complete,review)
        m=self.load(self.manifest); m['source_review_sha256']=sha256(self.complete); m['images'][1]['group']='group1'; self.dump(self.manifest,m)
        with self.assertRaisesRegex(ValueError,'crosses splits'):verify_dataset(self.data,self.config)


if __name__=='__main__':unittest.main()
