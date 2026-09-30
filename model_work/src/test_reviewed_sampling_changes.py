import copy,unittest
from reviewed_sampling_changes import validate_sampling_changes

class SamplingChangesTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{"image":"negative.jpg","origin":"legacy_train","sha256":"a","label_sha256":"b","weight":1}]
        self.review={"role":"development_negative_curation","approved":True,"parent_manifest_sha256":"parent","evidence_sha256":"evidence","changes":[{"image":"negative.jpg","sha256":"a","label_sha256":"b","decision":"reweight_negative","old_weight":1,"new_weight":12,"reason":"visually reviewed"}]}
    def check(self,r=None,rows=None,empty=None):
        return validate_sampling_changes(r or self.review,rows or self.rows,"parent","evidence",{"negative.jpg"} if empty is None else empty)
    def test_empty_train_reweight_and_explicit_exclusion(self):
        self.assertEqual(self.check()["negative.jpg"]["new_weight"],12)
        r=copy.deepcopy(self.review);r["changes"][0]["decision"]="exclude"
        self.assertEqual(self.check(r)["negative.jpg"]["decision"],"exclude")
    def test_pending_or_wrong_parent_evidence_rejected(self):
        for key,value in [("approved",False),("parent_manifest_sha256","other"),("evidence_sha256","edited")]:
            r=copy.deepcopy(self.review);r[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(r)
    def test_image_or_label_substitution_rejected(self):
        for key in ["sha256","label_sha256"]:
            r=copy.deepcopy(self.review);r["changes"][0][key]="edited"
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(r)
    def test_positive_label_cannot_be_reweighted_as_negative(self):
        with self.assertRaises(ValueError):self.check(empty=set())
    def test_bad_weight_and_changed_old_weight_rejected(self):
        for key,value in [("new_weight",True),("new_weight",0),("new_weight",999),("new_weight",1.5),("old_weight",2)]:
            r=copy.deepcopy(self.review);r["changes"][0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.check(r)
    def test_unknown_nonlegacy_duplicate_or_traversal_rejected(self):
        for name in ["missing.jpg","../negative.jpg","folder\\negative.jpg"]:
            r=copy.deepcopy(self.review);r["changes"][0]["image"]=name
            with self.subTest(name=name),self.assertRaises(ValueError):self.check(r)
        r=copy.deepcopy(self.review);r["changes"].append(r["changes"][0].copy())
        with self.assertRaises(ValueError):self.check(r)
        rows=copy.deepcopy(self.rows);rows[0]["origin"]="reviewed_flame"
        with self.assertRaises(ValueError):self.check(rows=rows)

if __name__=="__main__":unittest.main()
