import copy
import json
import tempfile
import unittest
from pathlib import Path
from data_integrity import sha256
from dataset_preflight import verify_reviewed_dataset


class DatasetPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / "base"
        self.folder = self.root / "reviewed"
        for name in ["images/train", "labels/train"]:
            (self.folder / name).mkdir(parents=True)
        self.base.mkdir()
        (self.base / "manifest.csv").write_text("split,sha256\nval,unused-diagnostic-hash\n", encoding="utf-8")
        self.image = self.folder / "images/train/sample.jpg"
        self.label = self.folder / "labels/train/sample.txt"
        self.image.write_bytes(b"source image identity")
        self.label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        self.sheet = self.folder / "label_review_sheet.jpg"
        self.sheet.write_bytes(b"review sheet identity")
        self.train = self.folder / "train.txt"
        self.train.write_text(self.image.as_posix() + "\n", encoding="utf-8")
        self.meta = {"role": "development_only", "base": str(self.base), "train_entries": 1, "unique_paths": 1,
                     "lineage": [{"image": self.image.name, "origin": "reviewed_flame", "weight": 1,
                                  "sha256": sha256(self.image), "label_sha256": sha256(self.label)}]}
        self.manifest = self.folder / "build_manifest.json"
        self.manifest.write_text(json.dumps(self.meta), encoding="utf-8")
        (self.folder / "review_approval.json").write_text(json.dumps({"approved": True,
            "manifest_sha256": sha256(self.manifest), "sheet_sha256": sha256(self.sheet)}), encoding="utf-8")
        self.config = {"path": str(self.folder), "train": str(self.train), "val": str(self.base / "images/val"), "nc": 1, "names": {0: "fire"}}

    def test_approved_actual_inputs_verified(self):
        r = verify_reviewed_dataset(self.folder, self.config)
        self.assertEqual((r["images_verified"], r["entries_verified"], r["labels_bound_by_manifest"]), (1, 1, 1))

    def test_label_edit_after_review_rejected(self):
        self.label.write_text("0 0.4 0.4 0.4 0.4\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "label identity changed"):
            verify_reviewed_dataset(self.folder, self.config)

    def test_image_edit_after_review_rejected(self):
        self.image.write_bytes(b"different image")
        with self.assertRaisesRegex(ValueError, "image identity changed"):
            verify_reviewed_dataset(self.folder, self.config)

    def test_train_redirection_even_same_bytes_rejected(self):
        alternate = self.root / "unreviewed.jpg"
        alternate.write_bytes(self.image.read_bytes())
        self.train.write_text(alternate.as_posix() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "weight/path changed"):
            verify_reviewed_dataset(self.folder, self.config)

    def test_yaml_split_redirect_rejected(self):
        config = copy.deepcopy(self.config)
        config["val"] = str(self.base / "images/test")
        with self.assertRaisesRegex(ValueError, "config val differs"):
            verify_reviewed_dataset(self.folder, config)

    def test_sampling_weight_change_rejected(self):
        self.train.write_text((self.image.as_posix() + "\n") * 2, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "train counts changed"):
            verify_reviewed_dataset(self.folder, self.config)

    def test_bound_source_review_edit_rejected(self):
        review_dir = self.folder / "source_reviews"
        review_dir.mkdir()
        review = review_dir / "bundle.json"
        review.write_text("{\"scope\": \"reviewed\"}", encoding="utf-8")
        approval_path = self.folder / "review_approval.json"
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        approval["source_review_files_sha256"] = {review.name: sha256(review)}
        approval_path.write_text(json.dumps(approval), encoding="utf-8")
        verify_reviewed_dataset(self.folder, self.config)
        review.write_text("{\"scope\": \"changed\"}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source approval/acquisition file changed"):
            verify_reviewed_dataset(self.folder, self.config)

    def test_bound_new_source_sheet_edit_rejected(self):
        sheet = self.folder / "new_source_review_sheet.jpg"
        sheet.write_bytes(b"reviewed additions")
        approval_path = self.folder / "review_approval.json"
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        approval["new_source_sheet_sha256"] = sha256(sheet)
        approval_path.write_text(json.dumps(approval), encoding="utf-8")
        verify_reviewed_dataset(self.folder, self.config)
        sheet.write_bytes(b"changed additions")
        with self.assertRaisesRegex(ValueError, "new source sheet changed"):
            verify_reviewed_dataset(self.folder, self.config)

    def test_bound_negative_sample_sheet_edit_rejected(self):
        sheet = self.folder / "negative_sample_review_sheet.jpg"
        sheet.write_bytes(b"visually reviewed negatives")
        approval_path = self.folder / "review_approval.json"
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        approval["negative_sample_sheet_sha256"] = sha256(sheet)
        approval_path.write_text(json.dumps(approval), encoding="utf-8")
        verify_reviewed_dataset(self.folder, self.config)
        sheet.write_bytes(b"different negatives")
        with self.assertRaisesRegex(ValueError, "negative sample sheet changed"):
            verify_reviewed_dataset(self.folder, self.config)

    def test_review_sheet_edit_rejected(self):
        self.sheet.write_bytes(b"another review")
        with self.assertRaisesRegex(ValueError, "sheet identity changed"):
            verify_reviewed_dataset(self.folder, self.config)


if __name__ == "__main__":
    unittest.main()
