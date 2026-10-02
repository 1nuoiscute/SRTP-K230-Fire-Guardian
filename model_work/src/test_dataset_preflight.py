import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
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

    def make_capped_child(self):
        import yaml
        from build_source_capped_dataset import main
        for name in ["images/train", "labels/train"]:
            (self.base / name).mkdir(parents=True)
        records = []
        for i in range(4):
            image = self.base / "images/train" / f"stove{i}.jpg"
            label = self.base / "labels/train" / f"stove{i}.txt"
            image.write_bytes(f"stove pixels {i}".encode())
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            records.append(f"train,{image.name},{sha256(image)},stove")
            self.meta["lineage"].append({"image": image.name, "origin": "legacy_train", "weight": 1,
                                          "sha256": sha256(image), "label_sha256": sha256(label)})
        (self.base / "manifest.csv").write_text("split,file,sha256,provenance\n" + "\n".join(records) + "\n", encoding="utf-8")
        self.meta.update(unique_paths=5, train_entries=5)
        self.manifest.write_text(json.dumps(self.meta), encoding="utf-8")
        approval_path = self.folder / "review_approval.json"
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        approval["manifest_sha256"] = sha256(self.manifest)
        approval_path.write_text(json.dumps(approval), encoding="utf-8")
        paths = [self.image, *(self.base / "images/train" / f"stove{i}.jpg" for i in range(4))]
        self.train.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
        (self.folder / "data.yaml").write_text(yaml.safe_dump(self.config), encoding="utf-8")
        out = self.root / "capped"
        argv = ["build_source_capped_dataset.py", "--parent", str(self.folder), "--out", str(out),
                "--source", "stove", "--cap", "2", "--rationale", "sampling test"]
        with patch("sys.argv", argv), redirect_stdout(StringIO()):
            main()
        return out, yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))

    def test_capped_child_checks_parent_and_actual_subset(self):
        child, config = self.make_capped_child()
        report = verify_reviewed_dataset(child, config)
        self.assertEqual(report["images_verified"], 3)
        self.assertEqual(report["source_sampling_parent"]["images_verified"], 5)
        # Alter even an excluded parent input: the child's provenance still binds it.
        meta = json.loads((child / "build_manifest.json").read_text(encoding="utf-8"))
        kept = {r["image"] for r in meta["lineage"]}
        excluded = next(r for r in self.meta["lineage"] if r["image"] not in kept)
        (self.base / "labels/train" / Path(excluded["image"]).with_suffix(".txt")).write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "label identity changed"):
            verify_reviewed_dataset(child, config)

    def test_capped_parent_cycle_rejected(self):
        child, config = self.make_capped_child()
        manifest = child / "build_manifest.json"
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        meta["source_sampling"]["parent_folder"] = str(child)
        manifest.write_text(json.dumps(meta), encoding="utf-8")
        ap = child / "review_approval.json"
        approval = json.loads(ap.read_text(encoding="utf-8"))
        approval.update(manifest_sha256=sha256(manifest), source_sampling_sha256=sha256(manifest))
        ap.write_text(json.dumps(approval), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "parent cycle"):
            verify_reviewed_dataset(child, config)


if __name__ == "__main__":
    unittest.main()
