import copy
import unittest
from source_sampling_plan import select_source_caps, verify_source_cap_revision


class SourceCapsTest(unittest.TestCase):
    def setUp(self):
        self.rows = [{"file": f"img{i}.jpg", "sha256": str(i), "split": "train",
                      "provenance": "stove" if i < 8 else "other"} for i in range(10)]
        self.lineage = [{"image": r["file"], "sha256": r["sha256"], "origin": "legacy_train",
                         "weight": 1, "label_sha256": "label"} for r in self.rows]
        self.lineage.append({"image": "blue.jpg", "sha256": "blue", "origin": "reviewed_blue", "weight": 12})

    def test_fixed_subset_unchanged_labels_and_other_sources(self):
        before = copy.deepcopy(self.lineage)
        kept, audit = select_source_caps(self.lineage, self.rows, {"stove": 3}, 42)
        self.assertEqual(self.lineage, before)
        self.assertEqual(len(kept), 6)
        self.assertEqual(audit["entries_after"]["derived:reviewed_blue"], 12)
        self.assertEqual(kept[-3:], self.lineage[-3:])
        reverse, other = select_source_caps(list(reversed(self.lineage)), self.rows, {"stove": 3}, 42)
        self.assertEqual({r["image"] for r in kept}, {r["image"] for r in reverse})
        self.assertEqual(audit, other)

    def test_test_image_and_duplicate_identity_rejected(self):
        for change in ("test", "duplicate"):
            rows = copy.deepcopy(self.rows)
            if change == "test": rows[0]["split"] = "test"
            else: rows.append(dict(rows[0]))
            with self.assertRaises(ValueError): select_source_caps(self.lineage, rows, {"stove": 3}, 42)

    def test_invalid_caps(self):
        for caps in ({}, {"stove": 0}, {"stove": True}, {"stove": 8}, {"unknown": 1}):
            with self.assertRaises(ValueError): select_source_caps(self.lineage, self.rows, caps, 42)

    def test_revision_cannot_silently_change_weight_label_or_sampling(self):
        kept, audit = select_source_caps(self.lineage, self.rows, {"stove": 3}, 42)
        parent = {"lineage": self.lineage, "base": "base", "role": "development_only"}
        meta = {**parent, "lineage": kept, "unique_paths": len(kept),
                "train_entries": sum(r["weight"] for r in kept),
                "source_sampling": {"caps": {"stove": 3}, "seed": 42, "audit": audit}}
        verify_source_cap_revision(meta, parent, self.rows)
        for field, value in (("weight", 2), ("label_sha256", "changed")):
            bad = copy.deepcopy(meta); bad["lineage"][0][field] = value
            with self.assertRaises(ValueError): verify_source_cap_revision(bad, parent, self.rows)
        bad = copy.deepcopy(meta); bad["source_sampling"]["seed"] = 43
        with self.assertRaises(ValueError): verify_source_cap_revision(bad, parent, self.rows)


if __name__ == "__main__": unittest.main()
