"""Dependency-free preflight binding a reviewed dataset to actual training inputs."""
import csv
import json
from collections import Counter
from pathlib import Path
from data_integrity import sha256, validate_yolo


def verify_reviewed_dataset(folder, config):
    folder = Path(folder).resolve()
    manifest_path = folder / "build_manifest.json"
    approval_path = folder / "review_approval.json"
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if meta.get("role") != "development_only" or approval.get("approved") is not True:
        raise ValueError("Dataset is not approved development data")
    if approval["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Reviewed manifest identity changed")
    if approval["sheet_sha256"] != sha256(folder / "label_review_sheet.jpg"):
        raise ValueError("Reviewed sheet identity changed")
    if "source_review_sha256" in meta and approval.get("source_review_sha256") != meta["source_review_sha256"]:
        raise ValueError("Source review identity differs")
    for name, digest in approval.get("source_rendering_sha256", {}).items():
        if Path(name).name != name or sha256(folder / "source_label_review" / name) != digest:
            raise ValueError("Reviewed source rendering changed")
    base = Path(meta["base"]).resolve()
    expected = {"path": folder, "train": folder / "train.txt", "val": base / "images/val"}
    for key, target in expected.items():
        if Path(config[key]).resolve() != target:
            raise ValueError(f"Training config {key} differs from reviewed dataset")
    if config.get("nc") != 1 or config.get("names") != {0: "fire"}:
        raise ValueError("Training config class definition changed")
    entries = [Path(v).resolve() for v in (folder / "train.txt").read_text(encoding="utf-8").splitlines() if v.strip()]
    counts = Counter(entries)
    lineage = meta["lineage"]
    if len(entries) != meta["train_entries"] or len(counts) != meta["unique_paths"] or len(lineage) != len(counts):
        raise ValueError("Reviewed train counts changed")
    with (base / "manifest.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    excluded = {r["sha256"] for r in rows if r["split"] != "train"}
    verified = set()
    bound_labels = 0
    for row in lineage:
        name = row["image"]
        if Path(name).name != name:
            raise ValueError("Source image must be a basename")
        source = base if row["origin"] == "legacy_train" else folder
        image = source / "images/train" / name
        if image in verified or counts[image] != row["weight"]:
            raise ValueError("Reviewed sample weight/path changed")
        verified.add(image)
        digest = sha256(image)
        if digest != row["sha256"] or digest in excluded:
            raise ValueError("Training image identity changed or overlaps diagnostic split")
        label = source / "labels/train" / (image.stem + ".txt")
        validate_yolo(label.read_text(encoding="utf-8"))
        if "label_sha256" in row:
            if sha256(label) != row["label_sha256"]:
                raise ValueError("Reviewed label identity changed")
            bound_labels += 1
    if verified != set(entries):
        raise ValueError("Train list contains an unreviewed path")
    return {"manifest_sha256": sha256(manifest_path), "approval_sha256": sha256(approval_path),
            "train_list_sha256": sha256(folder / "train.txt"), "base_manifest_sha256": sha256(base / "manifest.csv"),
            "images_verified": len(verified), "entries_verified": len(entries),
            "labels_bound_by_manifest": bound_labels,
            "scope": "Bytes, labels, weights and config checked; not proof of scene independence."}
