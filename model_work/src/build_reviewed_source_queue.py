"""Extend an approved development dataset with reviewed source queues.

Preserves the parent, excludes held photos, binds provenance, and never trains.
The new dataset remains unapproved until copied annotations are viewed again.
"""
import argparse
import copy
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

from build_reviewed_blue_extension import dump, label_for
from data_integrity import sha256
from dataset_preflight import verify_reviewed_dataset
from source_exposure import commons_source_key, training_source_keys
from source_metadata import checked_metadata_corrections


def basename(value):
    if not value or Path(value).name != value:
        raise ValueError("Source filename must be a basename")
    return value


def reviewed_blue(queue, audit_path, parent):
    manifest_path = queue / "manifest.json"
    review_path = queue / "label_review_approval.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review["source_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Acquisition manifest changed after review")
    if review["label_sheet_sha256"] != sha256(queue / "label_review_sheet.jpg"):
        raise ValueError("Source label sheet changed after review")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["missing_training_files"] or not audit["training_images_scanned"] or audit["training_byte_hashes_verified"] != audit["training_images_scanned"]:
        raise ValueError("Incomplete available training-chain audit")
    acquired = {v["file"]: v for v in json.loads(manifest_path.read_text(encoding="utf-8"))}
    audit_rows = {v["file"]: v for v in audit["candidates"]}
    existing_hashes = {v["sha256"] for v in parent["lineage"]}
    source_keys = training_source_keys(parent)
    rows = []
    for item in review["sources"]:
        if item["decision"] != "approved_for_development_training":
            continue
        item = copy.deepcopy(item)
        name = basename(item["file"])
        image = queue / name
        digest = sha256(image)
        original = acquired[name]
        item["metadata_corrections"] = checked_metadata_corrections(item, original)
        key = commons_source_key(item["source_page"])
        if digest != item["sha256"] or digest in existing_hashes or key in source_keys:
            raise ValueError("Approved source duplicates parent bytes/original work or was edited")
        checked = audit_rows[name]
        if checked["sha256"] != digest or checked["exact_training_match"] or checked["original_work_seen_in_diagnostics"]:
            raise ValueError("Approved candidate disagrees with source overlap audit")
        with Image.open(image) as im:
            if im.size != (item["width"], item["height"]):
                raise ValueError("Approved source dimensions changed")
        text = label_for(item["boxes_xyxy"], item["width"], item["height"])
        if not text or not item.get("scene_group") or not item.get("author"):
            raise ValueError("Reviewed blue source lacks flame boxes, scene group or attribution")
        existing_hashes.add(digest)
        source_keys.add(key)
        rows.append((item, image, text))
    if len(rows) != review["approved_photographs"] or sum(len(v[0]["boxes_xyxy"]) for v in rows) != review["approved_boxes"]:
        raise ValueError("Approved source counts changed")
    if not rows:
        raise ValueError("No approved blue sources")
    return rows


def reviewed_negatives(queue, parent_folder, parent):
    review_path = queue / "review_approval.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("approved") is not True or review.get("development_only") is not True:
        raise ValueError("Negative queue is not approved development data")
    if review["pending_manifest_sha256"] != sha256(queue / "review_pending.json") or review["crop_context_sha256"] != sha256(queue / "crop_context.jpg"):
        raise ValueError("Negative queue review artifact changed")
    by_name = {v["image"]: v for v in parent["lineage"]}
    hashes = {v["sha256"] for v in parent["lineage"]}
    rows = []
    for item in review["items"]:
        item = copy.deepcopy(item)
        image = queue / "images" / basename(item["image"])
        label = queue / "labels" / basename(item["label"])
        if sha256(image) != item["image_sha256"] or sha256(label) != item["label_sha256"] or label.read_text(encoding="utf-8").strip() or item["expected_boxes_xyxy"]:
            raise ValueError("Reviewed empty negative image/label changed")
        source_name = basename(item["source_image"])
        source_row = by_name[source_name]
        source = parent_folder / "images/train" / source_name
        if item["source_dataset_manifest_sha256"] != sha256(parent_folder / "build_manifest.json") or item["source_image_sha256"] != source_row["sha256"] or sha256(source) != item["source_image_sha256"]:
            raise ValueError("Negative parent source identity changed")
        original, cropped = cv2.imread(str(source)), cv2.imread(str(image))
        if original is None or cropped is None:
            raise ValueError("Negative source cannot be decoded")
        x1, y1, x2, y2 = item["crop_xyxy"]
        if not all(isinstance(v, int) for v in item["crop_xyxy"]) or not (0 <= x1 < x2 <= original.shape[1] and 0 <= y1 < y2 <= original.shape[0]):
            raise ValueError("Negative crop is outside source bounds")
        if not np.array_equal(cropped, original[y1:y2, x1:x2]):
            raise ValueError("Negative image is not the reviewed lossless crop")
        for gx1, gy1, gx2, gy2 in source_row["boxes_xyxy"]:
            if min(x2, gx2) > max(x1, gx1) and min(y2, gy2) > max(y1, gy1):
                raise ValueError("Negative crop intersects reviewed flame GT")
        if item["image_sha256"] in hashes:
            raise ValueError("Negative crop duplicates existing training bytes")
        hashes.add(item["image_sha256"])
        rows.append((item, image, ""))
    if not rows:
        raise ValueError("No approved negative crops")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--blue-queue", type=Path, required=True)
    p.add_argument("--blue-audit", type=Path, required=True)
    p.add_argument("--negative-queue", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--blue-weight", type=int, default=12)
    p.add_argument("--negative-weight", type=int, default=8)
    args = p.parse_args()
    parent, out = args.parent.resolve(), args.out.resolve()
    if out.exists():
        raise SystemExit(f"Refusing overwrite: {out}")
    if not all(1 <= v <= 32 for v in (args.blue_weight, args.negative_weight)):
        raise SystemExit("Sampling weights must be between 1 and 32")
    config = yaml.safe_load((parent / "data.yaml").read_text(encoding="utf-8"))
    verified = verify_reviewed_dataset(parent, config)
    meta = json.loads((parent / "build_manifest.json").read_text(encoding="utf-8"))
    blue = reviewed_blue(args.blue_queue.resolve(), args.blue_audit.resolve(), meta)
    negative = reviewed_negatives(args.negative_queue.resolve(), parent, meta)
    lineage = copy.deepcopy(meta["lineage"])
    base = Path(meta["base"]).resolve()
    with (base / "manifest.csv").open(encoding="utf-8-sig") as stream:
        excluded = {v["sha256"] for v in csv.DictReader(stream) if v["split"] != "train"}
    if any(sha256(image) in excluded for _, image, _ in blue + negative):
        raise ValueError("Added training image overlaps a legacy validation/test split")
    (out / "images/train").mkdir(parents=True)
    (out / "labels/train").mkdir(parents=True)
    for row in lineage:
        if row["origin"] != "legacy_train":
            for kind, name in (("images", row["image"]), ("labels", Path(row["image"]).stem + ".txt")):
                shutil.copyfile(parent / kind / "train" / name, out / kind / "train" / name)
    shutil.copyfile(parent / "label_review_sheet.jpg", out / "label_review_sheet.jpg")
    shutil.copytree(parent / "source_label_review", out / "source_label_review")
    review_dir = out / "source_label_review"
    for kind, rows, weight in (("blue", blue, args.blue_weight), ("negative", negative, args.negative_weight)):
        for item, image, text in rows:
            name = ("commons_" if kind == "blue" else "") + image.name
            target, label = out / "images/train" / name, out / "labels/train" / (Path(name).stem + ".txt")
            if target.exists():
                raise ValueError("New source basename collides with copied parent")
            shutil.copyfile(image, target)
            label.write_text(text, encoding="utf-8")
            new = copy.deepcopy(item)
            new.update(image=name, sha256=sha256(target), label_sha256=sha256(label), weight=weight,
                       origin="commons_reviewed_blue" if kind == "blue" else "food_hard_negative_crop",
                       source_group=item["scene_group"] if kind == "blue" else item["source_group"],
                       boxes_xyxy=item["boxes_xyxy"] if kind == "blue" else [])
            lineage.append(new)
            with Image.open(target) as im:
                annotated = im.convert("RGB")
                draw = ImageDraw.Draw(annotated)
                for box in new["boxes_xyxy"]:
                    draw.rectangle(box, outline="lime", width=4)
                annotated.thumbnail((1280, 1000))
                annotated.save(review_dir / (target.stem + ".jpg"))
    review_files = [("blue_acquisition.json", args.blue_queue / "manifest.json"),
                    ("blue_approval.json", args.blue_queue / "label_review_approval.json"),
                    ("blue_overlap_audit.json", args.blue_audit),
                    ("negative_approval.json", args.negative_queue / "review_approval.json"),
                    ("parent_approval.json", parent / "review_approval.json")]
    (out / "source_reviews").mkdir()
    review_hashes = {}
    for name, source in review_files:
        shutil.copyfile(source, out / "source_reviews" / name)
        review_hashes[name] = sha256(out / "source_reviews" / name)
    dump(out / "source_reviews/bundle.json", review_hashes)
    train = []
    for row in lineage:
        folder = base if row["origin"] == "legacy_train" else out
        train.extend([(folder / "images/train" / row["image"]).as_posix()] * row["weight"])
    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    config.update(path=out.as_posix(), train=(out / "train.txt").as_posix())
    (out / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    meta.update(lineage=lineage, train_entries=len(train), unique_paths=len(set(train)),
                derived_images=sum(v["origin"] != "legacy_train" for v in lineage),
                parent_manifest_sha256=verified["manifest_sha256"], parent_review_sha256=verified["approval_sha256"],
                source_review_sha256=sha256(out / "source_reviews/bundle.json"),
                added_sources=len(blue), added_negative_crops=len(negative),
                added_train_entries=len(blue) * args.blue_weight + len(negative) * args.negative_weight,
                provisional_added_scene_groups=len({v[0]["scene_group"] for v in blue}),
                review_required=f"Review {len(blue)} copied blue-source annotations and {len(negative)} negatives before approving; no training is started.")
    meta["limitations"].append("Queue2 additions are exposed development sources; negative crop reuses an existing video scene. Scene groups and sampling repetitions are not independent kitchen counts.")
    dump(out / "build_manifest.json", meta)
    dump(out / "review_approval.json", {"approved": False, "manifest_sha256": sha256(out / "build_manifest.json"),
        "sheet_sha256": sha256(out / "label_review_sheet.jpg"), "source_review_sha256": meta["source_review_sha256"],
        "source_review_files_sha256": {**review_hashes, "bundle.json": meta["source_review_sha256"]},
        "source_rendering_sha256": {v.name: sha256(v) for v in sorted(review_dir.iterdir())},
        "reason": "Pending visual review of final copied source annotations. Does not start training."})
    print(json.dumps({k: v for k, v in meta.items() if k not in ("lineage", "base")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
