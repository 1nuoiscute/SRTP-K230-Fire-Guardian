"""Prepare reviewed Commons additions; never start training or overwrite inputs."""
import argparse
import copy
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw
from data_integrity import sha256, validate_yolo

ROOT = Path(__file__).resolve().parents[2]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def label_for(boxes, width, height):
    text = ""
    for x1, y1, x2, y2 in boxes:
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Reviewed box is outside source dimensions")
        text += f"0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}\n"
    validate_yolo(text)
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--candidates", type=Path, default=ROOT / "model_work/data/blue_candidates_20260930")
    p.add_argument("--review", type=Path, default=ROOT / "docs/blue_candidate_label_review_20260930.json")
    p.add_argument("--weight", type=int, default=12)
    args = p.parse_args()
    parent, out = args.parent.resolve(), args.out.resolve()
    if out.exists():
        raise SystemExit(f"Refusing overwrite: {out}")
    if not 1 <= args.weight <= 32:
        raise SystemExit("Weight must be between 1 and 32")
    manifest_path = parent / "build_manifest.json"
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    approval_path = parent / "review_approval.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    sheet = parent / "label_review_sheet.jpg"
    if not approval.get("approved") or approval["manifest_sha256"] != sha256(manifest_path) or approval["sheet_sha256"] != sha256(sheet):
        raise SystemExit("Parent visual review identity mismatch")
    entries = [Path(v) for v in (parent / "train.txt").read_text(encoding="utf-8").splitlines() if v.strip()]
    counts = Counter(str(v.resolve()) for v in entries)
    if len(entries) != meta["train_entries"] or len(counts) != meta["unique_paths"]:
        raise SystemExit("Parent train list counts changed")
    by_name = {v.name: v.resolve() for v in entries}
    if len(by_name) != len(counts):
        raise SystemExit("Ambiguous duplicate source filenames")
    base_rows = list(csv.DictReader((Path(meta["base"]) / "manifest.csv").open(encoding="utf-8-sig")))
    excluded = {row["sha256"] for row in base_rows if row["split"] != "train"}
    lineage = copy.deepcopy(meta["lineage"])
    validated = []
    for row in lineage:
        image = by_name[row["image"]]
        label = image.parent.parent.parent / "labels" / image.parent.name / (image.stem + ".txt")
        if sha256(image) != row["sha256"] or row["sha256"] in excluded:
            raise SystemExit("Parent image changed or diagnostic overlap")
        if counts[str(image)] != row["weight"]:
            raise SystemExit("Parent sample weight changed")
        text = label.read_text(encoding="utf-8")
        validate_yolo(text)
        if row["origin"] != "legacy_train":
            with Image.open(image) as im:
                w, h = im.size
            if text != label_for(row["boxes_xyxy"], w, h):
                raise SystemExit("Parent derived label differs from reviewed box")
        row["label_sha256"] = sha256(label)
        validated.append((row, image, label))
    reviews = json.loads(args.review.read_text(encoding="utf-8"))
    additions = []
    existing_hashes = {row["sha256"] for row in lineage}
    for row in reviews:
        if row["decision"] != "approved_for_development_training":
            continue
        image = args.candidates / row["file"]
        digest = sha256(image)
        if digest != row["sha256"] or digest in excluded or digest in existing_hashes:
            raise SystemExit("Candidate identity/overlap failure")
        with Image.open(image) as im:
            if im.size != (row["width"], row["height"]):
                raise SystemExit("Candidate dimensions changed")
        text = label_for(row["boxes_xyxy"], row["width"], row["height"])
        if not text:
            raise SystemExit("Approved source has no reviewed flame")
        existing_hashes.add(digest)
        additions.append((row, image, text))
    if not additions:
        raise SystemExit("No approved new sources")
    (out / "images/train").mkdir(parents=True)
    (out / "labels/train").mkdir(parents=True)
    replacements = {}
    for row, image, label in validated:
        if row["origin"] == "legacy_train":
            continue
        target = out / "images/train" / image.name
        shutil.copyfile(image, target)
        shutil.copyfile(label, out / "labels/train" / label.name)
        replacements[str(image)] = target
    train = [replacements.get(str(v.resolve()), v.resolve()).as_posix() for v in entries]
    review_dir = out / "source_label_review"
    review_dir.mkdir()
    for row, image, text in additions:
        target = out / "images/train" / ("commons_blue_" + image.name)
        label = out / "labels/train" / (target.stem + ".txt")
        shutil.copyfile(image, target)
        label.write_text(text, encoding="utf-8")
        train.extend([target.as_posix()] * args.weight)
        new = {k: row[k] for k in ("sha256", "width", "height", "source_page", "author", "license", "license_url", "boxes_xyxy", "review_method")}
        new.update(image=target.name, origin="commons_reviewed_blue", source_group=row["source_page"], weight=args.weight, label_sha256=sha256(label))
        lineage.append(new)
        with Image.open(target) as im:
            annotated = im.convert("RGB")
            draw = ImageDraw.Draw(annotated)
            for box in row["boxes_xyxy"]:
                draw.rectangle(box, outline="lime", width=4)
            annotated.thumbnail((1280, 1000))
            annotated.save(review_dir / target.name)
    shutil.copyfile(sheet, out / "label_review_sheet.jpg")
    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(f'path: {out.as_posix()}\ntrain: {(out/"train.txt").as_posix()}\nval: {(Path(meta["base"])/"images/val").as_posix()}\nnc: 1\nnames:\n  0: fire\n', encoding="utf-8")
    meta.update(train_entries=len(train), unique_paths=len(set(train)), derived_images=sum(v["origin"] != "legacy_train" for v in lineage), lineage=lineage,
                parent_manifest_sha256=sha256(manifest_path), parent_review_sha256=sha256(approval_path),
                source_review_sha256=sha256(args.review), inherited_sheet_sha256=sha256(sheet),
                added_sources=len(additions), added_train_entries=len(additions)*args.weight,
                review_required="Verify the two new source_label_review images before setting approved=true; inherited 44 reviews stay byte-identical.")
    meta["limitations"].append("Two added Commons photographs are development training; duplicate weight does not create new scenes.")
    dump(out / "build_manifest.json", meta)
    dump(out / "review_approval.json", {"approved": False, "manifest_sha256": sha256(out / "build_manifest.json"), "sheet_sha256": sha256(out / "label_review_sheet.jpg"), "reason": "Pending final review of copied new source annotations"})
    print(json.dumps({k:v for k,v in meta.items() if k not in ("base", "lineage")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
