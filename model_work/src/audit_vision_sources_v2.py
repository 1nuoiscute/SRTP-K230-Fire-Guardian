"""Read-only inventory of the local SRTP vision sources.

Run with --out pointing to a new directory. Existing output is never replaced.
This audit inventories labels; it does not certify their visual correctness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(r"C:\Users\ASUS\Documents\Codex\kitchen-fire-detection\datasets")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SOURCES = {
    "fire_flame": ROOT / "fire-flame.v1i.yolov11",
    "stove": ROOT / "stove-detection",
    "pan": ROOT / "pan-detection",
    "kitchen": ROOT / "kitchen-images",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_label(path: Path) -> tuple[Counter[str], list[str]]:
    counts: Counter[str] = Counter()
    errors: list[str] = []
    if not path.is_file():
        return counts, ["missing_label"]
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5 or (len(parts) > 5 and (len(parts) - 1) % 2):
            errors.append(f"line_{line_no}_field_count")
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"line_{line_no}_non_numeric")
            continue
        if class_id < 0 or any(value < 0 or value > 1 for value in values):
            errors.append(f"line_{line_no}_out_of_range")
        if len(values) == 4 and (values[2] <= 0 or values[3] <= 0):
            errors.append(f"line_{line_no}_nonpositive_box")
        counts[str(class_id)] += 1
    return counts, errors


def source_images(source: str, root: Path):
    if source == "kitchen":
        for category in ("big_flame", "kitchen_fire", "normal_cooking", "no_fire"):
            for image in sorted((root / category).iterdir()):
                if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
                    yield category, image, image.with_suffix(".txt")
        return
    for split in ("train", "valid", "test"):
        for image in sorted((root / split / "images").iterdir()):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
                yield split, image, root / split / "labels" / f"{image.stem}.txt"


def audit(output: Path) -> None:
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    rows: list[dict[str, str | int]] = []
    per_source: dict[str, dict] = {}
    by_hash: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    by_group: dict[tuple[str, str], set[str]] = defaultdict(set)

    for source, root in SOURCES.items():
        if not root.is_dir():
            raise FileNotFoundError(root)
        split_counts: Counter[str] = Counter()
        class_counts: Counter[str] = Counter()
        empty_counts: Counter[str] = Counter()
        error_counts: Counter[str] = Counter()
        for split, image, label in source_images(source, root):
            labels, errors = inspect_label(label)
            try:
                with Image.open(image) as opened:
                    width, height = opened.size
                    opened.verify()
            except Exception as exc:
                width, height = 0, 0
                errors.append(f"image_decode:{type(exc).__name__}")
            checksum = sha256(image)
            source_group = image.stem.split(".rf.", 1)[0]
            split_counts[split] += 1
            class_counts.update(labels)
            if not labels:
                empty_counts[split] += 1
            error_counts.update(errors)
            by_hash[checksum].append((source, split, str(image)))
            by_group[(source, source_group)].add(split)
            rows.append({
                "source": source,
                "split_or_category": split,
                "source_group": source_group,
                "image": str(image),
                "label": str(label),
                "sha256": checksum,
                "width": width,
                "height": height,
                "label_counts": json.dumps(dict(labels), sort_keys=True),
                "label_errors": ";".join(errors),
            })
        per_source[source] = {
            "images_by_split_or_category": dict(split_counts),
            "boxes_by_source_class_id": dict(class_counts),
            "empty_or_missing_labels_by_split_or_category": dict(empty_counts),
            "errors": dict(error_counts),
        }

    duplicate_hashes = [items for items in by_hash.values() if len(items) > 1]
    cross_split_groups = [
        {"source": source, "group": group, "splits": sorted(splits)}
        for (source, group), splits in by_group.items()
        if len(splits) > 1
    ]
    kitchen_rows = [row for row in rows if row["source"] == "kitchen"]
    summary = {
        "source_roots": {source: str(path) for source, path in SOURCES.items()},
        "total_images": len(rows),
        "per_source": per_source,
        "exact_duplicate_sha256_groups": len(duplicate_hashes),
        "exact_duplicate_sha256_images": sum(len(items) for items in duplicate_hashes),
        "cross_split_source_groups": len(cross_split_groups),
        "cross_split_source_group_examples": cross_split_groups[:40],
        "kitchen_normal_cooking_with_fire_0": sum(
            json.loads(str(row["label_counts"])).get("0", 0) > 0
            for row in kitchen_rows if row["split_or_category"] == "normal_cooking"
        ),
        "kitchen_big_flame_empty_labels": sum(
            not json.loads(str(row["label_counts"]))
            for row in kitchen_rows if row["split_or_category"] == "big_flame"
        ),
        "limitations": [
            "Image hashes detect byte-identical files only; transformed near duplicates need separate review.",
            "Folder names and old label IDs are not verified visual ground truth.",
            "Kitchen images were previously copied into combined/train and cannot test the old model independently.",
        ],
    }
    output.mkdir(parents=True)
    with (output / "local_sources.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    audit(parser.parse_args().out)
