"""Read-only audit of the downloaded gc_kitchen_annotation Roboflow v2 export."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


CLASS_NAMES = ("flame-reflection", "flame-under-pot", "gas-stove-flame", "no-flame")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(root: Path) -> None:
    result: dict = {"root": str(root), "images": {}, "boxes_by_class": {}, "errors": {}}
    box_counts: Counter[int] = Counter()
    errors: Counter[str] = Counter()
    hash_locations: dict[str, list[str]] = defaultdict(list)
    original_groups: dict[str, set[str]] = defaultdict(set)
    class_groups: dict[int, set[str]] = defaultdict(set)
    class_groups_by_split: dict[tuple[str, int], set[str]] = defaultdict(set)
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    mapped_fire_boxes_by_split: Counter[str] = Counter()
    mapped_empty_images_by_split: Counter[str] = Counter()
    mapped_fire_groups_by_split: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "valid", "test"):
        images = sorted((root / split / "images").glob("*.jpg"))
        labels_dir = root / split / "labels"
        labels = {p.stem: p for p in labels_dir.glob("*.txt")}
        result["images"][split] = len(images)
        image_stems = {p.stem for p in images}
        errors[f"{split}_missing_labels"] = len(image_stems - labels.keys())
        errors[f"{split}_orphan_labels"] = len(labels.keys() - image_stems)
        for image in images:
            hash_locations[digest(image)].append(f"{split}:{image.name}")
            group = image.stem.split(".rf.", 1)[0]
            original_groups[group].add(split)
            groups_by_split[split].add(group)
            try:
                with Image.open(image) as opened:
                    opened.verify()
            except Exception:
                errors["image_decode"] += 1
            label = labels.get(image.stem)
            if label is None:
                continue
            contents = label.read_text(encoding="utf-8-sig", errors="replace").strip()
            mapped_fire_on_image = 0
            if not contents:
                errors[f"{split}_empty_labels"] += 1
            for line in contents.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    errors["field_count"] += 1
                    continue
                try:
                    class_id = int(parts[0])
                    values = [float(item) for item in parts[1:]]
                except ValueError:
                    errors["non_numeric"] += 1
                    continue
                if not (0 <= class_id < len(CLASS_NAMES)):
                    errors["unknown_class"] += 1
                elif any(not 0 <= value <= 1 for value in values) or values[2] <= 0 or values[3] <= 0:
                    errors["invalid_coordinates"] += 1
                else:
                    box_counts[class_id] += 1
                    class_groups[class_id].add(group)
                    class_groups_by_split[(split, class_id)].add(group)
                    if class_id in (1, 2):
                        mapped_fire_on_image += 1
            mapped_fire_boxes_by_split[split] += mapped_fire_on_image
            if mapped_fire_on_image:
                mapped_fire_groups_by_split[split].add(group)
            else:
                mapped_empty_images_by_split[split] += 1
    result["boxes_by_class"] = {CLASS_NAMES[index]: box_counts[index] for index in range(len(CLASS_NAMES))}
    result["source_image_groups_by_split"] = {split: len(groups_by_split[split]) for split in ("train", "valid", "test")}
    result["source_image_groups_by_class"] = {CLASS_NAMES[index]: len(class_groups[index]) for index in range(len(CLASS_NAMES))}
    result["source_image_groups_by_split_and_class"] = {
        split: {CLASS_NAMES[index]: len(class_groups_by_split[(split, index)]) for index in range(len(CLASS_NAMES))}
        for split in ("train", "valid", "test")
    }
    result["proposed_mapping_1_2_to_fire_0_0_3_ignored"] = {
        split: {
            "fire_boxes": mapped_fire_boxes_by_split[split],
            "images_with_empty_labels": mapped_empty_images_by_split[split],
            "source_image_groups_with_fire": len(mapped_fire_groups_by_split[split]),
        }
        for split in ("train", "valid", "test")
    }
    result["errors"] = {name: value for name, value in errors.items() if value}
    duplicates = [items for items in hash_locations.values() if len(items) > 1]
    result["exact_duplicate_groups"] = len(duplicates)
    result["cross_split_exact_duplicate_groups"] = sum(len({item.split(":", 1)[0] for item in items}) > 1 for items in duplicates)
    result["cross_split_original_name_groups"] = sum(len(splits) > 1 for splits in original_groups.values())
    result["visual_labels_validated"] = False
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
