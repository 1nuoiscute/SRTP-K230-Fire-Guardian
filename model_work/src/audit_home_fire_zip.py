"""Inspect a Home-fire ZIP without extracting it or changing source data."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path


def main(path: Path) -> None:
    boxes: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    empty = 0
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        images = {Path(name).stem for name in names if name.startswith("images/") and name.lower().endswith((".jpg", ".jpeg", ".png"))}
        labels = {Path(name).stem: name for name in names if name.startswith("labels/") and name.lower().endswith(".txt")}
        for name in labels.values():
            content = archive.read(name).decode("utf-8-sig", errors="replace")
            if not content.strip():
                empty += 1
            for line in content.splitlines():
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    issues["field_count"] += 1
                    continue
                try:
                    class_id = int(parts[0])
                    coordinates = [float(value) for value in parts[1:]]
                except ValueError:
                    issues["non_numeric"] += 1
                    continue
                if class_id < 0 or any(value < 0 or value > 1 for value in coordinates):
                    issues["out_of_range"] += 1
                boxes[str(class_id)] += 1
        report = {
            "archive": str(path),
            "zip_integrity_error": archive.testzip(),
            "images": len(images),
            "labels": len(labels),
            "images_missing_labels": len(images - labels.keys()),
            "labels_missing_images": len(labels.keys() - images),
            "empty_labels": empty,
            "boxes_by_raw_class_id": dict(boxes),
            "label_issues": dict(issues),
            "visual_quality_checked": False,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    main(parser.parse_args().archive)
