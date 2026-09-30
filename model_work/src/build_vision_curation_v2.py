"""Build a source-preserving curation index from the read-only local audit.

This does not copy or relabel images. It refuses to overwrite an output folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def stable_fraction(value: str) -> float:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) / 16**16


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(source_csv: Path, output: Path) -> None:
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    with source_csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))

    # Source filename stems identify original images before Roboflow variants.
    # Byte-identical copies also share a split, even when the names differ.
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    by_source_group: dict[tuple[str, str], int] = {}
    by_hash: dict[str, int] = {}
    for index, row in enumerate(rows):
        key = (row["source"], row["source_group"])
        if key in by_source_group:
            union(index, by_source_group[key])
        by_source_group[key] = index
        if row["sha256"] in by_hash:
            union(index, by_hash[row["sha256"]])
        by_hash[row["sha256"]] = index

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)

    # The old fire-flame labels only mean generic visible fire. They do not
    # establish normal stove flame or abnormal kitchen fire scene semantics.
    # Deduplicate byte-identical images in the manifest without altering source.
    retained: list[dict] = []
    seen_hash: set[str] = set()
    for index, row in enumerate(rows):
        if row["source"] != "fire_flame":
            continue
        if row["sha256"] in seen_hash:
            continue
        seen_hash.add(row["sha256"])
        member_keys = sorted(
            {f'{rows[i]["source"]}:{rows[i]["source_group"]}' for i in components[find(index)]}
        )
        group_id = hashlib.sha256("\n".join(member_keys).encode("utf-8")).hexdigest()[:16]
        fraction = stable_fraction(group_id)
        split = "train" if fraction < 0.8 else "val" if fraction < 0.9 else "test"
        retained.append({
            "split": split,
            "group_id": group_id,
            "source": row["source"],
            "source_split": row["split_or_category"],
            "image": row["image"],
            "label": row["label"],
            "sha256": row["sha256"],
            "label_semantics": "generic_visible_fire_only",
            "curation_status": "candidate_needs_domain_review",
        })

    kitchen_queue: list[dict] = []
    for row in rows:
        if row["source"] != "kitchen":
            continue
        counts = json.loads(row["label_counts"])
        flags = []
        if row["split_or_category"] == "big_flame" and not counts:
            flags.append("empty_old_label")
        if row["split_or_category"] == "normal_cooking" and counts.get("0", 0):
            flags.append("normal_scene_old_fire_label")
        if row["label_errors"]:
            flags.append("invalid_old_label")
        if "istock" in row["image"].lower() or "vcg" in row["image"].lower():
            flags.append("stock_license_to_verify")
        kitchen_queue.append({
            "image": row["image"],
            "old_category": row["split_or_category"],
            "old_label_counts": row["label_counts"],
            "old_label_errors": row["label_errors"],
            "review_flags": ";".join(flags),
            "visual_scene": "unreviewed",
            "flame_box_status": "unreviewed",
            "rights_status": "unreviewed",
            "training_status": "hold",
        })

    output.mkdir(parents=True)
    write_csv(output / "generic_fire_candidate_manifest.csv", retained, list(retained[0]))
    write_csv(output / "kitchen_visual_review_queue.csv", kitchen_queue, list(kitchen_queue[0]))
    split_hashes = {split: {r["sha256"] for r in retained if r["split"] == split} for split in ("train", "val", "test")}
    split_groups = {split: {r["group_id"] for r in retained if r["split"] == split} for split in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not split_hashes[a] & split_hashes[b]
        assert not split_groups[a] & split_groups[b]
    report = {
        "generic_fire_candidate_unique_images": len(retained),
        "generic_fire_split_counts": dict(Counter(r["split"] for r in retained)),
        "kitchen_review_queue_images": len(kitchen_queue),
        "kitchen_training_approved_images": 0,
        "source_images_unchanged": True,
        "note": "The generic-fire manifest is a candidate index, not a released training set. Kitchen old labels are held for visual and rights review.",
    }
    (output / "curation_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.source_csv, arguments.out)
