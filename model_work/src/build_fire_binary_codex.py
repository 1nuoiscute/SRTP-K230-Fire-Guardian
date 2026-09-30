"""Build a non-destructive, source-auditable fire-only dataset from v3.

One deterministic image per source group is retained in the training split.
The existing validation/test sessions are copied without changing their split.
Original images and labels are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

SOURCE = Path(r"D:\SRTP_Datasets\kitchen_vision_v3_20260926")
TARGET = Path(r"D:\SRTP_Datasets\kitchen_fire_binary_codex_20260927")
TRAIN_PROVENANCE = {
    "kitchen_stove_fire": None,
    "ks_flame": None,
    "generic_fire": 800,
    "nofire_real_indoor": None,
    "object_pan": 200,
}
EVAL_PROVENANCE = {
    "kitchen_stove_fire", "ks_flame", "ks_nofire_confirmed",
    "nofire_real_indoor", "object_pan",
}


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    if split != "train":
        return [r for r in rows if r["split"] == split and r["provenance"] in EVAL_PROVENANCE]
    picked = []
    for provenance, limit in TRAIN_PROVENANCE.items():
        source = [r for r in rows if r["split"] == split and r["provenance"] == provenance]
        by_group: dict[str, list[dict[str, str]]] = {}
        for row in source:
            by_group.setdefault(row["group_id"], []).append(row)
        keys = sorted(by_group, key=lambda k: stable_key(provenance + ":" + k))
        if limit is not None:
            keys = keys[:limit]
        for key in keys:
            picked.append(min(by_group[key], key=lambda r: stable_key(r["out_file"])))
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=TARGET)
    args = parser.parse_args()
    source, target = args.source, args.target
    if target.exists():
        raise SystemExit(f"Refusing to overwrite {target}")
    rows = list(csv.DictReader((source / "manifest.csv").open(encoding="utf-8", newline="")))
    chosen = {split: select(rows, split) for split in ("train", "val", "test")}
    train_groups = {(r["source"], r["group_id"]) for r in chosen["train"]}
    train_hashes = {r["sha256"] for r in chosen["train"]}
    for split in ("val", "test"):
        for row in chosen[split]:
            if (row["source"], row["group_id"]) in train_groups or row["sha256"] in train_hashes:
                raise SystemExit(f"Train overlap: {row['out_file']}")
    target.mkdir(parents=True)
    exported = []
    for split, subset in chosen.items():
        (target / "images" / split).mkdir(parents=True)
        (target / "labels" / split).mkdir(parents=True)
        for row in subset:
            name = row["out_file"]
            source_image = source / "images" / split / name
            target_image = target / "images" / split / name
            os.link(source_image, target_image)
            source_label = source / "labels" / split / (Path(name).stem + ".txt")
            fire = [line for line in source_label.read_text(encoding="utf-8").splitlines()
                    if line.strip() and line.split()[0] == "0"]
            (target / "labels" / split / source_label.name).write_text(
                "\n".join(fire) + ("\n" if fire else ""), encoding="utf-8")
            exported.append({"split": split, "file": name, "source": row["source"],
                             "provenance": row["provenance"], "group_id": row["group_id"],
                             "sha256": row["sha256"], "fire_boxes": len(fire),
                             "license": row["license"]})
    (target / "data.yaml").write_text(
        f"path: {target.as_posix()}\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnc: 1\nnames:\n  0: fire\n", encoding="utf-8")
    with (target / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    summary = {split: {"images": len(subset),
                       "by_provenance": dict(Counter(r["provenance"] for r in subset))}
               for split, subset in chosen.items()}
    (target / "build_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
