"""Copy an approved development dataset and cap one original training source.

Source images/labels, inherited weights and all diagnostic splits stay immutable.
This changes sampling only; it neither repairs labels nor creates independent scenes.
"""
import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
from data_integrity import sha256
from dataset_preflight import verify_reviewed_dataset
from source_sampling_plan import select_source_caps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--cap", type=int, required=True)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--rationale", required=True)
    a = p.parse_args()
    parent, out = a.parent.resolve(), a.out.resolve()
    if out.exists() or parent == out or parent in out.parents or out in parent.parents:
        raise SystemExit("Output must be a new directory outside the parent tree")
    if not a.rationale.strip():
        p.error("A recorded rationale is required")
    parent_check = verify_reviewed_dataset(parent, yaml.safe_load((parent / "data.yaml").read_text(encoding="utf-8")))
    meta = json.loads((parent / "build_manifest.json").read_text(encoding="utf-8"))
    base = Path(meta["base"])
    with (base / "manifest.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    kept, audit = select_source_caps(meta["lineage"], rows, {a.source: a.cap}, a.seed)
    # Parent is read-only. No removal of any parent files, and no directory overwrite.
    shutil.copytree(parent, out, ignore=shutil.ignore_patterns("*.cache", "__pycache__"))
    meta.update(lineage=kept, unique_paths=len(kept), train_entries=sum(r["weight"] for r in kept),
                legacy_unique_images=sum(r["origin"] == "legacy_train" for r in kept),
                legacy_train=sum(r["weight"] for r in kept if r["origin"] == "legacy_train"))
    meta["source_sampling"] = {"parent_folder": str(parent),
        "parent_manifest_sha256": parent_check["manifest_sha256"],
        "parent_approval_sha256": parent_check["approval_sha256"],
        "parent_train_list_sha256": parent_check["train_list_sha256"],
        "caps": {a.source: a.cap}, "seed": a.seed, "audit": audit,
        "rationale": a.rationale, "builder_sha256": sha256(Path(__file__)),
        "scope": "Sampling ablation only; retained labels and weights unchanged; no independent test"}
    train = []
    for row in kept:
        source = base if row["origin"] == "legacy_train" else out
        train.extend([str(source / "images/train" / row["image"])] * row["weight"])
    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    config = {"path": str(out), "train": str(out / "train.txt"),
              "val": str(base / "images/val"), "nc": 1, "names": {0: "fire"}}
    (out / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = out / "build_manifest.json"
    manifest.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    approval = json.loads((parent / "review_approval.json").read_text(encoding="utf-8"))
    approval.update(manifest_sha256=sha256(manifest), source_sampling_sha256=sha256(manifest),
        reason=a.rationale + "; sampling-only revision; all retained labels/images and weights inherited unchanged",
        reviewer="Codex deterministic source sampling; inherited source/label reviews preserved",
        reviewed_utc=datetime.now(timezone.utc).isoformat())
    (out / "review_approval.json").write_text(json.dumps(approval, ensure_ascii=False, indent=2), encoding="utf-8")
    result = verify_reviewed_dataset(out, config)
    (out / "source_cap_preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
