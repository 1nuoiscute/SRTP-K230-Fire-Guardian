"""Read-only inventory and byte/decoded overlap of acquired D-Fire vs known data.

pHash is recorded for later visual-family review, not scene-independence proof.
No source files or labels are changed; no automatic split or training approval.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import yaml
from data_integrity import sha256


def inspect(item):
    split, image, label = item
    row = {"split": split, "file": image.name, "image": str(image), "label": str(label),
           "sha256": sha256(image), "errors": [], "boxes": []}
    if not label.is_file(): row["errors"].append("missing_label")
    else:
        row["label_sha256"] = sha256(label)
        for line in label.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip(): continue
            try:
                values = list(map(float, line.split()))
                if len(values) != 5 or not all(math.isfinite(v) for v in values): raise ValueError()
                c, x, y, w, h = values
                if c != int(c) or c not in (0, 1) or not 0 <= x <= 1 or not 0 <= y <= 1 or not 0 < w <= 1 or not 0 < h <= 1: raise ValueError()
                row["boxes"].append([int(c), x, y, w, h])
                if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6:
                    row["errors"].append("box_exceeds_bounds")
            except ValueError: row["errors"].append("malformed_label")
    im = cv2.imread(str(image))
    if im is None: row["errors"].append("decode_failed")
    else:
        h, w = im.shape[:2]; row.update(width=w, height=h)
        # Shape is included: equal byte buffers with different shapes are not equal pictures.
        row["decoded_sha256"] = hashlib.sha256(f"{w}x{h}\0".encode() + im.tobytes()).hexdigest()
        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        coeff = cv2.dct(cv2.resize(gray, (32, 32)).astype(np.float32))[:8, :8].flatten()
        bits = coeff > np.median(coeff[1:]); bits[0] = False
        row["phash64"] = f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"
    row["classes"] = sorted({b[0] for b in row["boxes"]})
    row["errors"] = sorted(set(row["errors"]))
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acquisition", type=Path, required=True)
    p.add_argument("--known-data", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    if a.out.exists(): raise SystemExit("Refusing overwrite")
    if not 1 <= a.workers <= 12: p.error("Invalid worker count")
    acquisition = json.loads((a.acquisition / "acquisition.json").read_text(encoding="utf-8"))
    if acquisition.get("status") != "complete": raise SystemExit("Acquisition not complete")
    if sha256(a.acquisition / "dataset.zip") != acquisition["archive_sha256"]:
        raise SystemExit("Acquired archive identity differs")
    identity = json.loads((a.acquisition / "class_contract_identity.json").read_text(encoding="utf-8"))
    config_path = a.acquisition / "class_contract_dfire.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if sha256(config_path) != identity["sha256"] or config.get("names") != ["smoke", "fire"] or config.get("nc") != 2:
        raise SystemExit("Class contract differs")
    cv2.setNumThreads(1)
    inputs = []; orphan = {}
    root = a.acquisition / "raw/data"
    for split in ("train", "val", "test"):
        images = sorted(p for p in (root / split / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        labels = {p.stem: p for p in (root / split / "labels").glob("*.txt")}
        if len({p.stem for p in images}) != len(images): raise SystemExit("Duplicate source stems")
        orphan[split] = len(set(labels) - {p.stem for p in images})
        inputs.extend((split, image, root / split / "labels" / (image.stem + ".txt")) for image in images)
    a.out.mkdir(parents=True); records = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, record in enumerate(pool.map(inspect, inputs), 1):
            records.append(record)
            if i % 1000 == 0: print(f"Inventoried {i}/{len(inputs)} D-Fire images", flush=True)
    known = {}; manifests = []
    for folder in a.known_data:
        manifest = folder / "manifest.csv"; manifests.append({"manifest": str(manifest), "sha256": sha256(manifest)})
        with manifest.open(encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                name = row.get("file", row.get("out_file"))
                image = folder / "images" / row["split"] / name
                if row["sha256"] not in known:
                    if not image.is_file() or sha256(image) != row["sha256"]: raise SystemExit("Known input identity differs")
                    known[row["sha256"]] = {"split": row["split"], "image": str(image)}
    counts = {}; byte_groups = defaultdict(list); pixel_groups = defaultdict(list); known_overlaps = []
    for r in records:
        byte_groups[r["sha256"]].append(r)
        if "decoded_sha256" in r: pixel_groups[r["decoded_sha256"]].append(r)
        if r["sha256"] in known: known_overlaps.append({"candidate": r, "known": known[r["sha256"]]})
    for split in ("train", "val", "test"):
        rr = [r for r in records if r["split"] == split]; c = Counter()
        for r in rr:
            c["images"] += 1; c["empty_labels"] += not r["classes"]
            c["smoke_only"] += r["classes"] == [0]; c["fire_only"] += r["classes"] == [1]; c["both"] += r["classes"] == [0, 1]
            c["smoke_boxes"] += sum(b[0] == 0 for b in r["boxes"])
            c["fire_boxes"] += sum(b[0] == 1 for b in r["boxes"])
            c["images_with_errors"] += bool(r["errors"])
            for e in r["errors"]: c[e] += 1
        c["orphan_labels"] = orphan[split]; counts[split] = dict(c)
    def duplicates(groups):
        return [{"sha256": key, "members": [{"split": r["split"], "file": r["file"], "label_sha256": r.get("label_sha256")} for r in rows],
                 "cross_split": len({r["split"] for r in rows}) > 1,
                 "different_labels": len({r.get("label_sha256") for r in rows}) > 1}
                for key, rows in groups.items() if len(rows) > 1]
    byte_dups = duplicates(byte_groups); pixel_dups = duplicates(pixel_groups)
    result = {"role": "source development inventory; not scene-independent or label-quality acceptance",
              "archive_sha256": acquisition["archive_sha256"], "acquisition_json_sha256": sha256(a.acquisition / "acquisition.json"),
              "class_contract": identity, "script_sha256": sha256(Path(__file__)), "counts": counts,
              "known_manifests": manifests, "known_unique_byte_identities_checked": len(known),
              "byte_duplicate_groups": len(byte_dups), "byte_cross_split_groups": sum(d["cross_split"] for d in byte_dups),
              "decoded_duplicate_groups": len(pixel_dups), "decoded_cross_split_groups": sum(d["cross_split"] for d in pixel_dups),
              "known_byte_overlap_images": len(known_overlaps),
              "limitations": ["pHash values recorded, near-pair search and visual-family review not completed here.",
                              "Known manifest coverage only; generic pretraining/unrecorded sources not excluded.",
                              "Filename/publisher split does not prove kitchen or capture-event independence."]}
    for name, value in (("inventory.json", records), ("byte_duplicates.json", byte_dups), ("decoded_duplicates.json", pixel_dups), ("known_byte_overlaps.json", known_overlaps)):
        path = a.out / name; path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        result[name + "_sha256"] = sha256(path)
    (a.out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "known_manifests"}, indent=2), flush=True)


if __name__ == "__main__": main()
