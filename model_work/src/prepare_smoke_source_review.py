"""Read-only smoke label inventory and temporal-stratified train review sheets.

Never maps source Smoke to approved smoke truth or converts unlabeled images to
negatives. Source segment CSV is a historical temporal index, not kitchen identity.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw
from data_integrity import sha256

NAMES = ["Burner", "Flame", "Smoke", "Spill", "safe"]
COLORS = ["#55ffff", "#ff5555", "#ff55ff", "#ffff55", "#55ff55"]


def fit(im, size):
    im = im.copy(); im.thumbnail(size)
    canvas = Image.new("RGB", size, "#151515")
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--segments", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-segment", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261002)
    a = p.parse_args()
    if a.out.exists(): raise SystemExit("Refusing overwrite")
    if not 1 <= a.per_segment <= 10 or a.seed < 0: p.error("Invalid sample bounds")
    source_config = (a.root / "data.yaml").read_text(encoding="utf-8")
    # The export class order is a contract; don't silently guess a new source.
    if "['Burner', 'Flame', 'Smoke', 'Spill', 'safe']" not in source_config:
        raise SystemExit("Unexpected source class order")
    index = {}
    for row in csv.DictReader(a.segments.open(encoding="utf-8-sig")):
        key = row["split"], row["file"]
        if key in index: raise SystemExit("Duplicate temporal index record")
        index[key] = row
    counts = Counter(); groups = defaultdict(list); inventory = []; seen = set()
    for split in ("train", "valid", "test"):
        for image in sorted((a.root / split / "images").glob("*.jpg")):
            key = split, image.name
            if key not in index: raise SystemExit("Image missing from temporal index")
            seen.add(key)
            label = a.root / split / "labels" / (image.stem + ".txt")
            boxes = []
            for line in label.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip(): continue
                vals = list(map(float, line.split()))
                if len(vals) != 5 or not all(math.isfinite(v) for v in vals):
                    raise SystemExit("Invalid source label")
                cls, x, y, w, h = vals
                if cls != int(cls) or not 0 <= cls < 5 or not 0 <= x <= 1 or not 0 <= y <= 1 or not 0 < w <= 1 or not 0 < h <= 1:
                    raise SystemExit("Invalid source class/coordinates")
                boxes.append([int(cls), x, y, w, h])
                counts[(split, NAMES[int(cls)] + "_boxes")] += 1
            classes = sorted({b[0] for b in boxes})
            indexed_classes = [int(c) for c in index[key]["classes"].split("|") if c]
            if classes != indexed_classes or len(boxes) != int(index[key]["n_boxes"]):
                raise SystemExit("Historical temporal index no longer matches source labels")
            counts[(split, "images")] += 1
            if 2 not in classes: continue
            counts[(split, "Smoke_images")] += 1
            counts[(split, "Smoke_with_Flame")] += 1 in classes
            row = {"split": split, "image": str(image.resolve()), "label": str(label.resolve()),
                   "sha256": sha256(image), "label_sha256": sha256(label), "classes": classes,
                   "boxes": boxes, "segment": index[key]["segment"]}
            inventory.append(row)
            if split == "train": groups[row["segment"]].append(row)
    if seen != set(index): raise SystemExit("Temporal index has missing/extra source images")
    selected = []
    for group, rows in sorted(groups.items()):
        def rank(row):
            return hashlib.sha256(f"{a.seed}\0{group}\0{row['sha256']}".encode()).hexdigest(), row["image"]
        selected.extend(sorted(rows, key=rank)[:a.per_segment])
    a.out.mkdir(parents=True); sheets = []
    for page in range((len(selected) + 11) // 12):
        sheet = Image.new("RGB", (2048, 1152), "#303030")
        for j, row in enumerate(selected[page * 12:(page + 1) * 12]):
            im = Image.open(row["image"]).convert("RGB"); w, h = im.size
            annotated = im.copy(); draw = ImageDraw.Draw(annotated); smoke = []
            for cls, x, y, bw, bh in row["boxes"]:
                box = [max(0, int((x-bw/2)*w)), max(0, int((y-bh/2)*h)),
                       min(w, int((x+bw/2)*w)), min(h, int((y+bh/2)*h))]
                draw.rectangle(box, outline=COLORS[cls], width=max(2, w//250))
                draw.text((box[0], box[1]), NAMES[cls], fill=COLORS[cls])
                if cls == 2: smoke.append(box)
            union = [min(b[0] for b in smoke), min(b[1] for b in smoke),
                     max(b[2] for b in smoke), max(b[3] for b in smoke)]
            tile = Image.new("RGB", (512, 384), "#202020"); td = ImageDraw.Draw(tile)
            idx = page*12 + j + 1
            td.text((8, 8), f"{idx:02d} train {row['segment']} source Smoke boxes={len(smoke)}", fill="white")
            tile.paste(fit(annotated, (252, 310)), (2, 40))
            tile.paste(fit(im.crop(union), (252, 310)), (258, 40))
            td.text((8, 360), "Source labels / Smoke union; semantic truth UNREVIEWED", fill="white")
            sheet.paste(tile, ((j%4)*512, (j//4)*384))
            row.update(index=idx, size=[w, h], manual_review=None)
        output = a.out / f"sheet_{page+1:02d}.jpg"
        sheet.save(output, quality=95); sheets.append({"file": output.name, "sha256": sha256(output)})
    summary = {"role": "source label inventory and train semantic review; not approved smoke training truth",
               "counts": {s: {k: v for (sp, k), v in sorted(counts.items()) if sp == s} for s in ("train", "valid", "test")},
               "source_data_yaml_sha256": sha256(a.root / "data.yaml"),
               "temporal_index_sha256": sha256(a.segments), "script_sha256": sha256(Path(__file__)),
               "selection": {"seed": a.seed, "per_segment": a.per_segment, "train_temporal_groups": len(groups),
                             "samples": len(selected), "method": "smallest SHA(seed, temporal segment, image SHA) per train segment",
                             "scope": "Temporal index checked against file/class/box inventory; not scene independence"},
               "sheets": sheets, "reviews": selected, "inventory": inventory}
    (a.out / "review_pending.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("reviews", "inventory")}, indent=2))


if __name__ == "__main__": main()
