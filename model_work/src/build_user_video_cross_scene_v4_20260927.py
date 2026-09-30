"""Extend the video adaptation with source-attributed cross-scene examples.

The previously diagnostic Commons 4 blue-flame and 3 negative scenes become
training data. The separate five-photo blue probe is kept out of training.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "data" / "user_video_adaptation_20260927"
SOURCE = ROOT / "data" / "manual_kitchen_labels_v2_20260927" / "commons_candidates"
TARGET = ROOT / "data" / "user_video_cross_scene_v4_20260927"
KEEP_OUT = ROOT / "data" / "commons_blue_review_20260927"
SELECT = ("gas_stove_burner_flame", "gas_stove_blue_flames",
          "gas_cooker_blue_flame", "gas_stove_flame",
          "outdoor_wok", "gas_cooker", "pot_on_stove")
REPEAT = 10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if TARGET.exists():
        raise SystemExit(f"Refusing overwrite: {TARGET}")
    rows = {r["slug"]: r for r in csv.DictReader((SOURCE / "manifest.csv").open(encoding="utf-8-sig"))}
    held = {r["sha256"] for r in csv.DictReader((KEEP_OUT / "manifest.csv").open(encoding="utf-8-sig"))}
    prior = [p.strip() for p in (PREVIOUS / "train.txt").read_text(encoding="utf-8").splitlines() if p.strip()]
    assert len(prior) == 770
    (TARGET / "images" / "train").mkdir(parents=True)
    (TARGET / "labels" / "train").mkdir(parents=True)
    added = []
    sheet = np.full((4 * 340, 2 * 510, 3), 30, dtype=np.uint8)
    for index, slug in enumerate(SELECT):
        row = rows[slug]
        src = SOURCE / f"{slug}.jpg"
        label = SOURCE / "labels" / f"{slug}.txt"
        digest = sha256(src)
        if digest != row["sha256"] or digest in held:
            raise SystemExit(f"Source changed or held-out overlap: {slug}")
        lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != int(row["fire_boxes"]):
            raise SystemExit(f"Label count mismatch: {slug}")
        frame = cv2.imread(str(src))
        if frame is None:
            raise SystemExit(f"Unreadable image: {src}")
        height, width = frame.shape[:2]
        boxes = []
        for line in lines:
            cls, x, y, w, h = map(float, line.split())
            if int(cls) != 0 or not all(0 <= v <= 1 for v in (x, y, w, h)):
                raise SystemExit(f"Bad label: {label}")
            box = [int((x-w/2)*width), int((y-h/2)*height),
                   int((x+w/2)*width), int((y+h/2)*height)]
            boxes.append(box)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), max(2, width//700))
        thumb = cv2.resize(frame, (480, 300), interpolation=cv2.INTER_AREA)
        col, row_no = index % 2, index // 2
        xx, yy = col * 510, row_no * 340
        sheet[yy:yy+300, xx:xx+480] = thumb
        cv2.putText(sheet, slug, (xx+5, yy+326), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (255, 255, 255), 1)
        for rep in range(REPEAT):
            stem = f"commons_{slug}_r{rep}"
            dest = TARGET / "images" / "train" / (stem + ".jpg")
            os.link(src, dest)
            shutil.copy2(label, TARGET / "labels" / "train" / (stem + ".txt"))
            prior.append(str(dest).replace("\\", "/"))
        added.append({"slug": slug, "source_page": row["source_page"],
                      "author": row["author"], "license": row["license"],
                      "sha256": digest, "boxes_xyxy": boxes,
                      "original_label": str(label), "copies": REPEAT})
    cv2.imwrite(str(TARGET / "label_review_sheet.jpg"), sheet)
    (TARGET / "train.txt").write_text("\n".join(prior) + "\n", encoding="utf-8")
    old_yaml = (PREVIOUS / "data.yaml").read_text(encoding="utf-8")
    old_yaml = old_yaml.replace(PREVIOUS.as_posix(), TARGET.as_posix())
    (TARGET / "data.yaml").write_text(old_yaml, encoding="utf-8")
    meta = {"previous_train_entries": 770, "new_images": len(added),
            "new_train_entries": len(added)*REPEAT, "total_train_entries": len(prior),
            "held_out_blue_photo_count": len(held), "added": added,
            "warning": "These seven Commons scenes were development diagnostics and are now train-exposed; do not re-use as evaluation."}
    (TARGET / "build_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "added"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
