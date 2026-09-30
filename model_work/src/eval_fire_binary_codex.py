"""Evaluate the frozen kitchen test split once after fire-only training.

Reports both image-level detection and IoU>=0.5 localization at fixed
confidence thresholds. Outputs one row per held-out image.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work")
DATA = Path(r"D:\SRTP_Datasets\kitchen_fire_binary_codex_20260927")
WEIGHTS = ROOT / "runs" / "fire-binary-codex-v1" / "weights" / "best.pt"
OUT = ROOT / "out" / "fire_binary_codex_eval_20260927"
TARGET_PROVENANCE = {"ks_flame", "kitchen_stove_fire", "ks_nofire_confirmed"}


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def ground_truth(label: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = map(float, line.split())
        if int(cls) != 0:
            raise ValueError(f"Unexpected class in {label}")
        boxes.append([(x - w / 2) * width, (y - h / 2) * height,
                      (x + w / 2) * width, (y + h / 2) * height])
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing overwrite: {args.out}")
    rows = [r for r in csv.DictReader((DATA / "manifest.csv").open(encoding="utf-8-sig"))
            if r["split"] == "test" and r["provenance"] in TARGET_PROVENANCE]
    train_hashes = {r["sha256"] for r in
                    csv.DictReader((DATA / "manifest.csv").open(encoding="utf-8-sig"))
                    if r["split"] == "train"}
    if any(r["sha256"] in train_hashes for r in rows):
        raise SystemExit("Test/train exact overlap")
    args.out.mkdir(parents=True)
    model = YOLO(str(args.weights))
    output = []
    for index, row in enumerate(rows, 1):
        image = DATA / "images" / "test" / row["file"]
        label = DATA / "labels" / "test" / (image.stem + ".txt")
        with Image.open(image) as opened:
            width, height = opened.size
        gt = ground_truth(label, width, height)
        prediction = model.predict(str(image), imgsz=640, conf=0.25,
                                   iou=0.6, device=0, verbose=False, save=False)[0]
        detections = [{"confidence": round(float(box.conf.item()), 4),
                       "xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()]}
                      for box in prediction.boxes]
        item = {"file": row["file"], "provenance": row["provenance"],
                "source": row["source"], "group_id": row["group_id"],
                "gt_boxes": len(gt), "gt_xyxy": json.dumps(gt),
                "max_conf": max((d["confidence"] for d in detections), default=0.0),
                "predictions": json.dumps(detections)}
        for threshold in (0.25, 0.5):
            selected = [d for d in detections if d["confidence"] >= threshold]
            best = max((iou(g, d["xyxy"]) for g in gt for d in selected), default=0.0)
            suffix = "025" if threshold == 0.25 else "050"
            item[f"pred_{suffix}"] = len(selected)
            item[f"best_iou_{suffix}"] = round(best, 4)
            item[f"box_match_{suffix}"] = int(best >= 0.5)
        output.append(item)
        if index % 25 == 0:
            print(f"{index}/{len(rows)}", flush=True)
    with (args.out / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    summary = {}
    for threshold, suffix in ((0.25, "025"), (0.5, "050")):
        counts = defaultdict(Counter)
        for item in output:
            counter = counts[item["provenance"]]
            counter["images"] += 1
            if item["gt_boxes"]:
                counter["fire_images"] += 1
                counter["detected"] += int(item[f"pred_{suffix}"] > 0)
                counter["localized"] += item[f"box_match_{suffix}"]
            else:
                counter["no_fire_images"] += 1
                counter["false_positive"] += int(item[f"pred_{suffix}"] > 0)
        summary[str(threshold)] = {key: dict(value) for key, value in counts.items()}
    meta = {"model": str(args.weights),
            "model_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            "data": str(DATA), "test_images": len(output), "summary": summary,
            "note": "Image-level detection is not box localization; IoU>=0.5 is reported separately."}
    (args.out / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
