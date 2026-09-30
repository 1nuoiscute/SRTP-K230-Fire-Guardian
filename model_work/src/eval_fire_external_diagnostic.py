"""Image-level diagnostic evaluation on the 13 photographed scenes.

These images are for failure analysis, not an independent performance claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO
from data_integrity import sha256
from source_exposure import commons_source_key,training_source_keys,diagnostic_summary,source_exposure_summary


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "manual_kitchen_labels_v2_20260927" / "commons_candidates"


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def ground_truth(label: Path, width: int, height: int) -> list[list[float]]:
    result = []
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = map(float, line.split())
        if int(cls) != 0:
            raise ValueError(f"Unexpected class in {label}")
        result.append([(x - w / 2) * width, (y - h / 2) * height,
                       (x + w / 2) * width, (y + h / 2) * height])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comparison-data", type=Path, required=True,
                        help="Reviewed training manifest for comparison-relative source exposure")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing overwrite: {args.out}")
    with (DATA / "manifest.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    comparison_meta=json.loads(args.comparison_data.read_text(encoding="utf-8"))
    source_keys=training_source_keys(comparison_meta)
    if len(rows) != 13:
        raise SystemExit(f"Expected 13 diagnostic images, got {len(rows)}")
    args.out.mkdir(parents=True)
    model = YOLO(str(args.weights))
    output = []
    for row in rows:
        image = DATA / f"{row['slug']}.jpg"
        label = DATA / "labels" / f"{row['slug']}.txt"
        if sha256(image)!=row["sha256"]: raise SystemExit("Diagnostic image identity mismatch")
        source_key=commons_source_key(row["source_page"])
        with Image.open(image) as opened:
            width, height = opened.size
        gt = ground_truth(label, width, height)
        pred = model.predict(str(image), imgsz=640, conf=0.25,
                             iou=0.6, device=0, verbose=False, save=False)[0]
        boxes = [{"confidence": round(float(b.conf.item()), 4),
                  "xyxy": [round(float(v), 2) for v in b.xyxy[0].tolist()]}
                 for b in pred.boxes]
        item = {"slug": row["slug"], "scene": row["scene"],
                "source_page":row["source_page"], "source_key":source_key,
                "source_exposure": "training_source_in_comparison" if source_key in source_keys else "untrained_source_development_diagnostic",
                "intended": row["intended"], "gt_boxes": len(gt),
                "max_conf": max((b["confidence"] for b in boxes), default=0.0),
                "predictions": json.dumps(boxes)}
        for conf, suffix in ((0.25, "025"), (0.5, "050")):
            selected = [b for b in boxes if b["confidence"] >= conf]
            best = max((iou(g, b["xyxy"]) for g in gt for b in selected), default=0.0)
            item[f"det_{suffix}"] = int(bool(selected))
            item[f"best_iou_{suffix}"] = round(best, 4)
            item[f"localized_{suffix}"] = int(best >= 0.5)
        output.append(item)
    with (args.out / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    summaries = diagnostic_summary(output)
    result = {"weights": str(args.weights),
              "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "data": str(DATA), "images": len(output), "summary": summaries,
              "comparison_manifest_sha256":sha256(args.comparison_data),
              "source_exposure":source_exposure_summary(output),
              "summary_scope":"All thirteen development diagnostics, including sources used in the comparison training manifest.",
              "warning": "Selected small diagnostic set; use source-exposure strata. Untrained sources are still development diagnostics, not independent acceptance."}
    (args.out / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
