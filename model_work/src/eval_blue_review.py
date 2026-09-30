"""Count detections on five source-checked blue-flame diagnostic photos.

All five have visible flame. This is a selected failure probe, not a test set.
No bounding-box truth is asserted here; predicted fire anywhere is only detection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "commons_blue_review_20260927"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing overwrite: {args.out}")
    rows = list(csv.DictReader((DATA / "manifest.csv").open(encoding="utf-8-sig")))
    if len(rows) != 5:
        raise SystemExit(f"Expected five images, found {len(rows)}")
    model = YOLO(str(args.weights))
    output = []
    for row in rows:
        image = DATA / (row["slug"] + ".jpg")
        if hashlib.sha256(image.read_bytes()).hexdigest() != row["sha256"]:
            raise SystemExit(f"Image changed: {image}")
        prediction = model.predict(
            str(image), imgsz=640, conf=0.05, iou=0.6,
            device=args.device, verbose=False, save=False,
        )[0]
        boxes = [{"confidence": round(float(box.conf.item()), 4),
                  "xyxy": [round(float(x), 1) for x in box.xyxy[0].tolist()]}
                 for box in prediction.boxes]
        output.append({"slug": row["slug"], "source_page": row["source_page"],
                       "det_025": int(any(b["confidence"] >= 0.25 for b in boxes)),
                       "det_050": int(any(b["confidence"] >= 0.5 for b in boxes)),
                       "max_conf": max((b["confidence"] for b in boxes), default=0.0),
                       "predictions": json.dumps(boxes)})
    args.out.mkdir(parents=True)
    with (args.out / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    summary = {"weights": str(args.weights),
               "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
               "images": len(output),
               "detected_025": sum(r["det_025"] for r in output),
               "detected_050": sum(r["det_050"] for r in output),
               "warning": "Selected blue-flame diagnostic photos, no box truth; not an accuracy estimate."}
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
