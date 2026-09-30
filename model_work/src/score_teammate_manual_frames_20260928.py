"""Score ten manually reviewed flame locations; this is a spot check, not video accuracy."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "model_work" / "out" / "teammate_videos_20260928" / "manual_frames"
manifest = {r["image"]: r for r in json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))}
reviews = json.loads((BASE / "flame_gt_review.json").read_text(encoding="utf-8"))

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)

for review in reviews:
    source = manifest[review["image"]]
    overlaps = [round(iou(review["flame_xyxy"], p["xyxy"]), 3) for p in source["predictions"]]
    result = {
        "image": review["image"],
        "time_seconds": source["time_seconds"],
        "manual_flame_xyxy": review["flame_xyxy"],
        "prediction_ious": overlaps,
        "has_correct_iou50": any(v >= 0.5 for v in overlaps),
        "has_wrong_box": any(v < 0.5 for v in overlaps),
    }
    print(json.dumps(result, ensure_ascii=False))
