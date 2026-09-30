"""Summarize frozen predictions without treating adjacent frames as independent events."""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "model_work" / "out" / "teammate_videos_20260928" / "frozen_v3_5fps"
grouped = defaultdict(list)
with (BASE / "per_frame.csv").open(encoding="utf-8-sig", newline="") as stream:
    for row in csv.DictReader(stream):
        grouped[row["video"]].append(row)

def streak(rows, has_box):
    best = (0, 0, 0)
    start = None
    for i, row in enumerate(rows + [None]):
        matching = row is not None and (int(row["boxes"]) > 0) == has_box
        if matching and start is None:
            start = i
        if not matching and start is not None:
            length = i - start
            if length > best[0]:
                best = (length, float(rows[start]["time_seconds"]),
                        float(rows[i-1]["time_seconds"]))
            start = None
    return best

for name, rows in grouped.items():
    detections = []
    for row in rows:
        detections.extend(json.loads(row["predictions"]))
    scores = sorted(d["confidence"] for d in detections)
    med = scores[len(scores)//2] if scores else 0
    example = [(round(float(r["time_seconds"]), 1), json.loads(r["predictions"]))
               for r in rows if int(r["boxes"]) > 0]
    print(json.dumps({
        "video": name,
        "sampled": len(rows),
        "frames_with_any_box": sum(int(r["boxes"]) > 0 for r in rows),
        "total_boxes": len(detections),
        "median_box_score": med,
        "max_box_score": scores[-1] if scores else 0,
        "longest_no_box": streak(rows, False),
        "longest_with_box": streak(rows, True),
        "first_box": example[0] if example else None,
        "last_box": example[-1] if example else None,
    }, ensure_ascii=False))
