"""Score same-kitchen development clips by location, not mere box presence.

The fixed reference boxes were manually placed on six reviewed frames per
clip and propagated to 1 fps frames in static shots. This is diagnostic only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "out" / "user_videos_20260927"
GT = {
    "0.mp4": [],
    "1.mp4": [[184, 277, 514, 518]],
    "2.mp4": [[329, 344, 447, 393]],
    "3.mp4": [[184, 312, 544, 397]],
}


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (area_a+area_b-inter) if area_a+area_b-inter else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    folder = BASE / args.run_name
    rows = list(csv.DictReader((folder / "per_frame.csv").open(encoding="utf-8-sig")))
    per_video = {}
    scored = []
    for name, gt in GT.items():
        subset = [r for r in rows if r["video"] == name]
        if not subset:
            raise SystemExit(f"Missing rows for {name}")
        best_ious = []
        for row in subset:
            preds = json.loads(row["predictions_025"])
            best = max((iou(box["xyxy"], g) for box in preds for g in gt), default=0.0)
            best_ious.append(best)
            scored.append({"video": name, "time": row["time_seconds"],
                           "boxes": len(preds), "best_iou": round(best, 4),
                           "matched_03": int(best >= 0.3), "matched_05": int(best >= 0.5)})
        per_video[name] = {"frames": len(subset),
                           "frames_with_box": sum(int(r["boxes_025"]) > 0 for r in subset),
                           "matched_iou_03": sum(v >= .3 for v in best_ious),
                           "matched_iou_05": sum(v >= .5 for v in best_ious),
                           "mean_best_iou": round(sum(best_ious)/len(best_ious), 4)}
    (folder / "location_score.json").write_text(
        json.dumps({"run_name": args.run_name, "ground_truth_xyxy": GT,
                    "caveat": "Development data from one apparent kitchen. Static reference boxes on unreviewed 1 fps frames are approximate; no independent accuracy estimate.",
                    "videos": per_video}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (folder / "location_per_frame.csv").open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    print(json.dumps(per_video, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
