"""Extract exact frozen sampled source frames for manual box-location review."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "model_work" / "out" / "teammate_videos_20260928"
PRED = BASE / "frozen_v3_5fps"
SOURCE = ROOT / "viedos" / "视频" / "视频"
OUT = BASE / "manual_frames"
OUT.mkdir(parents=True, exist_ok=True)
TIMES = {
    "219bd5": [1.4, 13.0],
    "35cf4b": [6.8, 17.4],
    "82f51e": [7.9, 23.8],
    "88400a": [2.6, 38.7],
    "f0cb3e": [17.8, 53.5],
}
rows = defaultdict(list)
with (PRED / "per_frame.csv").open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        rows[row["video"]].append(row)
manifest = []
for name, video_rows in rows.items():
    targets = next(v for k, v in TIMES.items() if name.startswith(k))
    cap = cv2.VideoCapture(str(SOURCE / name))
    for requested in targets:
        row = min(video_rows, key=lambda r: abs(float(r["time_seconds"]) - requested))
        frame_index = int(row["frame_index"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read {name} frame {frame_index}")
        stem = f"{name[:7]}_{frame_index}"
        cv2.imwrite(str(OUT / f"{stem}.jpg"), frame)
        manifest.append({"source": name, "frame_index": frame_index,
                         "time_seconds": float(row["time_seconds"]),
                         "image": f"{stem}.jpg",
                         "predictions": json.loads(row["predictions"])})
    cap.release()
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
