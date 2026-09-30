"""Create reproducible sampling sheets from the four newly provided videos."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT.parent / "viedos"
OUT = ROOT / "out" / "user_videos_20260927"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing overwrite: {OUT}")
    files = sorted(VIDEOS.glob("*.mp4"))
    if [p.name for p in files] != [f"{i}.mp4" for i in range(4)]:
        raise SystemExit(f"Unexpected video inventory: {files}")
    OUT.mkdir(parents=True)
    (OUT / "sampled_frames").mkdir()
    rows = []
    for path in files:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frames / fps
        seconds = [2.5, 7.5, 12.5, 17.5, 22.5, 27.5]
        sheet = np.zeros((455, 6 * 245, 3), dtype=np.uint8)
        for j, second in enumerate(seconds):
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"Cannot read {path} at {second}s")
            still = OUT / "sampled_frames" / f"{path.stem}_{second:04.1f}s.jpg"
            cv2.imwrite(str(still), frame)
            thumb = cv2.resize(frame, (240, 426), interpolation=cv2.INTER_AREA)
            x = j * 245
            sheet[0:426, x:x + 240] = thumb
            cv2.putText(sheet, f"{path.name} {second:.1f}s", (x + 2, 447),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(OUT / f"{path.stem}_contact_sheet.jpg"), sheet)
        cap.release()
        rows.append({"file": str(path), "sha256": sha256(path),
                     "bytes": path.stat().st_size, "width": width, "height": height,
                     "fps": fps, "frames": frames, "duration_seconds": round(duration, 3),
                     "sampled_seconds": ";".join(map(str, seconds))})
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} video sheets to {OUT}")


if __name__ == "__main__":
    main()
