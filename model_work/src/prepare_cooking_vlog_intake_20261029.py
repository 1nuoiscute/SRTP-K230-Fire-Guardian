"""Prepare sparse source-only review for a quarantined public cooking video."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "model_work/data/cooking_vlog_quarantine_20261029/original.webm"
PLAN = SOURCE.with_name("quarantine_plan.json")
OUT = ROOT / "model_work/out/cooking_vlog_intake_20261029"
EXPECTED = "5c7150580f8665eb7e20c6bd813f072b8a7d9f87a7c73ab09d6dbabd19073204"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if OUT.exists():
        raise ValueError("Refusing to overwrite output")
    if sha(SOURCE) != EXPECTED:
        raise ValueError("Original changed")
    plan_hash = sha(PLAN)
    cap = cv2.VideoCapture(str(SOURCE))
    if not cap.isOpened():
        raise ValueError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 0 < fps <= 240:
        raise ValueError("Invalid fps metadata")
    step = round(fps * 10)
    frames, records, decoded = [], [], 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if decoded % step == 0:
            records.append({"frame_index": decoded, "approx_seconds": decoded / fps,
                            "pixel_sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
                            "width": image.shape[1], "height": image.shape[0]})
            frames.append(cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA))
        decoded += 1
    cap.release()
    if decoded == 0 or sha(SOURCE) != EXPECTED or sha(PLAN) != plan_hash:
        raise ValueError("Source or plan changed during intake")
    OUT.mkdir(parents=True)
    sheets = []
    for start in range(0, len(frames), 12):
        canvas = np.full((3 * 296, 4 * 480, 3), 255, dtype=np.uint8)
        for offset, image in enumerate(frames[start:start + 12]):
            row, col = divmod(offset, 4)
            y, x = row * 296, col * 480
            canvas[y:y+270, x:x+480] = image
            record = records[start + offset]
            cv2.putText(canvas, f"frame {record['frame_index']} / ~{record['approx_seconds']:.1f}s",
                        (x + 5, y + 290), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        sheet = OUT / f"overview_{start // 12 + 1:02d}.png"
        if not cv2.imwrite(str(sheet), canvas):
            raise ValueError("Failed to write source overview")
        sheets.append({"file": sheet.name, "sha256": sha(sheet)})
    result = {"role": "quarantined source-only overview; no predictions, no approved labels",
              "source_sha256": EXPECTED, "quarantine_plan_sha256": plan_hash,
              "script_sha256": sha(Path(__file__)), "fps_metadata": fps,
              "decoded_frames": decoded, "sampling_step_frames": step,
              "samples": records, "sheets": sheets,
              "limitations": ["Sparse thumbnails cannot establish whole-event truth or visibility between samples.",
                              "Index/fps timestamps are approximate and may differ from source presentation timestamps.",
                              "Editing and scene changes prevent treating the video as one continuous camera event."]}
    (OUT / "intake.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(f"Decoded {decoded} frames at metadata {fps} fps; {len(records)} source-only overview frames.")


if __name__ == "__main__":
    main()
