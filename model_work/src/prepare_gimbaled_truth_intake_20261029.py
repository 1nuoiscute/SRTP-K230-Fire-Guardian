"""Prepare source-only review sheets for an unlabelled, held video; no predictions."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "model_work/data/public_video_probe_20260930/gimbaled.webm"
OUT = ROOT / "model_work/out/gimbaled_truth_intake_20261029"
EXPECTED = "bdb375cb67d9ddbc3b4521c7fb48c00e2db14a75c5fae5e1cbef7a1455f96cd8"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise ValueError("Refusing to overwrite output")
    if sha(SOURCE) != EXPECTED:
        raise ValueError("Source video identity differs")
    cap = cv2.VideoCapture(str(SOURCE))
    if not cap.isOpened():
        raise ValueError("Cannot open source")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 0 < fps <= 240:
        raise ValueError("Invalid fps")
    step = round(fps)
    frames, decoded, records = [], 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if decoded % step == 0:
            records.append({"frame_index": decoded, "approx_seconds": decoded / fps,
                            "pixel_sha256": hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest(),
                            "width": frame.shape[1], "height": frame.shape[0]})
            frames.append(frame)
        decoded += 1
    cap.release()
    if decoded != 838 or len(frames) != 28 or sha(SOURCE) != EXPECTED:
        raise ValueError("Unexpected source decoding or source changed")
    OUT.mkdir(parents=True)
    sheets = []
    for start in range(0, len(frames), 4):
        batch = frames[start:start + 4]
        h, w = batch[0].shape[:2]
        canvas = np.full(((h + 35) * 2, w * 2, 3), 255, dtype=np.uint8)
        for offset, frame in enumerate(batch):
            y, x = divmod(offset, 2)
            top, left = y * (h + 35), x * w
            canvas[top:top+h, left:left+w] = frame
            record = records[start + offset]
            cv2.putText(canvas, f"frame {record['frame_index']} / ~{record['approx_seconds']:.1f}s",
                        (left + 8, top + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        path = OUT / f"source_sheet_{start // 4 + 1:02d}.png"
        if not cv2.imwrite(str(path), canvas):
            raise ValueError("Failed to write review sheet")
        sheets.append({"file": path.name, "sha256": sha(path)})
    manifest = {"role": "Unlabelled source-only intake; no model predictions and no independent-test claim",
                "source_sha256": EXPECTED, "script_sha256": sha(Path(__file__)), "fps_metadata": fps,
                "decoded_frames": decoded, "sampling_step_frames": step, "samples": records,
                "sheets": sheets, "annotations_approved": False, "inference_run": False,
                "limitations": ["Frame index divided by metadata fps is approximate; no presentation timestamps checked.",
                                "One-second samples do not establish event-level truth or visible flame between samples."]}
    (OUT / "intake.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(f"Prepared {len(records)} source frames and {len(sheets)} sheets; no predictions.")


if __name__ == "__main__":
    main()
