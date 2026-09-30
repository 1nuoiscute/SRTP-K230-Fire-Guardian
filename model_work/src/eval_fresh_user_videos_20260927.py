"""One-pass inference for new, training-unseen kitchen videos.

This freezes a checkpoint hash and inference settings before any new-video
inspection. It reports predictions; human ground truth is recorded separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from data_integrity import reject_exposed

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
KNOWN = ROOT.parent / "viedos"
IMGSZ = 640
CONF = 0.25
IOU = 0.6
SAMPLE_FPS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    videos_dir, weights, out = args.videos_dir.resolve(), args.weights.resolve(), args.out.resolve()
    if out.exists():
        raise SystemExit(f"Refusing overwrite: {out}")
    files = sorted(videos_dir.glob("*.mp4"))
    if not files or not weights.is_file():
        raise SystemExit("No MP4 files or missing checkpoint")
    weight_hash = sha256(weights)
    if weight_hash.lower() != args.expected_weights_sha256.lower():
        raise SystemExit(f"Checkpoint hash mismatch: {weight_hash}")
    known = {sha256(p) for p in KNOWN.glob("*.mp4")}
    source = []
    for path in files:
        digest = reject_exposed(path, ROOT.parent / 'docs' / 'data_exposure_registry.json')
        if digest in known:
            raise SystemExit(f"Training-video duplicate: {path}")
        source.append({"path": str(path), "sha256": digest, "bytes": path.stat().st_size})
    out.mkdir(parents=True)
    model = YOLO(str(weights))
    frame_rows = []
    summaries = []
    for path, identity in zip(files, source):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SystemExit(f"Unreadable video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise SystemExit(f"Bad video metadata: {path}")
        step = max(1, round(fps / SAMPLE_FPS))
        rendered = out / f"{path.stem}_boxed.mp4"
        writer = cv2.VideoWriter(str(rendered), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / step, (width, height))
        if not writer.isOpened():
            raise SystemExit(f"Cannot write {rendered}")
        index = 0
        clip_rows = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                prediction = model.predict(frame, imgsz=IMGSZ, conf=CONF,
                                           iou=IOU, device=0, verbose=False, save=False)[0]
                boxes = []
                for box in prediction.boxes:
                    if int(box.cls.item()) != 0:
                        raise SystemExit("Unexpected class")
                    score = float(box.conf.item())
                    xyxy = [round(float(x), 1) for x in box.xyxy[0].tolist()]
                    boxes.append({"confidence": round(score, 4), "xyxy": xyxy})
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(frame, f"fire {score:.2f}", (x1, max(25, y1-7)),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 2)
                cv2.rectangle(frame, (0, height-43), (width, height), (0, 0, 0), -1)
                cv2.putText(frame, f"{path.name}  t={index/fps:.1f}s  boxes={len(boxes)}",
                            (12, height-13), cv2.FONT_HERSHEY_SIMPLEX,
                            .65, (255, 255, 255), 2)
                writer.write(frame)
                row = {"video": path.name, "frame_index": index,
                       "time_seconds": round(index/fps, 3), "boxes": len(boxes),
                       "max_conf": max((b["confidence"] for b in boxes), default=0.0),
                       "predictions": json.dumps(boxes)}
                clip_rows.append(row)
                frame_rows.append(row)
            index += 1
        cap.release()
        writer.release()
        summaries.append({**identity, "fps": fps, "width": width, "height": height,
                          "source_frames": total, "sampled_frames": len(clip_rows),
                          "sampled_frames_with_box": sum(r["boxes"] > 0 for r in clip_rows),
                          "rendered": str(rendered)})
        print(f"{path.name}: {len(clip_rows)} sampled, {summaries[-1]['sampled_frames_with_box']} with box", flush=True)
    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    (out / "summary.json").write_text(json.dumps({
        "weights": str(weights), "weights_sha256": weight_hash,
        "settings": {"imgsz": IMGSZ, "conf": CONF, "nms_iou": IOU,
                     "target_sample_fps": SAMPLE_FPS},
        "videos": summaries,
        "warning": "Predictions only. Manually review flame truth, box location, and independent scene identity before performance claims."
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
