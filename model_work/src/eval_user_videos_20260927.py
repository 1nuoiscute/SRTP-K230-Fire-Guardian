"""Evaluate four fixed-camera clips at 5 fps and render model boxes.

This is a development diagnostic. Frame counts are not independent samples.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT.parent / "viedos"
DEFAULT_WEIGHTS = ROOT / "runs" / "fire-binary-v2-no-manual-control" / "weights" / "best.pt"
OUT_BASE = ROOT / "out" / "user_videos_20260927"
CONF_DISPLAY = 0.25
CONF_PROBE = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--run-name", default="model_eval_5fps")
    parser.add_argument("--sample-fps", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    weights = args.weights.resolve()
    out = OUT_BASE / args.run_name
    if out.exists():
        raise SystemExit(f"Refusing overwrite: {out}")
    files = sorted(VIDEOS.glob("*.mp4"))
    if [p.name for p in files] != [f"{i}.mp4" for i in range(4)]:
        raise SystemExit("Expected 0.mp4 through 3.mp4")
    if not weights.is_file():
        raise SystemExit(f"Missing weights: {weights}")
    if not 1 <= args.sample_fps <= 30:
        raise SystemExit("sample-fps must be in [1, 30]")
    out.mkdir(parents=True)
    model = YOLO(str(weights))
    all_rows = []
    for path in files:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = round(fps / args.sample_fps)
        if fps <= 0 or step < 1:
            raise SystemExit(f"Invalid fps: {path}")
        output_video = out / f"{path.stem}_boxed_{args.sample_fps}fps.mp4"
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / step, (width, height))
        if not writer.isOpened():
            raise SystemExit(f"Cannot write: {output_video}")
        frame_index = 0
        sampled = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                result = model.predict(frame, imgsz=args.imgsz, conf=CONF_PROBE, iou=0.6,
                                       device=0, verbose=False, save=False)[0]
                boxes = []
                for box in result.boxes:
                    if int(box.cls.item()) != 0:
                        raise SystemExit(f"Unexpected class in {path}")
                    confidence = float(box.conf.item())
                    xyxy = [float(x) for x in box.xyxy[0].tolist()]
                    boxes.append({"confidence": round(confidence, 4),
                                  "xyxy": [round(x, 1) for x in xyxy]})
                    if confidence >= CONF_DISPLAY:
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        cv2.putText(frame, f"fire {confidence:.2f}",
                                    (x1, max(y1 - 7, 25)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 0, 0), 2)
                active = [box for box in boxes if box["confidence"] >= CONF_DISPLAY]
                time_seconds = frame_index / fps
                cv2.rectangle(frame, (0, height - 43), (width, height), (0, 0, 0), -1)
                cv2.putText(frame, f"{path.name}  t={time_seconds:.1f}s  model boxes={len(active)}",
                            (12, height - 13), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 2)
                writer.write(frame)
                all_rows.append({"video": path.name, "frame_index": frame_index,
                                 "time_seconds": round(time_seconds, 3),
                                 "boxes_025": len(active),
                                 "boxes_050": sum(b["confidence"] >= 0.5 for b in boxes),
                                 "max_conf_005": round(max((b["confidence"] for b in boxes),
                                                           default=0.0), 4),
                                 "predictions_025": json.dumps(active)})
                sampled += 1
            frame_index += 1
        cap.release()
        writer.release()
        print(f"{path.name}: {frame_index} frames, {sampled} sampled, output {output_video}",
              flush=True)
    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {}
    for path in files:
        rows = [row for row in all_rows if row["video"] == path.name]
        summary[path.name] = {"input_sha256": sha256(path),
                              "sampled_frames": len(rows),
                              "frames_with_boxes_025": sum(row["boxes_025"] > 0 for row in rows),
                              "frames_with_boxes_050": sum(row["boxes_050"] > 0 for row in rows),
                              "max_confidence": max(row["max_conf_005"] for row in rows)}
    provenance = {"weights": str(weights), "weights_sha256": sha256(weights),
                  "imgsz": args.imgsz, "probe_confidence": CONF_PROBE,
                  "display_confidence": CONF_DISPLAY, "nms_iou": 0.6,
                  "sampling": f"Every {step}th frame from {fps:g} fps source, {fps / step:g} fps output",
                  "videos": summary,
                  "warning": "Diagnostic clips from one apparent kitchen; frame fractions are not accuracy estimates."}
    (out / "summary.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
