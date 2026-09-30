"""Continue v3 with seven verified cross-scene images and the prior training set."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "user_video_cross_scene_v4_20260927"
START = ROOT / "runs" / "fire-binary-user-video-v3" / "weights" / "best.pt"
RUN = ROOT / "runs" / "fire-binary-user-video-cross-scene-v4"


def main() -> None:
    if RUN.exists():
        raise SystemExit(f"Refusing overwrite: {RUN}")
    if not DATA.is_dir() or not START.is_file() or not torch.cuda.is_available():
        raise SystemExit("Missing dataset, start checkpoint, or CUDA GPU")
    actual = hashlib.sha256(START.read_bytes()).hexdigest()
    expected = "48f284f9094fe334e02c66647f95fed8bfb3396b09f4488753508f514ed38980"
    if actual != expected:
        raise SystemExit(f"Start checkpoint changed: {actual}")
    YOLO(str(START)).train(
        data=str(DATA / "data.yaml"), epochs=10, patience=5,
        imgsz=640, batch=4, device=0, workers=0, seed=20260927,
        optimizer="AdamW", lr0=0.00005, lrf=0.1, weight_decay=0.0005,
        hsv_h=0.015, hsv_s=0.35, hsv_v=0.3, degrees=3.0,
        translate=0.05, scale=0.25, fliplr=0.5,
        mosaic=0.2, mixup=0.0, close_mosaic=4,
        project=str(ROOT / "runs"), name=RUN.name, exist_ok=False,
        save=True, plots=True, val=True)
    (RUN / "adaptation_meta.json").write_text(json.dumps({
        "start": str(START), "start_sha256": actual, "data": str(DATA / "data.yaml"),
        "limitation": "All four videos and seven Commons scenes are train-exposed; only fresh recordings can provide blind acceptance evidence."
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
