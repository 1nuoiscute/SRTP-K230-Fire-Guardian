"""Fine-tune v1 on reviewed video frames plus a balanced legacy subset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "user_video_adaptation_20260927"
START = ROOT / "runs" / "fire-binary-codex-v1" / "weights" / "best.pt"
RUN = ROOT / "runs" / "fire-binary-user-video-v3"


def main() -> None:
    if RUN.exists():
        raise SystemExit(f"Refusing overwrite: {RUN}")
    if not DATA.is_dir() or not START.is_file():
        raise SystemExit("Dataset or start checkpoint missing")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU unavailable")
    expected = "5f0fa05ebb92eb73cb39971159751e499df3f91d075c70ca818877c64ba05148"
    actual = hashlib.sha256(START.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"Start checkpoint changed: {actual}")
    model = YOLO(str(START))
    model.train(data=str(DATA / "data.yaml"), epochs=12, patience=6,
                imgsz=640, batch=4, device=0, workers=0, seed=20260927,
                optimizer="AdamW", lr0=0.0001, lrf=0.1, weight_decay=0.0005,
                hsv_h=0.015, hsv_s=0.3, hsv_v=0.25, degrees=3.0,
                translate=0.05, scale=0.2, fliplr=0.5,
                mosaic=0.2, mixup=0.0, close_mosaic=5,
                project=str(ROOT / "runs"), name=RUN.name, exist_ok=False,
                save=True, plots=True, val=True)
    (RUN / "adaptation_meta.json").write_text(json.dumps({
        "start": str(START), "start_sha256": actual, "data": str(DATA / "data.yaml"),
        "limitation": "Four same-kitchen videos were used for training; revisit them only as development regression, not blind validation."
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
