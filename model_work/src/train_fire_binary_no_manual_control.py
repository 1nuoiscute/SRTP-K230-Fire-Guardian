"""Matched control for the 2026-09-27 manual25 fine-tune.

Start from the same v1 checkpoint and use the same hyperparameters as
train_fire_binary_manual25_v2.py, but train on the original 2382 images.
This isolates the effect of adding 25 corrected images in that experiment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"D:\SRTP_Datasets\kitchen_fire_binary_codex_20260927")
WEIGHT = ROOT / "runs" / "fire-binary-codex-v1" / "weights" / "best.pt"
RUN = ROOT / "runs" / "fire-binary-v2-no-manual-control"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if RUN.exists():
        raise SystemExit(f"Refusing to overwrite: {RUN}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    for path in (DATA / "data.yaml", DATA / "manifest.csv", WEIGHT):
        if not path.is_file():
            raise SystemExit(f"Missing input: {path}")
    configuration = dict(
        data=str(DATA / "data.yaml"), epochs=40, patience=10,
        imgsz=640, batch=4, device=0, workers=0, seed=20260927,
        optimizer="AdamW", lr0=0.0003, lrf=0.01, weight_decay=0.0005,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, degrees=3.0,
        translate=0.05, scale=0.2, fliplr=0.5,
        mosaic=0.4, mixup=0.0, close_mosaic=10,
        project=str(ROOT / "runs"), name=RUN.name, exist_ok=False,
        save=True, plots=True, val=True,
    )
    provenance = {
        "purpose": "matched control for 25 corrected training images",
        "start_weight": str(WEIGHT),
        "start_sha256": sha256(WEIGHT),
        "data_yaml": str(DATA / "data.yaml"),
        "data_yaml_sha256": sha256(DATA / "data.yaml"),
        "manifest_sha256": sha256(DATA / "manifest.csv"),
        "train_images": 2382,
        "comparison_run": str(ROOT / "runs" / "fire-binary-codex-v2-manual25"),
        "configuration": configuration,
    }
    print(json.dumps(provenance, ensure_ascii=False, indent=2), flush=True)
    model = YOLO(str(WEIGHT))
    model.train(**configuration)
    (RUN / "control_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best = RUN / "weights" / "best.pt"
    print(json.dumps({"best": str(best), "best_sha256": sha256(best)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
