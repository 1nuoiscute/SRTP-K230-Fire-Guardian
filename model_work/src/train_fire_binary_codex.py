"""Train a fire-only YOLO11s candidate from the audited v3 weights.

This run is exploratory. The kitchen test split is not used for fitting or
checkpoint choice. Deployment requires separate validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work")
DATA = Path(r"D:\SRTP_Datasets\kitchen_fire_binary_codex_20260927")
START = Path(r"C:\Users\ASUS\Documents\Codex\kitchen-fire-detection"
             r"\runs\kitchen-vision-v3-640\weights\best.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    name = "fire-binary-codex-smoke" if args.smoke else "fire-binary-codex-v1"
    run = ROOT / "runs" / name
    if run.exists():
        raise SystemExit(f"Refusing to overwrite existing run: {run}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available")
    rows = list(csv.DictReader((DATA / "manifest.csv").open(encoding="utf-8-sig")))
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    if len(train) != 2382 or len(val) != 243:
        raise SystemExit("Dataset count changed; audit before training")
    hashes = {r["sha256"] for r in train}
    if any(r["sha256"] in hashes for r in rows if r["split"] != "train"):
        raise SystemExit("Image hash overlap across splits")
    for split in ("train", "val", "test"):
        for label in (DATA / "labels" / split).glob("*.txt"):
            if any(line.strip() and line.split()[0] != "0"
                   for line in label.read_text(encoding="utf-8").splitlines()):
                raise SystemExit(f"Non-fire class in {label}")
    counts = Counter(r["provenance"] for r in train)
    source_hash = hashlib.sha256(START.read_bytes()).hexdigest()
    config = dict(
        data=str(DATA / "data.yaml"), epochs=60, patience=15,
        imgsz=640, batch=4, device=0, workers=0, seed=20260927,
        optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, degrees=3.0,
        translate=0.05, scale=0.2, fliplr=0.5,
        mosaic=0.4, mixup=0.0, close_mosaic=10,
        project=str(ROOT / "runs"), name=name, exist_ok=False,
        save=True, plots=True, val=True,
    )
    if args.smoke:
        config.update(epochs=1, fraction=0.1)
    print(json.dumps({"name": name, "start_sha256": source_hash,
                      "train": dict(counts), "config": config}, indent=2), flush=True)
    model = YOLO(str(START))
    model.train(**config)
    (run / "codex_run_meta.json").write_text(
        json.dumps({"start_weights": str(START), "start_sha256": source_hash,
                    "dataset": str(DATA), "train_sources": dict(counts),
                    "config": config}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
