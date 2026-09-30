"""Evaluate any frozen fire-only checkpoint on the unchanged 286-image test split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"D:\SRTP_Datasets\kitchen_fire_binary_codex_20260927\data.yaml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.weights = args.weights.resolve()
    args.out = args.out.resolve()
    if not args.weights.is_file() or not DATA.is_file():
        raise SystemExit("Missing weights or frozen data.yaml")
    if args.out.exists():
        raise SystemExit(f"Refusing overwrite: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metrics = YOLO(str(args.weights)).val(
        data=str(DATA), split="test", imgsz=640, batch=4, device=0,
        workers=0, plots=False, project=str(args.out.parent),
        name=args.out.name, exist_ok=False,
    )
    result = {
        "weights": str(args.weights),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "data": str(DATA),
        "split": "test",
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    (args.out / "metrics_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
