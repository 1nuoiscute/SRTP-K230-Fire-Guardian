#!/usr/bin/env python3
"""Compile the vendor-compatible ONNX model with nncase 2.9.0 for K230."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import re

import nncase
import numpy as np
from PIL import Image


def calibration_samples(directory: Path, limit: int, size: int) -> list[list[np.ndarray]]:
    image_paths = sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        raise SystemExit(f"No calibration images found below {directory}")

    # Augmented copies end in _d0, _d1, ... . Use one image per source.
    unique: dict[str, Path] = {}
    for path in image_paths:
        source = re.sub(r"_d\d+$", "", path.stem)
        unique.setdefault(source, path)

    labels_dir = directory.parent.parent / "labels" / directory.name
    if not labels_dir.is_dir():
        raise SystemExit(f"Cannot find matching YOLO labels: {labels_dir}")
    buckets: dict[int | None, list[Path]] = {0: [], 1: [], 2: [], 3: [], None: []}
    for path in unique.values():
        label = labels_dir / f"{path.stem}.txt"
        classes = {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()} if label.exists() else set()
        for category in classes or {None}:
            if category in buckets:
                buckets[category].append(path)

    available = [category for category, paths in buckets.items() if paths]
    rng = random.Random(20260924)
    for paths in buckets.values():
        rng.shuffle(paths)
    selected: list[Path] = []
    seen: set[Path] = set()
    while len(selected) < limit:
        added = False
        for category in available:
            while buckets[category] and buckets[category][0] in seen:
                buckets[category].pop(0)
            if buckets[category] and len(selected) < limit:
                path = buckets[category].pop(0)
                selected.append(path)
                seen.add(path)
                added = True
        if not added:
            break
    if len(selected) < limit:
        raise SystemExit(f"Only {len(selected)} distinct calibration images available; requested {limit}")
    print(f"calibration_sources={len(selected)}, available_categories={available}")

    result: list[list[np.ndarray]] = []
    for path in selected:
        image = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)[None, ...]
        result.append([array])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("calibration_dir", type=Path)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--size", type=int, default=576)
    args = parser.parse_args()

    samples = calibration_samples(args.calibration_dir, args.samples, args.size)
    print(f"calibration_samples={len(samples)}")

    options = nncase.CompileOptions()
    options.target = "k230"
    options.preprocess = True
    options.swapRB = False
    options.input_shape = [1, 3, args.size, args.size]
    options.input_type = "uint8"
    options.input_range = [0, 1]
    options.mean = [0, 0, 0]
    options.std = [1, 1, 1]
    options.input_layout = "NCHW"
    options.quant_type = "uint8"

    compiler = nncase.Compiler(options)
    compiler.import_onnx(args.model.read_bytes(), nncase.ImportOptions())
    ptq = nncase.PTQTensorOptions()
    ptq.samples_count = len(samples)
    ptq.set_tensor_data(samples)
    compiler.use_ptq(ptq)
    compiler.compile()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(compiler.gencode_tobytes())
    print(f"saved {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
