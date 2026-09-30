"""Fine-tune fire-only v1 with 25 corrected old kitchen labels, train split only."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work")
BASE = Path(r"D:\SRTP_Datasets\kitchen_fire_binary_codex_20260927")
MANUAL = ROOT / "data" / "manual_kitchen_labels_v2_20260927"
DATA = ROOT / "data" / "fire_binary_manual25_v2_20260927"
WEIGHT = ROOT / "runs" / "fire-binary-codex-v1" / "weights" / "best.pt"
RUN = ROOT / "runs" / "fire-binary-codex-v2-manual25"


def prepare() -> dict:
    if DATA.exists() or RUN.exists():
        raise SystemExit(f"Refusing existing dataset/run: {DATA} / {RUN}")
    if not BASE.exists() or not WEIGHT.exists():
        raise SystemExit("Base dataset or start weight missing")
    base_rows = list(csv.DictReader((BASE / "manifest.csv").open(encoding="utf-8-sig")))
    manual_rows = list(csv.DictReader((MANUAL / "manifest.csv").open(encoding="utf-8-sig")))
    selected = [r for r in manual_rows if int(r["new_fire_boxes"]) > 0]
    assert len(base_rows) == 2911 and len(selected) == 25
    base_hashes = {r["sha256"] for r in base_rows}
    assert all(r["image_sha256"] not in base_hashes for r in selected)
    DATA.mkdir(parents=True)
    image_dir = DATA / "images" / "train"
    label_dir = DATA / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    added = []
    for index, row in enumerate(selected):
        source = Path(row["source_image"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["image_sha256"]
        source_label = MANUAL / "labels" / Path(row["relative_image"]).with_suffix(".txt")
        lines = source_label.read_text(encoding="utf-8").splitlines()
        assert len(lines) == int(row["new_fire_boxes"])
        assert all(line.split()[0] == "0" for line in lines)
        stem = f"manual25_{index:02d}_{source.stem}"
        image = image_dir / (stem + source.suffix.lower())
        label = label_dir / (stem + ".txt")
        shutil.copy2(source, image)
        shutil.copy2(source_label, label)
        added.append({"source": str(source), "source_sha256": row["image_sha256"],
                      "image": str(image), "label": str(label), "boxes": len(lines)})
    train_images = sorted(p for p in (BASE / "images" / "train").glob("*")
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    assert len(train_images) == 2382
    images = [str(p) for p in train_images] + [r["image"] for r in added]
    (DATA / "train.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
    (DATA / "data.yaml").write_text(
        f"path: {DATA.as_posix()}\ntrain: {str(DATA / 'train.txt').replace(chr(92), '/')}\n"
        f"val: {str(BASE / 'images' / 'val').replace(chr(92), '/')}\n"
        f"test: {str(BASE / 'images' / 'test').replace(chr(92), '/')}\n"
        "nc: 1\nnames:\n  0: fire\n", encoding="utf-8")
    metadata = {"base_dataset": str(BASE), "base_train_count": 2382,
                "added_count": len(added), "added_fire_boxes": sum(r["boxes"] for r in added),
                "added": added, "val_count": 243, "test_count": 286,
                "start_weight": str(WEIGHT),
                "start_sha256": hashlib.sha256(WEIGHT.read_bytes()).hexdigest(),
                "external_commons_13": "excluded from fitting and checkpoint choice"}
    (DATA / "build_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available")
    meta = prepare()
    print(json.dumps({"dataset": str(DATA), "start_sha256": meta["start_sha256"],
                      "train": 2407, "val": 243, "test": 286}, indent=2), flush=True)
    model = YOLO(str(WEIGHT))
    model.train(data=str(DATA / "data.yaml"), epochs=40, patience=10,
                imgsz=640, batch=4, device=0, workers=0, seed=20260927,
                optimizer="AdamW", lr0=0.0003, lrf=0.01, weight_decay=0.0005,
                hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, degrees=3.0,
                translate=0.05, scale=0.2, fliplr=0.5,
                mosaic=0.4, mixup=0.0, close_mosaic=10,
                project=str(ROOT / "runs"), name=RUN.name, exist_ok=False,
                save=True, plots=True, val=True)
    (RUN / "manual25_run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
