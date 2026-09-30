"""Freeze and visualize predictions for the six user-supplied kitchen photos."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs" / "fire-binary-v2-no-manual-control" / "weights" / "best.pt"
OUT = ROOT / "out" / "user_six_probe_20260927"
INPUTS = [
    Path(r"C:\Users\ASUS\AppData\Local\Temp\codex-clipboard-6af0887e-4da1-4109-be5f-4536cc0a38a7.png"),
    Path(r"C:\Users\ASUS\AppData\Local\Temp\codex-clipboard-bc43e24b-c83b-4af9-8626-2642c5fbdb21.png"),
    Path(r"C:\Users\ASUS\AppData\Local\Temp\codex-clipboard-ebd5e471-9a88-4d4d-9a0d-d7b70ec8f764.png"),
    Path(r"C:\Users\ASUS\AppData\Local\Temp\codex-clipboard-ce4cccb3-e0ee-4845-8c76-0156097faf38.png"),
    Path(r"C:\Users\ASUS\AppData\Local\Temp\codex-clipboard-b4e4a9ba-a1ad-4638-937e-f0ee55e92c7e.png"),
    Path(r"C:\Users\ASUS\AppData\Local\Temp\codex-clipboard-438d6ae6-60e7-4fff-b361-3e1e5566e191.png"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing overwrite: {OUT}")
    if not WEIGHTS.is_file() or not all(p.is_file() for p in INPUTS):
        raise SystemExit("Missing model or input photo")
    OUT.mkdir(parents=True)
    frozen = []
    metadata = []
    for index, path in enumerate(INPUTS, 1):
        destination = OUT / f"{index:02d}_original.png"
        shutil.copy2(path, destination)
        if sha256(path) != sha256(destination):
            raise SystemExit(f"Copy hash mismatch: {path}")
        with Image.open(destination) as image:
            width, height = image.size
        frozen.append(destination)
        metadata.append({"index": index, "input": str(path),
                         "original": str(destination), "sha256": sha256(destination),
                         "width": width, "height": height})

    model = YOLO(str(WEIGHTS))
    paths = [str(p) for p in frozen]
    shown = model.predict(paths, imgsz=640, conf=0.25, iou=0.6,
                          device=0, verbose=False, save=False)
    low = model.predict(paths, imgsz=640, conf=0.05, iou=0.6,
                        device=0, verbose=False, save=False)
    output = []
    for item, result, low_result in zip(metadata, shown, low, strict=True):
        if len(result.boxes) and any(int(b.cls.item()) != 0 for b in result.boxes):
            raise SystemExit("Unexpected non-fire class")
        plotted = result.plot(labels=True, conf=True)
        boxed_path = OUT / f"{item['index']:02d}_boxed.png"
        Image.fromarray(plotted[:, :, ::-1]).save(boxed_path)
        boxes = [{"confidence": round(float(b.conf.item()), 4),
                  "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()]}
                 for b in result.boxes]
        item.update({"boxed": str(boxed_path), "boxes_025": len(boxes),
                     "boxes_050": sum(b["confidence"] >= 0.5 for b in boxes),
                     "max_conf_005": round(max((float(b.conf.item())
                                                  for b in low_result.boxes), default=0.0), 4),
                     "predictions": json.dumps(boxes)})
        output.append(item)
    with (OUT / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    provenance = {"weights": str(WEIGHTS), "weights_sha256": sha256(WEIGHTS),
                  "model_task": "single-class visible-fire detection",
                  "imgsz": 640, "display_confidence": 0.25, "nms_iou": 0.6,
                  "low_confidence_probe": 0.05, "images": len(output),
                  "note": "Photos are a user-selected diagnostic set, with no independent ground-truth boxes."}
    (OUT / "run_meta.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"provenance": provenance, "per_image": output},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
