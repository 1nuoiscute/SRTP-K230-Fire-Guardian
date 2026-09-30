"""Evaluate frozen manual25 v2 on unchanged complete 286-image test split."""
from pathlib import Path
import json
from ultralytics import YOLO

ROOT = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work")
DATA = Path(r"D:\SRTP_Datasets\kitchen_fire_binary_codex_20260927\data.yaml")
WEIGHT = ROOT / "runs" / "fire-binary-codex-v2-manual25" / "weights" / "best.pt"
OUT = ROOT / "out" / "fire_binary_manual25_v2_fulltest_20260927"
if OUT.exists():
    raise SystemExit(f"Refusing overwrite: {OUT}")
model = YOLO(str(WEIGHT))
metrics = model.val(data=str(DATA), split="test", imgsz=640, batch=4, device=0,
                    workers=0, plots=False, project=str(ROOT / "out"),
                    name=OUT.name, exist_ok=False)
result = {"weights": str(WEIGHT), "data": str(DATA), "split": "test",
          "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
          "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)}
(OUT / "metrics_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
