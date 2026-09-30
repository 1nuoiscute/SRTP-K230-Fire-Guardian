"""Build a traceable development-only fine-tuning set from four user videos.

Six visually reviewed times per clip are used. Near-duplicate copies provide
training weight, not independent samples; they must never be counted as such.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = Path(r"D:\SRTP_Datasets\kitchen_fire_binary_codex_20260927")
SAMPLES = ROOT / "out" / "user_videos_20260927" / "sampled_frames"
TARGET = ROOT / "data" / "user_video_adaptation_20260927"
TIMES = ("02.5", "07.5", "12.5", "17.5", "22.5", "27.5")
REPEAT = 5
# Pixel xyxy on the original 720 x 1280 portrait frames. Contact sheets and
# full-resolution representative frames were visually reviewed before fixing.
BOXES = {
    0: [],  # Unlit burner: an intentional hard negative.
    1: [[184, 277, 514, 518]],  # Open blue burner ring.
    2: [[329, 344, 447, 393]],  # Small visible flame below the pan.
    3: [[184, 312, 544, 397]],  # Visible blue arc below the pan.
}
BASE_QUOTA = {
    "kitchen_stove_fire": 200,
    "ks_flame": 100,
    "generic_fire": 150,
    "nofire_real_indoor": 100,
    "object_pan": 100,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yolo_line(box: list[int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
    return f"0 {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}"


def main() -> None:
    if TARGET.exists():
        raise SystemExit(f"Refusing overwrite: {TARGET}")
    manifest = list(csv.DictReader((BASE / "manifest.csv").open(encoding="utf-8-sig")))
    base = []
    for source, count in BASE_QUOTA.items():
        candidates = [r for r in manifest if r["split"] == "train" and r["provenance"] == source]
        candidates.sort(key=lambda r: hashlib.sha256(r["sha256"].encode()).hexdigest())
        assert len(candidates) >= count
        base.extend(candidates[:count])
    (TARGET / "images" / "train").mkdir(parents=True)
    (TARGET / "labels" / "train").mkdir(parents=True)
    train_paths = [str(BASE / "images" / "train" / r["file"]) for r in base]
    details = []
    sheet = np.full((4 * 344, 6 * 202, 3), 30, dtype=np.uint8)
    for video in range(4):
        for col, time in enumerate(TIMES):
            src = SAMPLES / f"{video}_{time}s.jpg"
            frame = cv2.imread(str(src))
            if frame is None or frame.shape[:2] != (1280, 720):
                raise SystemExit(f"Missing or unexpected image dimensions: {src}")
            labels = [yolo_line(b, 720, 1280) for b in BOXES[video]]
            source_hash = sha256(src)
            for rep in range(REPEAT):
                stem = f"video{video}_{time}s_r{rep}"
                dst = TARGET / "images" / "train" / (stem + ".jpg")
                label = TARGET / "labels" / "train" / (stem + ".txt")
                shutil.copy2(src, dst)
                label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
                train_paths.append(str(dst))
            for x1, y1, x2, y2 in BOXES[video]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            thumb = cv2.resize(frame, (180, 320), interpolation=cv2.INTER_AREA)
            x, y = col * 202, video * 344
            sheet[y:y+320, x:x+180] = thumb
            cv2.putText(sheet, f"{video}.mp4 {time}s", (x+2, y+338),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            details.append({"video": f"{video}.mp4", "time": float(time),
                            "source": str(src), "sha256": source_hash,
                            "boxes_xyxy": BOXES[video], "copies": REPEAT})
    cv2.imwrite(str(TARGET / "label_review_sheet.jpg"), sheet)
    (TARGET / "train.txt").write_text("\n".join(p.replace("\\", "/") for p in train_paths) + "\n", encoding="utf-8")
    (TARGET / "data.yaml").write_text(
        f"path: {TARGET.as_posix()}\ntrain: {(TARGET / 'train.txt').as_posix()}\n"
        f"val: {(BASE / 'images' / 'val').as_posix()}\n"
        f"test: {(BASE / 'images' / 'test').as_posix()}\n"
        "nc: 1\nnames:\n  0: fire\n", encoding="utf-8")
    meta = {"purpose": "development adaptation only; all 4 clips are now train-exposed",
            "base": str(BASE), "base_selected": dict(Counter(r["provenance"] for r in base)),
            "sampled_source_frames": len(details), "copies_per_frame": REPEAT,
            "new_train_entries": len(details) * REPEAT,
            "total_train_entries": len(train_paths), "samples": details}
    (TARGET / "build_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "samples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
