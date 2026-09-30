"""Rank blue-colored pixels inside existing fire labels for manual review.

HSV counts are only a triage hint, never a flame annotation or class truth.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


DATA = Path(r"D:\SRTP_Datasets\kitchen_fire_binary_codex_20260927")
PROVENANCE = {"kitchen_stove_fire", "ks_flame"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing overwrite: {args.out}")
    manifest = list(csv.DictReader((DATA / "manifest.csv").open(encoding="utf-8-sig")))
    result = []
    for row in manifest:
        if row["split"] != "train" or row["provenance"] not in PROVENANCE:
            continue
        image = DATA / "images" / "train" / row["file"]
        label = DATA / "labels" / "train" / (image.stem + ".txt")
        img = cv2.imread(str(image))
        if img is None:
            raise SystemExit(f"Unreadable image: {image}")
        height, width = img.shape[:2]
        for box_index, line in enumerate(label.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            cls, x, y, w, h = map(float, line.split())
            if int(cls) != 0:
                continue
            x1, x2 = max(0, int((x - w / 2) * width)), min(width, int((x + w / 2) * width))
            y1, y2 = max(0, int((y - h / 2) * height)), min(height, int((y + h / 2) * height))
            if x2 <= x1 or y2 <= y1:
                continue
            hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
            blue = cv2.inRange(hsv, (85, 55, 45), (135, 255, 255))
            blue_pixels = int(cv2.countNonZero(blue))
            area = (x2 - x1) * (y2 - y1)
            result.append({"file": row["file"], "provenance": row["provenance"],
                           "group_id": row["group_id"], "box_index": box_index,
                           "box_area": area, "blue_pixels": blue_pixels,
                           "blue_fraction": round(blue_pixels / area, 5)})
    result.sort(key=lambda r: r["blue_pixels"], reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f"Scored {len(result)} fire boxes; top 20:")
    for row in result[:20]:
        print(row["provenance"], row["file"], row["blue_pixels"], row["blue_fraction"])


if __name__ == "__main__":
    main()
