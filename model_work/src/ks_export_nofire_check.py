"""导出指定拍摄段的抽样图，供人眼确认「空标签图是否真的无火」。"""
from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

KS = Path(r"D:\Kitchen Safety.v3i.yolov11")
ROWS = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out\ks_segments"
            r"\segment_rows.csv")
OUT = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out\ks_nofire_check")
NAMES = ["Burner", "Flame", "Smoke", "Spill", "safe"]

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

rows = list(csv.DictReader(ROWS.open(encoding="utf-8")))


def draw(r: dict, dest: Path) -> bool:
    src = KS / r["split"] / "images" / r["file"]
    if not src.exists():
        return False
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((760, 760))
            W, H = im.size
            dr = ImageDraw.Draw(im)
            lp = KS / r["split"] / "labels" / (Path(r["file"]).stem + ".txt")
            if lp.exists():
                for ln in lp.read_text(encoding="utf-8", errors="replace").splitlines():
                    f = ln.split()
                    if len(f) < 5:
                        continue
                    c = int(f[0])
                    cx, cy, w, h = (float(x) for x in f[1:5])
                    x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
                    x1, y1 = (cx + w / 2) * W, (cy + h / 2) * H
                    col = [(255, 70, 70), (70, 220, 120), (0, 170, 255),
                           (255, 220, 60), (220, 120, 255)][c % 5]
                    dr.rectangle([x0, y0, x1, y1], outline=col, width=3)
                    dr.text((x0 + 3, max(0, y0 - 12)), NAMES[c], fill=col)
            im.save(dest, quality=88)
        return True
    except Exception:  # noqa: BLE001
        return False


# 1) seg006 的空标签图（候选无火负例）
seg006_empty = [r for r in rows if r["segment"] == "seg006" and r["empty"] == "1"]
seg006_lab = [r for r in rows if r["segment"] == "seg006" and r["empty"] == "0"]
print(f"seg006: 空标签 {len(seg006_empty)}；有标签 {len(seg006_lab)}")
random.seed(20260926)
random.shuffle(seg006_empty)
n = 0
for r in seg006_empty[:8]:
    if draw(r, OUT / f"seg006_empty_{n:02d}__{r['file'][:44]}.jpg"):
        n += 1
for i, r in enumerate(seg006_lab[:2]):
    draw(r, OUT / f"seg006_labeled_{i:02d}__{r['file'][:44]}.jpg")

# 2) seg008：只有 Smoke/safe、无 Flame（28 张）
seg008 = [r for r in rows if r["segment"] == "seg008"]
for i, r in enumerate(seg008[:4]):
    draw(r, OUT / f"seg008_{i:02d}__{r['file'][:44]}.jpg")

# 3) 全数据集的空标签图（不只 seg006）
all_empty = [r for r in rows if r["empty"] == "1" and r["segment"] != "seg006"]
print(f"seg006 之外的空标签图: {len(all_empty)}")
for i, r in enumerate(all_empty[:4]):
    draw(r, OUT / f"otherempty_{i:02d}_{r['segment']}__{r['file'][:40]}.jpg")

print("\n导出清单:")
for p in sorted(OUT.iterdir()):
    print("  ", p.name)
