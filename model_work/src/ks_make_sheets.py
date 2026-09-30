"""密集拼版：批量目视确认 seg006 空标签帧是否真的无火，以及 seg008 的 Smoke 是什么。"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

KS = Path(r"D:\Kitchen Safety.v3i.yolov11")
ROWS = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out\ks_segments"
            r"\segment_rows.csv")
OUT = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out\ks_sheets")
NAMES = ["Burner", "Flame", "Smoke", "Spill", "safe"]

if OUT.exists():
    import shutil
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

rows = list(csv.DictReader(ROWS.open(encoding="utf-8")))


def render(r: dict, cell: tuple[int, int]) -> Image.Image:
    src = KS / r["split"] / "images" / r["file"]
    with Image.open(src) as im0:
        im = im0.convert("RGB")
    im.thumbnail(cell)
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
            col = [(255, 70, 70), (60, 230, 120), (0, 170, 255),
                   (255, 220, 60), (220, 120, 255)][c % 5]
            dr.rectangle([x0, y0, x1, y1], outline=col, width=2)
    return im


def sheet(items: list[dict], cols: int, rows_: int, cell: tuple[int, int],
          title: str) -> Path:
    W, H = cell[0] * cols, cell[1] * rows_
    canvas = Image.new("RGB", (W, H + 26), (18, 18, 18))
    dr = ImageDraw.Draw(canvas)
    dr.text((6, 6), title, fill=(255, 255, 255))
    for i, r in enumerate(items[: cols * rows_]):
        try:
            t = render(r, cell)
        except Exception:  # noqa: BLE001
            continue
        x = (i % cols) * cell[0] + (cell[0] - t.size[0]) // 2
        y = (i // cols) * cell[1] + 26
        canvas.paste(t, (x, y))
    p = OUT / f"{title.split()[0]}.jpg"
    canvas.save(p, quality=86)
    return p


# A. seg006 空标签帧 24 张（4x6）
seg006e = [r for r in rows if r["segment"] == "seg006" and r["empty"] == "1"]
random.seed(7)
random.shuffle(seg006e)
p = sheet(seg006e, 4, 6, (400, 225), f"seg006_empty24 n={len(seg006e)}")
print("生成", p)

# B. seg008 全部 28 张（4x7）
seg008 = [r for r in rows if r["segment"] == "seg008"]
p = sheet(seg008, 4, 7, (400, 225), f"seg008_all n={len(seg008)}")
print("生成", p)

# C. 各含 Flame 段的抽样（确认映射目标正确）
fl = [r for r in rows if r["has_flame"] == "1"]
random.shuffle(fl)
p = sheet(fl, 4, 6, (400, 225), f"flame_samples n={len(fl)}")
print("生成", p)

print("\n图例：红=Burner 绿=Flame 蓝=Smoke 黄=Spill 紫=safe")
for f in sorted(OUT.iterdir()):
    print("  ", f.name)
