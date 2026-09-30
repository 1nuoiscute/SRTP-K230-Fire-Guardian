"""按「原类别组合」分组导出拼版，用于逐组目视确认「无 Flame 图是否真的无火」。

执行单 8.1-3 要求：删除非目标类别框后标签空了的图，必须确认确实无可见火焰，
才可当负例。这里给出紧凑证据：每个组合一张拼版。
"""
from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

KS = Path(r"D:\Kitchen Safety.v3i.yolov11")
ROWS = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out\ks_segments"
            r"\segment_rows.csv")
OUT = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out\ks_group_sheets")
NAMES = ["Burner", "Flame", "Smoke", "Spill", "safe"]

if OUT.exists():
    import shutil
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

rows = list(csv.DictReader(ROWS.open(encoding="utf-8")))


def boxes(r):
    p = KS / r["split"] / "labels" / (r["stem"] + ".txt")
    out = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            f = ln.split()
            if len(f) >= 5:
                out.append(int(f[0]))
    return out


def combo(r):
    cs = set(boxes(r))
    if 1 in cs:
        return None
    return "+".join(NAMES[c] for c in sorted(cs)) or "EMPTY"


groups: dict[str, list[dict]] = defaultdict(list)
for r in rows:
    k = combo(r)
    if k:
        groups[k].append(r)

for name, items in groups.items():
    cols, rows_n = 4, min(6, max(1, (len(items) + 3) // 4))
    cell = (400, 225)
    canvas = Image.new("RGB", (cell[0] * cols, cell[1] * rows_n + 26), (15, 15, 15))
    dr = ImageDraw.Draw(canvas)
    dr.text((6, 6), f"NoFlame group = {name}   total={len(items)}   "
                    f"segments={dict(Counter(x['segment'] for x in items))}",
            fill=(255, 255, 255))
    for i, r in enumerate(items[: cols * rows_n]):
        try:
            with Image.open(KS / r["split"] / "images" / r["file"]) as im0:
                t = im0.convert("RGB")
            t.thumbnail(cell)
            W, H = t.size
            d2 = ImageDraw.Draw(t)
            for ln in (KS / r["split"] / "labels" / (r["stem"] + ".txt")
                       ).read_text(encoding="utf-8", errors="replace").splitlines():
                f = ln.split()
                if len(f) < 5:
                    continue
                c = int(f[0])
                cx, cy, w, h = (float(x) for x in f[1:5])
                x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
                x1, y1 = (cx + w / 2) * W, (cy + h / 2) * H
                col = [(255, 70, 70), (60, 230, 120), (0, 170, 255),
                       (255, 220, 60), (220, 120, 255)][c % 5]
                d2.rectangle([x0, y0, x1, y1], outline=col, width=2)
            x = (i % cols) * cell[0] + (cell[0] - W) // 2
            y = (i // cols) * cell[1] + 26
            canvas.paste(t, (x, y))
        except Exception:  # noqa: BLE001
            pass
    safe = name.replace("+", "_").replace("(", "").replace(")", "")
    p = OUT / f"{safe}.jpg"
    canvas.save(p, quality=84)
    print(f"  {name:<24} {len(items):>4} 张 -> {p.name}")

print(f"\n图例：红=Burner 绿=Flame 蓝=Smoke 黄=Spill 紫=safe")
print(f"输出：{OUT}")
