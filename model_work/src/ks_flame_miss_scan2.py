"""在「无 Flame 框」的图里筛查真实可见火焰（修正版）。

前一版判据 `b>120 and b-r>55 and b-g>30` 被**蓝色橱柜**大量误报
（seg006 的 100 帧我目视确认无火，却全部命中）。
本版收紧：蓝焰是**高饱和、高亮度**的小面积；橱柜是**大面积、低饱和**。

判据：
  蓝焰像素：b>=170 且 b-r>=110 且 b-g>=60 且 r<110
  橙焰像素：r>=185 且 r-b>=110 且 g>=80
并报告「最大连通团面积占比」，避免大面积误报。
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image

KS = Path(r"D:\Kitchen Safety.v3i.yolov11")
ROWS = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out\ks_segments"
            r"\segment_rows.csv")
OUT = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out\ks_flame_miss2")


def boxes(r):
    p = KS / r["split"] / "labels" / (r["stem"] + ".txt")
    out = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            f = ln.split()
            if len(f) >= 5:
                out.append(int(f[0]))
    return out


def scan(im: Image.Image):
    sm = im.convert("RGB").resize((320, 180), Image.Resampling.BILINEAR)
    W, H = sm.size
    px = list(sm.getdata())
    blue = orange = 0
    for r, g, b in px:
        if b >= 170 and b - r >= 110 and b - g >= 60 and r < 110:
            blue += 1
        elif r >= 185 and r - b >= 110 and g >= 80:
            orange += 1
    n = W * H
    # 最大连通团（4 邻域），用于区分「火焰小团」与「大面积色块」
    mask = bytearray(n)
    for i, (r, g, b) in enumerate(px):
        if (b >= 170 and b - r >= 110 and b - g >= 60 and r < 110) or \
           (r >= 185 and r - b >= 110 and g >= 80):
            mask[i] = 1
    seen = bytearray(n)
    best = 0
    for i in range(n):
        if mask[i] and not seen[i]:
            stack = [i]
            seen[i] = 1
            cnt = 0
            while stack:
                j = stack.pop()
                cnt += 1
                x, y = j % W, j // W
                for nj in ((j - 1 if x > 0 else -1), (j + 1 if x < W - 1 else -1),
                           (j - W if y > 0 else -1), (j + W if y < H - 1 else -1)):
                    if nj >= 0 and mask[nj] and not seen[nj]:
                        seen[nj] = 1
                        stack.append(nj)
            best = max(best, cnt)
    return blue / n, orange / n, best / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.out.exists():
        import shutil
        shutil.rmtree(args.out)
    (args.out / "top").mkdir(parents=True)

    rows = list(csv.DictReader(ROWS.open(encoding="utf-8")))
    noflame = [r for r in rows if 1 not in set(boxes(r))]
    print(f"无 Flame 框的图 {len(noflame)} 张")

    scored = []
    for r in noflame:
        try:
            with Image.open(KS / r["split"] / "images" / r["file"]) as im0:
                b, o, blob = scan(im0.convert("RGB"))
            scored.append({**r, "blue": round(b, 5), "orange": round(o, 5),
                           "blob": round(blob, 5)})
        except Exception as e:  # noqa: BLE001
            scored.append({**r, "blue": 0.0, "orange": 0.0, "blob": 0.0, "error": str(e)})

    scored.sort(key=lambda x: -(x["blue"] + x["orange"]))
    with (args.out / "miss_scores2.csv").open("w", encoding="utf-8", newline="") as f:
        cols = ["split", "file", "stem", "segment", "empty", "classes",
                "blue", "orange", "blob"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(scored)

    # 阈值：小面积高饱和火焰
    SUS = 0.0015
    sus = [s for s in scored if (s["blue"] + s["orange"]) >= SUS]
    print(f"疑似漏标（阈值 {SUS}）：{len(sus)} 张")
    for s in sus[:15]:
        print(f"  blue={s['blue']:.5f} orange={s['orange']:.5f} blob={s['blob']:.5f} "
              f"{s['segment']:<9} [{s['classes']:<10}] {s['file'][:42]}")
    print(f"\n按段: {dict(Counter(s['segment'] for s in sus))}")

    for i, s in enumerate(scored[:16]):
        try:
            with Image.open(KS / s["split"] / "images" / s["file"]) as im:
                t = im.convert("RGB")
                t.thumbnail((640, 640))
                t.save(args.out / "top" / f"{i:02d}_b{s['blue']:.4f}_o{s['orange']:.4f}"
                                          f"__{s['file'][:42]}.jpg", quality=88)
        except Exception:  # noqa: BLE001
            pass
    print(f"\n输出：{args.out}")


if __name__ == "__main__":
    main()
