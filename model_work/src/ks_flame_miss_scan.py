"""在「无 Flame 框」的图里筛查是否存在真实可见火焰（漏标检测）。

依据执行单 8.1-3：负例若画面中存在未框出的真实火苗，不能当作负例；
应补框或弃图。本脚本用颜色特征做初筛，产出待人工确认清单。

蓝焰：高蓝低红、亮度中等；橙焰：高红高绿低蓝、高亮度。
只筛「无 Flame 框」的图，输出分数与候选，供人眼复核。
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

KS = Path(r"D:\Kitchen Safety.v3i.yolov11")
ROWS = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out\ks_segments"
            r"\segment_rows.csv")
OUT = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out\ks_flame_miss")
NAMES = ["Burner", "Flame", "Smoke", "Spill", "safe"]


def boxes(r):
    p = KS / r["split"] / "labels" / (r["stem"] + ".txt")
    out = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            f = ln.split()
            if len(f) >= 5:
                out.append(int(f[0]))
    return out


def flame_pixels(im: Image.Image) -> tuple[float, float, int]:
    """返回 (蓝焰像素比, 橙焰像素比, 两种合计像素数)。"""
    sm = im.convert("RGB").resize((320, 180), Image.Resampling.BILINEAR)
    px = list(sm.getdata())
    n = len(px)
    blue = orange = 0
    for r, g, b in px:
        # 蓝焰：蓝显著高于红、且不是纯黑背景 / 不是蓝天
        if b > 120 and b - r > 55 and b - g > 30 and g > 60:
            blue += 1
        # 橙焰：红最高、蓝最低、亮度高
        elif r > 165 and r - b > 85 and g > 70 and g < r:
            orange += 1
    return blue / n, orange / n, blue + orange


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
    print(f"无 Flame 框的图 {len(noflame)} 张，逐张筛查疑似漏标火焰 ...")

    scored = []
    for r in noflame:
        p = KS / r["split"] / "images" / r["file"]
        try:
            with Image.open(p) as im0:
                im = im0.convert("RGB")
            blue, orange, cnt = flame_pixels(im)
            scored.append({**r, "blue": round(blue, 5), "orange": round(orange, 5),
                           "flame_px": cnt})
        except Exception as e:  # noqa: BLE001
            scored.append({**r, "blue": 0.0, "orange": 0.0, "flame_px": -1,
                           "error": str(e)})

    scored.sort(key=lambda x: -(x["blue"] + x["orange"]))
    with (args.out / "miss_scores.csv").open("w", encoding="utf-8", newline="") as f:
        cols = ["split", "file", "stem", "segment", "empty", "classes",
                "blue", "orange", "flame_px"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(scored)

    # 阈值：蓝焰或橙焰像素占比明显高于噪声
    SUSPECT = 0.004
    suspects = [s for s in scored if s["blue"] + s["orange"] >= SUSPECT]
    print(f"\n疑似含未标注火焰（阈值 {SUSPECT}）：{len(suspects)} 张")
    for s in suspects[:20]:
        print(f"  blue={s['blue']:.4f} orange={s['orange']:.4f} "
              f"{s['segment']:<9} empty={s['empty']} [{s['classes']:<14}] {s['file'][:44]}")

    # 导出前 12 张供人眼确认
    for i, s in enumerate(scored[:12]):
        try:
            with Image.open(KS / s["split"] / "images" / s["file"]) as im:
                t = im.convert("RGB")
                t.thumbnail((640, 640))
                t.save(args.out / "top" / f"{i:02d}_b{s['blue']:.3f}_o{s['orange']:.3f}"
                                          f"__{s['file'][:44]}.jpg", quality=88)
        except Exception:  # noqa: BLE001
            pass

    by_seg = Counter(s["segment"] for s in suspects)
    print(f"\n疑似漏标按段: {dict(by_seg)}")
    print(f"输出：{args.out}")


if __name__ == "__main__":
    main()
