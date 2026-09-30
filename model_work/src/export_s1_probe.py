"""导出 S1（重复斜向水印）高分与低分对照图，供人眼核实检测是否可信。"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

OUT = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out\tugas_source_verdict")
ROOT = Path(r"D:\Tugas Akhir.v3-3.0.yolov11")

rows = list(csv.DictReader((OUT / "tugas_verdict.csv").open(encoding="utf-8")))
for r in rows:
    for k in ("S1_repeat_diagonal_wm", "S2_wm_bar", "S3_overlay_block",
              "S4_studio_look", "total"):
        try:
            r[k] = float(r[k])
        except (KeyError, ValueError):
            r[k] = 0.0

rows.sort(key=lambda r: -r["S1_repeat_diagonal_wm"])

dest = OUT / "s1_probe"
if dest.exists():
    shutil.rmtree(dest)
dest.mkdir(parents=True)

print("=== S1 最高的 12 张（导出供人眼核实）===")
for i, r in enumerate(rows[:12]):
    src = ROOT / r["split"] / "images" / r["file"]
    if src.exists():
        name = f"{i:02d}_S1_{r['S1_repeat_diagonal_wm']:.3f}__{r['file'][:48]}.jpg"
        shutil.copy2(src, dest / name)
        print(f"   S1={r['S1_repeat_diagonal_wm']:.3f}  {r['split']:<6} {r['file'][:56]}")

print("\n=== S1 最低的 3 张（对照）===")
for r in rows[-3:]:
    src = ROOT / r["split"] / "images" / r["file"]
    if src.exists():
        name = f"zz_low_S1_{r['S1_repeat_diagonal_wm']:.3f}__{r['file'][:48]}.jpg"
        shutil.copy2(src, dest / name)
        print(f"   S1={r['S1_repeat_diagonal_wm']:.3f}  {r['split']:<6} {r['file'][:56]}")

print(f"\n导出目录：{dest}")
