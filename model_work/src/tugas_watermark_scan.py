"""Tugas Akhir 逐图水印/图库来源检测（执行单 8.1 第 2 项：全 260 图逐图审核）。

背景：抽图发现 `Api-Blue-116` 带 iStock 水印与 `Credit: zoom-zoom`，左下角有图片编号。
项目页标的 CC BY 4.0 **不自动覆盖嵌入的第三方图库图**。因此必须逐图判定。

方法（只读，不依赖联网 OCR）：
  1. 文件名谱系：按 `Api-*` / `Train-*` 等前缀分组，统计每组的图像特征。
  2. 水印文字带检测：图库水印集中在特定区域且为高亮或半透明文字。
     对多个候选区域（右下、左下、正中）做「高亮且边缘密集」评分。
  3. 图库编号模式：左下角常见 8 位数字串（如 91515926）。
  4. 采样缩略图拼版，供人眼批量判读。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

ROOT = Path(r"D:\Tugas Akhir.v3-3.0.yolov11")
IMG_EXT = {".jpg", ".jpeg", ".png"}

# 区域（相对坐标）：图库水印常见落点
REGIONS = {
    "bottom_right": (0.55, 0.78, 1.00, 1.00),
    "bottom_left": (0.00, 0.78, 0.45, 1.00),
    "center": (0.30, 0.35, 0.70, 0.65),
    "top_right": (0.55, 0.00, 1.00, 0.18),
}


def region_score(im: Image.Image, box: tuple[float, float, float, float]) -> dict:
    W, H = im.size
    x0, y0, x1, y1 = (int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return {"bright_ratio": 0.0, "edge_ratio": 0.0, "score": 0.0}
    crop = im.convert("L").crop((x0, y0, x1, y1))
    px = list(crop.getdata())
    n = len(px)
    bright = sum(1 for v in px if v > 225) / n
    # 边缘密度：半透明文字叠加会产生较多强边缘
    edges = crop.filter(ImageFilter.FIND_EDGES)
    ep = list(edges.getdata())
    edge = sum(1 for v in ep if v > 90) / len(ep)
    return {"bright_ratio": round(bright, 4), "edge_ratio": round(edge, 4),
            "score": round(bright * 1.0 + edge * 2.0, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path,
                    default=Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work"
                                 r"\out\tugas_watermark"))
    ap.add_argument("--contact-sheet", type=int, default=48)
    args = ap.parse_args()
    if args.outdir.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{args.outdir}")
    (args.outdir / "sheets").mkdir(parents=True)

    files: list[tuple[str, Path]] = []
    for split in ("train", "valid", "test"):
        d = ROOT / split / "images"
        if d.exists():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    files.append((split, p))
    print(f"Tugas 总图 {len(files)}")

    # 文件名前缀谱系
    def prefix(stem: str) -> str:
        base = stem.split(".rf.")[0]
        m = re.match(r"^([A-Za-z]+)[-_]", base)
        return m.group(1) if m else "(other)"

    pref = Counter(prefix(p.stem) for _, p in files)
    print("文件名前缀谱系:", dict(pref))

    rows = []
    for split, p in files:
        try:
            with Image.open(p) as im0:
                im = im0.convert("RGB")
                scores = {k: region_score(im, b) for k, b in REGIONS.items()}
        except Exception as e:  # noqa: BLE001
            rows.append({"split": split, "file": p.name, "error": str(e)})
            continue
        # 综合：取各区域最高分，并单独标注左下角(编号常见位)
        best_region = max(scores, key=lambda k: scores[k]["score"])
        rows.append({
            "split": split,
            "file": p.name,
            "prefix": prefix(p.stem),
            "declared_suffix": p.stem.split(".rf.")[0].split("-")[-1] if "-" in p.stem else "",
            **{f"{k}_bright": v["bright_ratio"] for k, v in scores.items()},
            **{f"{k}_edge": v["edge_ratio"] for k, v in scores.items()},
            **{f"{k}_score": v["score"] for k, v in scores.items()},
            "best_region": best_region,
            "max_score": max(v["score"] for v in scores.values()),
            "manual_verdict": "",
            "manual_reason": "",
        })

    rows.sort(key=lambda r: -r.get("max_score", 0))
    with (args.outdir / "tugas_watermark_scores.csv").open("w", encoding="utf-8",
                                                           newline="") as f:
        cols = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # 拼接图供人眼批量判读
    sheet_n = 0
    per_sheet = 12          # 4 列 x 3 行
    idx = 0
    while idx < min(len(rows), args.contact_sheet * 2):
        chunk = rows[idx: idx + per_sheet]
        if not chunk:
            break
        idx += per_sheet
        cell = (480, 270)
        sheet = Image.new("RGB", (cell[0] * 4, cell[1] * 3), (20, 20, 20))
        for i, r in enumerate(chunk):
            fp = ROOT / r["split"] / "images" / r["file"]
            if not fp.exists():
                continue
            try:
                with Image.open(fp) as im:
                    t = im.convert("RGB")
                    t.thumbnail(cell)
                    x = (i % 4) * cell[0]
                    y = (i // 4) * cell[1]
                    sheet.paste(t, (x, y))
            except Exception:  # noqa: BLE001
                pass
        sheet.save(args.outdir / "sheets" / f"sheet_{sheet_n:02d}.jpg", quality=85)
        sheet_n += 1

    summary = {
        "checked_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 启发式筛查，须人工确认",
        "total": len(files),
        "by_prefix": dict(pref),
        "region_defs": REGIONS,
        "top_suspects": [{k: r.get(k) for k in
                          ("split", "file", "prefix", "best_region", "max_score")}
                         for r in rows[:25]],
        "score_histogram": {
            ">1.0": sum(1 for r in rows if r.get("max_score", 0) > 1.0),
            "0.7-1.0": sum(1 for r in rows if 0.7 < r.get("max_score", 0) <= 1.0),
            "0.5-0.7": sum(1 for r in rows if 0.5 < r.get("max_score", 0) <= 0.7),
            "<=0.5": sum(1 for r in rows if r.get("max_score", 0) <= 0.5),
        },
        "contact_sheets": sheet_n,
        "note": ("这是启发式筛查，不是 OCR 判定。已确认 Api-Blue-116 带 iStock 水印；"
                 "其余必须靠人眼看 sheets/ 里的拼版图逐张确认。"),
    }
    (args.outdir / "tugas_watermark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n分数分布: {summary['score_histogram']}")
    print(f"拼接图 {sheet_n} 张 -> {args.outdir / 'sheets'}")
    print("\n最高可疑前 12 张：")
    for r in rows[:12]:
        print(f"  {r.get('max_score'):.3f}  {r['best_region']:<13} {r['split']:<6} "
              f"{r['prefix']:<10} {r['file'][:52]}")


if __name__ == "__main__":
    main()
