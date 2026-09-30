"""Tugas Akhir 逐图来源判定（执行单 8.1 第 2 项：全 260 图逐图审核）。

目标：把 260 张分成「可作训练候选」与「hold/exclude」，并给出可核对的证据。

判定信号（只读、不联网）：
  S1 重复斜向水印：pngtree 类水印会在整幅图重复出现，产生**全图范围内的规则斜线纹理**。
  S2 水印条：iStock 类会在中部/右下出现「深色半透明条 + 高亮文字」。
  S3 缩略图版式：YouTube/短视频缩略图常有**纯色文字块**（大块高饱和/纯白区域）与描边大字。
  S4 专业摄影特征：浅景深、暗背景单点布光、影棚白底 —— 与"实机厨房照片"不符。
  S5 低分辨率/强压缩：网页抓取的常见特征。

输出：每张图的各信号分与建议结论；并导出可疑图供人眼确认。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFilter

ROOT = Path(r"D:\Tugas Akhir.v3-3.0.yolov11")
IMG_EXT = {".jpg", ".jpeg", ".png"}


def gx(im: Image.Image) -> Image.Image:
    return im.convert("L")


def s1_diagonal_watermark(im: Image.Image) -> float:
    """检测全图范围的规则斜向重复纹理（pngtree 类）。

    做法：在多个不同位置取小方块，测其"高亮像素比例"的一致性；
    重复水印会让彼此远离的区块呈现相似的亮度分布，而真实场景不会。
    """
    W, H = im.size
    g = gx(im)
    cells = []
    for ny in range(3):
        for nx in range(4):
            x = int(W * (0.10 + 0.25 * nx))
            y = int(H * (0.15 + 0.30 * ny))
            box = (x, y, min(x + int(W * 0.15), W), min(y + int(H * 0.20), H))
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                continue
            px = list(g.crop(box).getdata())
            cells.append(sum(1 for v in px if v > 235) / max(len(px), 1))
    if len(cells) < 6:
        return 0.0
    m = sum(cells) / len(cells)
    if m <= 0:
        return 0.0
    sd = math.sqrt(sum((c - m) ** 2 for c in cells) / len(cells))
    # 水印：各区块都有一点高亮且方差小；场景：要么都无，要么方差大
    spread_penalty = 1.0 / (1.0 + sd * 25)
    return round(min(m * 6.0, 1.0) * spread_penalty, 4)


def s2_watermark_bar(im: Image.Image) -> float:
    """中部/右下「深底 + 高亮文字」条带。"""
    W, H = im.size
    best = 0.0
    for (rx0, ry0, rx1, ry1) in ((0.45, 0.42, 1.00, 0.72), (0.50, 0.72, 1.00, 0.92)):
        box = (int(rx0 * W), int(ry0 * H), int(rx1 * W), int(ry1 * H))
        if box[2] - box[0] < 16 or box[3] - box[1] < 8:
            continue
        crop = gx(im).crop(box)
        n = crop.size[0] * crop.size[1]
        px = list(crop.getdata())
        bright = sum(1 for v in px if v > 225) / n
        dark = sum(1 for v in px if v < 60) / n
        e = crop.filter(ImageFilter.FIND_EDGES)
        ep = list(e.getdata())
        edge = sum(1 for v in ep if v > 80) / len(ep)
        # 水印条：既有亮字也有暗底，且边缘丰富
        best = max(best, min(bright * 3.0, 1.0) * min(edge * 8.0, 1.0))
    return round(best, 4)


def s3_overlay_block(im: Image.Image) -> float:
    """纯色文字块（缩略图版式）：大面积高饱和/纯白矩形。"""
    W, H = im.size
    small = im.convert("RGB").resize((80, 45), Image.Resampling.NEAREST)
    px = list(small.getdata())
    n = len(px)
    white = sum(1 for r, g, b in px if r > 235 and g > 235 and b > 235) / n
    sat = sum(1 for r, g, b in px
              if max(r, g, b) - min(r, g, b) > 120 and max(r, g, b) > 150) / n
    return round(min((white + sat) * 3.0, 1.0), 4)


def s4_studio_look(im: Image.Image) -> float:
    """影棚感：极暗背景占比大 + 主体高对比。"""
    g = gx(im).resize((160, 90))
    px = list(g.getdata())
    n = len(px)
    dark = sum(1 for v in px if v < 40) / n
    bright = sum(1 for v in px if v > 200) / n
    return round(min(dark * 2.0, 1.0) * min(bright * 3.0, 1.0), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path,
                    default=Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work"
                                 r"\out\tugas_source_verdict"))
    ap.add_argument("--export-top", type=int, default=24)
    args = ap.parse_args()
    if args.outdir.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{args.outdir}")
    (args.outdir / "suspect").mkdir(parents=True)

    files: list[tuple[str, Path]] = []
    for split in ("train", "valid", "test"):
        d = ROOT / split / "images"
        if d.exists():
            files.extend((split, p) for p in sorted(d.iterdir())
                         if p.suffix.lower() in IMG_EXT)
    print(f"Tugas 总图 {len(files)}")

    rows = []
    for split, p in files:
        try:
            with Image.open(p) as im0:
                im = im0.convert("RGB")
                W, H = im.size
                s1 = s1_diagonal_watermark(im)
                s2 = s2_watermark_bar(im)
                s3 = s3_overlay_block(im)
                s4 = s4_studio_look(im)
                kb = p.stat().st_size / 1024
        except Exception as e:  # noqa: BLE001
            rows.append({"split": split, "file": p.name, "error": str(e),
                         "total": 0, "verdict": "error"})
            continue
        total = round(s1 + s2 + s3 + s4, 4)
        # 判定阈值（保守：宁可 hold 也不误收）
        if s1 >= 0.25 or s2 >= 0.30 or s3 >= 0.40:
            verdict = "hold_watermark_or_overlay"
        elif total >= 0.55:
            verdict = "hold_suspect_stock_look"
        else:
            verdict = "candidate"
        rows.append({
            "split": split, "file": p.name, "w": W, "h": H, "kb": round(kb, 1),
            "S1_repeat_diagonal_wm": s1, "S2_wm_bar": s2, "S3_overlay_block": s3,
            "S4_studio_look": s4, "total": total, "verdict": verdict,
            "manual_verdict": "", "manual_reason": "",
        })

    cols = ["split", "file", "w", "h", "kb", "S1_repeat_diagonal_wm", "S2_wm_bar",
            "S3_overlay_block", "S4_studio_look", "total", "verdict",
            "manual_verdict", "manual_reason"]
    with (args.outdir / "tugas_verdict.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    vc = Counter(r.get("verdict") for r in rows)
    rows_sorted = sorted(rows, key=lambda r: -r.get("total", 0))
    for i, r in enumerate(rows_sorted[: args.export_top]):
        src = ROOT / r["split"] / "images" / r["file"]
        if src.exists():
            try:
                with Image.open(src) as im:
                    t = im.convert("RGB")
                    t.thumbnail((760, 760))
                    t.save(args.outdir / "suspect" / f"{i:02d}_{r['file'][:60]}", quality=88)
            except Exception:  # noqa: BLE001
                pass

    summary = {
        "checked_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 启发式，须人工确认",
        "total": len(rows),
        "verdict_counts": dict(vc),
        "confirmed_human_reviewed": {
            "Api-Blue-116": "可见 iStock 水印 + 'Credit: zoom-zoom' + 左下编号 91515926（已人眼确认）",
            "Api-Blue-127": "pngtree 重复斜向水印铺满画面（已人眼确认）",
            "Api-Blue-87":  "可见 iStock 水印 + 'Credit: kvkirillov' + 编号 856908552（已人眼确认）",
        },
        "contact_sheet_observations": [
            "拼接图 sheet_00 / sheet_03 中多张带 iStock 署名水印：lonelyking / artitwpd / akiyoko / "
            "PitukTV / Yummy_delight / Yuliia Vaisman / Pluk_Studio",
            "另见 123RF 水印",
            "另有 YouTube 缩略图版式：印尼语大字 '+TEPAT'/'KOMPOR API KECIL' 等纯色文字块",
            "其余为影棚级专业摄影（浅景深、暗背景单点布光、白底）",
        ],
        "signals": {"S1": "全图规则斜向重复水印", "S2": "深底高亮文字水印条",
                    "S3": "缩略图纯色文字块", "S4": "影棚感（暗背景+高对比主体）"},
        "recommendation": ("Tugas Akhir v3 整体 hold：260 张统一 `Api-` 前缀、同一批次来源，"
                           "混入 iStock/pngtree/123RF 图库水印图与短视频缩略图，"
                           "项目页的 CC BY 4.0 不覆盖这些嵌入内容。"
                           "执行单 8.1-2 明确：若 0 张合格就明确排除此源，不凑数。"),
    }
    (args.outdir / "tugas_source_verdict.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"判定分布: {dict(vc)}")
    print(f"高分前 12：")
    for r in rows_sorted[:12]:
        print(f"  {r.get('total'):.3f}  S1={r.get('S1_repeat_diagonal_wm')} "
              f"S2={r.get('S2_wm_bar')} S3={r.get('S3_overlay_block')} "
              f"S4={r.get('S4_studio_look')}  {r['split']:<6} {r['file'][:46]}")
    print(f"\n输出：{args.outdir}")


if __name__ == "__main__":
    main()
