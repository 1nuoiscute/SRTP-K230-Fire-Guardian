"""只读标注质量审计：对 gc_kitchen_annotation v2 做几何异常筛查 + 人工复核队列导出。

目的（对应执行单第 1 项）：
  在不改任何原标签的前提下，用几何统计把「值得人眼看的」标注挑出来排队，
  并导出带真值框的可视化图，供人工判读「正确 / 错误 / 不确定」。

只读：只读 D 盘原目录；所有输出写到 --outdir 新目录，拒绝覆盖。

注意边界：几何异常 ≠ 标注错误。本脚本只产出**候选线索**与可视化，判定必须由人做。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

CLASS_NAMES = ("flame-reflection", "flame-under-pot", "gas-stove-flame", "no-flame")
IMG_EXT = {".jpg", ".jpeg", ".png"}

# 几何阈值（保守：只挑明显可疑的，避免淹没人工）
MIN_AREA = 0.0005      # 相对面积 < 0.05% 视为极小
MIN_SIDE = 0.01        # 任一边 < 1% 图宽/高
MIN_EDGE_MARGIN = 0.002  # 距图边 < 0.2% 视为贴边
EXTREME_AR = 6.0       # 长宽比 > 6 视为极端长条
HUGE_AREA = 0.85       # 相对面积 > 85% 视为几乎整图


def group_of(path: Path) -> str:
    b = path.stem
    i = b.find(".rf.")
    return b[:i] if i > 0 else b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path(r"D:\gc_kitchen_annotation.v2-version_two.yolov11"))
    ap.add_argument("--outdir", type=Path,
                    default=Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work"
                                 r"\out\gc_v2_label_audit"))
    ap.add_argument("--max-viz", type=int, default=120,
                    help="最多导出多少张带框可视化图")
    args = ap.parse_args()

    if args.outdir.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{args.outdir}")
    (args.outdir / "viz").mkdir(parents=True)

    rows: list[dict] = []
    flags: Counter[str] = Counter()
    per_class = defaultdict(lambda: {"n": 0, "area": [], "ar": [], "w": [], "h": []})
    split_stats = {}

    for split in ("train", "valid", "test"):
        lbl_dir = args.root / split / "labels"
        img_dir = args.root / split / "images"
        if not lbl_dir.exists():
            continue
        n_img = n_lbl = n_box = n_empty = 0
        for lbl in sorted(lbl_dir.glob("*.txt")):
            n_lbl += 1
            img = None
            for ext in sorted(IMG_EXT):
                cand = img_dir / (lbl.stem + ext)
                if cand.exists():
                    img = cand
                    break
            if img is None:
                # 标签与图片扩展名不一致时，按 stem 精确匹配任意扩展名
                for cand in img_dir.iterdir():
                    if cand.stem == lbl.stem and cand.suffix.lower() in IMG_EXT:
                        img = cand
                        break
            boxes = []
            for ln in lbl.read_text(encoding="utf-8", errors="replace").splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                try:
                    cid = int(p[0])
                    cx, cy, w, h = (float(x) for x in p[1:5])
                except ValueError:
                    flags["unparsable_line"] += 1
                    continue
                boxes.append((cid, cx, cy, w, h))
                n_box += 1
                per_class[cid]["n"] += 1
                per_class[cid]["area"].append(w * h)
                per_class[cid]["ar"].append(max(w, h) / max(min(w, h), 1e-9))
                per_class[cid]["w"].append(w)
                per_class[cid]["h"].append(h)
            if not boxes:
                n_empty += 1

            # 该图的异常标记
            my: list[str] = []
            for cid, cx, cy, w, h in boxes:
                x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                if not (0 <= x0 and 0 <= y0 and x1 <= 1 and y1 <= 1):
                    my.append("out_of_bounds")
                if w * h < MIN_AREA:
                    my.append("tiny_area")
                if min(w, h) < MIN_SIDE:
                    my.append("thin_side")
                if max(w, h) / max(min(w, h), 1e-9) > EXTREME_AR:
                    my.append("extreme_aspect")
                if w * h > HUGE_AREA:
                    my.append("near_full_frame")
                if min(x0, y0, 1 - x1, 1 - y1) < MIN_EDGE_MARGIN:
                    my.append("touching_edge")
            # 同图多个同类框高度重叠（可能重复标）
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    if a[0] != b[0]:
                        continue
                    ix = max(0, min(a[1] + a[3] / 2, b[1] + b[3] / 2)
                             - max(a[1] - a[3] / 2, b[1] - b[3] / 2))
                    iy = max(0, min(a[2] + a[4] / 2, b[2] + b[4] / 2)
                             - max(a[2] - a[4] / 2, b[2] - b[4] / 2))
                    inter = ix * iy
                    if inter > 0.8 * min(a[3] * a[4], b[3] * b[4]):
                        my.append("duplicate_overlap")
                        break
            if len(boxes) >= 6:
                my.append("many_boxes")

            my = sorted(set(my))
            for f in my:
                flags[f] += 1
            if img is not None:
                n_img += 1
            rows.append({
                "split": split,
                "file": lbl.name,
                "img": str(img) if img else "",
                "group": group_of(lbl),
                "n_boxes": len(boxes),
                "classes": "|".join(str(c) for c in sorted({b[0] for b in boxes})),
                "flags": "|".join(my),
                "img_found": bool(img),
            })
        split_stats[split] = {
            "images": n_img, "labels": n_lbl, "boxes": n_box, "empty_label_images": n_empty,
        }

    # 统计汇总
    summary = {}
    for cid, d in sorted(per_class.items()):
        if not d["n"]:
            continue
        area = sorted(d["area"])
        ar = sorted(d["ar"])
        summary[CLASS_NAMES[cid] if cid < 4 else str(cid)] = {
            "boxes": d["n"],
            "area_median": round(area[len(area) // 2], 6),
            "area_p05": round(area[int(len(area) * 0.05)], 6),
            "area_p95": round(area[min(int(len(area) * 0.95), len(area) - 1)], 6),
            "aspect_median": round(ar[len(ar) // 2], 3),
            "aspect_max": round(ar[-1], 3),
        }

    # 人工复核队列：按异常标记数 + 类别覆盖排序
    queue = [r for r in rows if r["flags"]]
    queue.sort(key=lambda r: (-len(r["flags"].split("|")), r["split"], r["file"]))

    with (args.outdir / "label_audit_rows.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with (args.outdir / "human_review_queue.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["priority", "split", "file", "img", "group", "n_boxes",
                    "classes", "flags", "verdict", "reason"])
        for i, r in enumerate(queue, 1):
            w.writerow([i, r["split"], r["file"], r["img"], r["group"], r["n_boxes"],
                        r["classes"], r["flags"], "", ""])

    # 导出带真值框的可视化（便于人眼判读）
    made = 0
    for r in queue:
        if made >= args.max_viz:
            break
        if not r["img"]:
            continue
        try:
            with Image.open(r["img"]) as im:
                im = im.convert("RGB")
                W, H = im.size
                scale = 640 / max(W, H)
                if scale < 1:
                    im = im.resize((int(W * scale), int(H * scale)))
                W, H = im.size
                dr = ImageDraw.Draw(im)
                lblp = Path(r["img"]).parent.parent / "labels" / r["file"]
                for ln in lblp.read_text(encoding="utf-8", errors="replace").splitlines():
                    p = ln.split()
                    if len(p) < 5:
                        continue
                    cid = int(p[0]); cx, cy, bw, bh = (float(x) for x in p[1:5])
                    x0 = (cx - bw / 2) * W; y0 = (cy - bh / 2) * H
                    x1 = (cx + bw / 2) * W; y1 = (cy + bh / 2) * H
                    col = [(255, 60, 60), (60, 160, 255), (60, 220, 120), (255, 220, 60)][cid % 4]
                    dr.rectangle([x0, y0, x1, y1], outline=col, width=2)
                    try:
                        dr.text((x0 + 2, max(0, y0 - 10)), str(cid), fill=col)
                    except Exception:  # 某些环境无默认字体，文字可省
                        pass
                out = args.outdir / "viz" / f"{r['split']}__{Path(r['img']).stem[:70]}.jpg"
                im.save(out, quality=88)
                made += 1
        except Exception as e:  # noqa: BLE001
            flags["viz_failed"] += 1
            if made == 0 and flags["viz_failed"] <= 3:
                flags[f"viz_err:{type(e).__name__}:{str(e)[:70]}"] += 1

    report = {
        "audited_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 仅供参考，判定须人工",
        "root": str(args.root),
        "split_stats": split_stats,
        "per_class_geometry": summary,
        "flag_counts": dict(flags.most_common()),
        "rows": len(rows),
        "flagged": len(queue),
        "viz_exported": made,
        "thresholds": {
            "MIN_AREA": MIN_AREA, "MIN_SIDE": MIN_SIDE, "EXTREME_AR": EXTREME_AR,
            "HUGE_AREA": HUGE_AREA, "MIN_EDGE_MARGIN": MIN_EDGE_MARGIN,
        },
        "disclaimer": "几何异常不等于标注错误；本文件只提供待人工判读的线索与可视化。",
    }
    (args.outdir / "audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"分 split：")
    for s, d in split_stats.items():
        print(f"  {s:<6} 图={d['images']:<5} 标签={d['labels']:<5} 框={d['boxes']:<6} 空标签={d['empty_label_images']}")
    print(f"\n逐类几何：")
    for k, v in summary.items():
        print(f"  {k:<18} 框={v['boxes']:<6} 面积中位={v['area_median']:<9} "
              f"面积p05={v['area_p05']:<9} 长宽比中位={v['aspect_median']:<7} 最大={v['aspect_max']}")
    print(f"\n异常标记统计：")
    for k, v in flags.most_common():
        print(f"  {k:<22} {v}")
    print(f"\n待人工复核：{len(queue)} 张；已导出可视化 {made} 张")
    print(f"输出：{args.outdir}")


if __name__ == "__main__":
    main()
