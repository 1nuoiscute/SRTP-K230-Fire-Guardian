"""v3 新增数据源审计（执行单 8.1）：Tugas Akhir v3 与 Kitchen Safety v3。

只读原目录；输出写到工作区新目录（拒绝覆盖）。

覆盖：
  A. 逐文件 SHA-256 清单
  B. data.yaml / README / 图签配对 / 标签范围 复核
  C. Kitchen Safety「840 vs 实际」差额说明
  D. 逐类框数、空标签图、图/标签 stem 是否一一对应
  E. 拍摄段推断（文件名时间戳/连续编号）
  F. 跨 split 的 SHA-256 与 dHash 近重复（含对 train）
  G. 与 GC v2、旧 fire-flame 的跨源 dHash 对照
  H. Tugas 水印检测（pngtree 等图库水印线索）
  I. 导出抽样带框可视化，供人工审图
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

WORK = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work")
SOURCES = {
    "tugas": Path(r"D:\Tugas Akhir.v3-3.0.yolov11"),
    "kitchen_safety": Path(r"D:\Kitchen Safety.v3i.yolov11"),
}
GC = Path(r"D:\gc_kitchen_annotation.v2-version_two.yolov11")
FF = Path(r"C:\Users\ASUS\Documents\Codex\kitchen-fire-detection\datasets"
          r"\fire-flame.v1i.yolov11")
IMG_EXT = {".jpg", ".jpeg", ".png"}

# 图库水印常见线索（OCR 不可用时用文件名与像素级特征兜底）
WATERMARK_HINTS = ("pngtree", "shutterstock", "istock", "getty", "dreamstime",
                   "vecteezy", "freepik", "123rf", "alamy", "adobe stock")


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dhash(p: Path, size: int = 8) -> int:
    with Image.open(p) as im:
        g = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        px = list(g.getdata())
    bits = 0
    idx = 0
    for r in range(size):
        for c in range(size):
            if px[r * (size + 1) + c] > px[r * (size + 1) + c + 1]:
                bits |= 1 << idx
            idx += 1
    return bits


def ham(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def read_label(p: Path):
    boxes = []
    if not p.exists():
        return boxes
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        f = ln.split()
        if len(f) < 5:
            continue
        try:
            if len(f) > 5:
                vals = [float(x) for x in f[1:]]
                xs, ys = vals[0::2], vals[1::2]
                cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                w, h = max(xs) - min(xs), max(ys) - min(ys)
            else:
                cx, cy, w, h = (float(x) for x in f[1:5])
            boxes.append((int(f[0]), cx, cy, w, h))
        except ValueError:
            continue
    return boxes


def seq_of(stem: str) -> str:
    """推断拍摄段：取文件名里最长的一串数字前缀前的部分 + 数字段地标。"""
    m = re.match(r"^([A-Za-z_\-]*?)(\d+)", stem)
    if m:
        return f"{m.group(1)}#{int(m.group(2)) // 50}"   # 50 张一档作为粗分桶
    return stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path,
                    default=WORK / "out" / "audit_v3_sources")
    ap.add_argument("--max-viz", type=int, default=60)
    args = ap.parse_args()
    if args.outdir.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{args.outdir}")
    (args.outdir / "viz").mkdir(parents=True)

    report: dict = {
        "audited_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 只读审计，须人工复核",
        "date": "2026-09-26",
        "sources": {},
        "cross_source": {},
    }
    rows: list[dict] = []

    # 预载对照指纹
    print("[0] 载入 GC 与 fire-flame 指纹用于跨源对照 ...")
    refs: dict[str, list[tuple[str, int]]] = {"gc": [], "ff": []}
    for tag, root in (("gc", GC), ("ff", FF)):
        n = 0
        for p in root.rglob("*"):
            if p.suffix.lower() in IMG_EXT:
                try:
                    refs[tag].append((str(p.relative_to(root)), dhash(p)))
                    n += 1
                except Exception:  # noqa: BLE001
                    pass
        print(f"    {tag}: {n} 张")

    for tag, root in SOURCES.items():
        print(f"\n[{tag}] {root}")
        info: dict = {"root": str(root), "splits": {}, "problems": []}
        if not root.exists():
            info["problems"].append("目录不存在")
            report["sources"][tag] = info
            continue

        # A/B. yaml + README
        yaml_p = root / "data.yaml"
        info["data_yaml"] = yaml_p.read_text(encoding="utf-8") if yaml_p.exists() else None
        info["readme"] = {}
        for rn in ("README.roboflow.txt", "README.dataset.txt"):
            rp = root / rn
            if rp.exists():
                info["readme"][rn] = rp.read_text(encoding="utf-8", errors="replace")

        # 图/标签配对 + 逐文件哈希 + 逐类统计
        per_split: dict[str, dict] = {}
        all_files: list[tuple[str, Path]] = []
        for split in ("train", "valid", "test"):
            idir, ldir = root / split / "images", root / split / "labels"
            if not idir.exists():
                continue
            imgs = sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT)
            lbls = {p.stem for p in ldir.glob("*.txt")} if ldir.exists() else set()
            cls = Counter()
            empty = 0
            no_lbl = 0
            bad_range = 0
            for img in imgs:
                all_files.append((split, img))
                lp = ldir / f"{img.stem}.txt"
                if img.stem not in lbls:
                    no_lbl += 1
                    continue
                boxes = read_label(lp)
                if not boxes:
                    empty += 1
                for c, cx, cy, w, h in boxes:
                    cls[c] += 1
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        bad_range += 1
            per_split[split] = {
                "images": len(imgs), "labels": len(lbls),
                "missing_label": no_lbl, "empty_label": empty,
                "boxes_by_class": dict(sorted(cls.items())),
                "coord_out_of_range": bad_range,
            }
            if len(imgs) != len(lbls):
                info["problems"].append(
                    f"{split}: 图 {len(imgs)} != 标签 {len(lbls)}")
        info["splits"] = per_split
        info["total_images"] = sum(v["images"] for v in per_split.values())

        # C. README 声明 vs 实际
        declared = None
        rope = info["readme"].get("README.roboflow.txt", "")
        m = re.search(r"The dataset includes ([\d,]+) images", rope)
        if m:
            declared = int(m.group(1).replace(",", ""))
        info["declared_images"] = declared
        info["discrepancy"] = (None if declared is None
                               else declared - info["total_images"])

        # D/E/F. 哈希 + 拍摄段 + 跨 split 近重复
        hashmap: dict[str, list[str]] = defaultdict(list)
        phash: list[tuple[str, str, int]] = []
        seqs: dict[str, set[str]] = defaultdict(set)
        for split, img in all_files:
            h = sha256(img)
            hashmap[h].append(f"{split}/{img.name}")
            try:
                phash.append((split, img.name, dhash(img)))
            except Exception:  # noqa: BLE001
                pass
            seqs[split].add(seq_of(img.stem))
        dup_sha = {h: v for h, v in hashmap.items() if len(v) > 1}
        cross_sha = {h: v for h, v in dup_sha.items()
                     if len({x.split("/")[0] for x in v}) > 1}
        tr = [(n, h) for s, n, h in phash if s == "train"]
        near = {}
        for ev in ("valid", "test"):
            hit4 = hit0 = 0
            for s, n, h in phash:
                if s != ev:
                    continue
                best = min((ham(h, th) for _, th in tr), default=99)
                if best <= 4:
                    hit4 += 1
                if best == 0:
                    hit0 += 1
            near[ev] = {"images": per_split.get(ev, {}).get("images", 0),
                        "dhash_le4_vs_train": hit4, "dhash_eq0_vs_train": hit0}
        info["dup_sha_within_source"] = len(dup_sha)
        info["cross_split_sha"] = len(cross_sha)
        info["near_dup_vs_train"] = near
        info["seq_buckets"] = {k: len(v) for k, v in seqs.items()}

        # G. 跨源对照
        for ref_tag, ref_list in refs.items():
            hit = 0
            for s, n, h in phash:
                if any(ham(h, rh) <= 4 for _, rh in ref_list):
                    hit += 1
            report["cross_source"][f"{tag}_vs_{ref_tag}"] = {
                "source_images": len(phash), "dhash_le4_matches": hit}

        # H. 水印线索
        wm_name = [n for _, n in [(s, n) for s, n, _ in phash]
                   if any(w in n.lower() for w in WATERMARK_HINTS)]
        info["watermark_name_hits"] = wm_name[:20]
        # 像素级：右下角高对比白色文字带常见于图库水印，做粗略线索统计
        wm_pixel = []
        for split, img in all_files:
            try:
                with Image.open(img) as im:
                    W, H = im.size
                    crop = im.convert("L").crop((int(W * 0.55), int(H * 0.80), W, H))
                    px = list(crop.getdata())
                hi = sum(1 for v in px if v > 235)
                if px and hi / len(px) > 0.02:      # 右下角 >2% 极亮像素
                    wm_pixel.append(f"{split}/{img.name}")
            except Exception:  # noqa: BLE001
                pass
        info["watermark_pixel_suspect"] = wm_pixel[:60]
        info["watermark_pixel_suspect_count"] = len(wm_pixel)

        # 逐图记录
        for split, img in all_files:
            lp = root / split / "labels" / f"{img.stem}.txt"
            boxes = read_label(lp)
            rows.append({
                "source": tag,
                "orig_path": str(img),
                "sha256": sha256(img),
                "orig_split": split,
                "orig_classes": "|".join(str(c) for c in sorted({c for c, *_ in boxes})),
                "n_boxes": len(boxes),
                "empty_label": int(not boxes),
                "seq_guess": seq_of(img.stem),
                "watermark_suspect": int(f"{split}/{img.name}" in wm_pixel),
                "license": "CC BY 4.0（项目页）",
                "review_verdict": "",
                "exclude_reason": "",
                "target_split": "",
            })

        # I. 抽样可视化（每类至少一张）
        made = 0
        by_class: dict[int, tuple[str, Path]] = {}
        for split, img in all_files:
            boxes = read_label(root / split / "labels" / f"{img.stem}.txt")
            for c, *_ in boxes:
                by_class.setdefault(c, (split, img))
        picks = list(by_class.values())[: args.max_viz // 4 + 1]
        picks += [all_files[i] for i in range(0, len(all_files),
                                            max(1, len(all_files) // 20))][: args.max_viz]
        seen = set()
        for split, img in picks:
            if img in seen or made >= args.max_viz:
                continue
            seen.add(img)
            try:
                with Image.open(img) as im:
                    im = im.convert("RGB")
                    im.thumbnail((800, 800))
                    W, H = im.size
                    dr = ImageDraw.Draw(im)
                    for c, cx, cy, w, h in read_label(
                            root / split / "labels" / f"{img.stem}.txt"):
                        x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
                        x1, y1 = (cx + w / 2) * W, (cy + h / 2) * H
                        col = [(255, 70, 70), (70, 160, 255), (70, 220, 120),
                               (255, 220, 70), (220, 120, 255)][c % 5]
                        dr.rectangle([x0, y0, x1, y1], outline=col, width=3)
                        try:
                            dr.text((x0 + 3, max(0, y0 - 12)), str(c), fill=col)
                        except Exception:  # noqa: BLE001
                            pass
                im.save(args.outdir / "viz" / f"{tag}__{split}__{img.stem[:60]}.jpg",
                        quality=88)
                made += 1
            except Exception:  # noqa: BLE001
                pass
        info["viz_exported"] = made
        report["sources"][tag] = info

    with (args.outdir / "source_manifest_v3.csv").open("w", encoding="utf-8",
                                                        newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (args.outdir / "audit_v3_sources.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台摘要
    print("\n" + "=" * 78)
    for tag, info in report["sources"].items():
        print(f"\n【{tag}】 总图 {info.get('total_images')}  "
              f"README 声明 {info.get('declared_images')}  "
              f"差额 {info.get('discrepancy')}")
        for s, d in info.get("splits", {}).items():
            print(f"    {s:<6} 图={d['images']:<5} 缺标签={d['missing_label']:<3} "
                  f"空标签={d['empty_label']:<4} 越界={d['coord_out_of_range']:<3} "
                  f"框={d['boxes_by_class']}")
        print(f"    跨split同SHA256: {info.get('cross_split_sha')}   "
              f"源内重复: {info.get('dup_sha_within_source')}")
        print(f"    近重复 vs train: {info.get('near_dup_vs_train')}")
        print(f"    拍摄段粗分桶: {info.get('seq_buckets')}")
        print(f"    水印 文件名命中={len(info.get('watermark_name_hits', []))} "
              f"像素可疑={info.get('watermark_pixel_suspect_count')}")
        print(f"    可视化导出 {info.get('viz_exported')} 张")
        if info.get("problems"):
            print(f"    ⚠ 问题: {info['problems']}")
    print("\n跨源对照:", json.dumps(report["cross_source"], ensure_ascii=False))
    print(f"\n输出：{args.outdir}")


if __name__ == "__main__":
    main()
