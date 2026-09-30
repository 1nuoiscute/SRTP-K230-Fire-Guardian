"""构建「冻结回归评测套件」test_suite_v1。

背景（2026-09-25 只读审计结论，见 out/audit/）：
  - kitchen-images 的 64 张 100% 在 combined/images/train 内（前缀 manual_*）→ 完全泄漏
  - fire-flame / stove / pan 的 test 分片被 prepare_data.py:26 映射进 combined 的 val → 泄漏
  - D:\\Codes\\flame-detection\\dataset_yolo 的 239 张与 combined 零重叠 → **干净**
因此本套件按来源打标签，绝不假装某一部分是"独立测试集"。

统一类别空间（与旧模型一致，保证可比）：0 fire / 1 smoke / 2 stove / 3 pan

来源与映射：
  A. fire-flame test       : class 0 -> 0 (fire)            [in_combined=val]
  B. stove test            : class 0 -> 2 (stove)           [in_combined=val]
  C. pan test              : class 1 -> 3 (pan)，class 0 丢弃 [in_combined=val]
  D. kitchen-images        : 已是 0/2/3 → 原样              [in_combined=train，完全泄漏]
  E. flame-detection CCTV  : class 0 -> 0 (fire), 1 -> 1 (smoke) [clean]
     (该源原 data.yaml 的 names 与标签不一致：names=[fire,smoke] 但 dataset2/README 提到 [smoke,fire]。
      本脚本按 data.yaml 的 nc=2 names=[fire,smoke] 解释，即 0=fire、1=smoke。此假设已在审计中标注。)

只写新目录；不修改、不移动任何既有文件。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

KF = Path(r"C:\Users\ASUS\Documents\Codex\kitchen-fire-detection")
FD = Path(r"D:\Codes\flame-detection")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = ["fire", "smoke", "stove", "pan"]


def md5(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def list_images(root: Path) -> list[Path]:
    out = []
    for dirpath, _d, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in IMG_EXT:
                out.append(Path(dirpath) / fn)
    return sorted(out)


def poly_to_bbox(parts: list[str]) -> list[str]:
    cls = parts[0]
    vals = [float(v) for v in parts[1:]]
    xs, ys = vals[0::2], vals[1::2]
    return [
        cls,
        f"{(min(xs) + max(xs)) / 2:.6f}",
        f"{(min(ys) + max(ys)) / 2:.6f}",
        f"{max(xs) - min(xs):.6f}",
        f"{max(ys) - min(ys):.6f}",
    ]


def resolve_label(img: Path) -> Path | None:
    """找图片对应的 YOLO 标签。

    兼容三种布局：
      1. 同目录同名 .txt
      2. Roboflow 的 .../images/x.jpg + .../labels/x.txt
      3. flame-detection 的 dataset_yolo/images/... + dataset_yolo/labels/...
    """
    same = img.with_suffix(".txt")
    if same.exists():
        return same
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            cand = Path(*parts[:i], "labels", *parts[i + 1 :]).with_suffix(".txt")
            if cand.exists():
                return cand
            break
    if "dataset_yolo" in parts:
        i = parts.index("dataset_yolo")
        cand = Path(*parts[: i + 1], "labels", *parts[i + 2 :]).with_suffix(".txt")
        if cand.exists():
            return cand
    return None


def convert_label(lbl: Path, class_map: dict[int, int]) -> list[str]:
    lines = []
    for line in lbl.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        if len(parts) > 5:
            parts = poly_to_bbox(parts)
        cls = int(parts[0])
        if cls not in class_map:
            continue
        parts[0] = str(class_map[cls])
        lines.append(" ".join(parts))
    return lines


# (tag, 根目录, split 子目录, class_map, 是否泄漏进 combined)
# 注意：Roboflow 导出的结构是 <split>/images/ 与 <split>/labels/，标签与图片同 stem 但不同目录。
SOURCES: list[tuple[str, Path, str | None, dict[int, int], str]] = [
    ("fireflame_test", KF / "datasets" / "fire-flame.v1i.yolov11", "test/images", {0: 0}, "val"),
    ("stove_test", KF / "datasets" / "stove-detection", "test/images", {0: 2}, "val"),
    ("pan_test", KF / "datasets" / "pan-detection", "test/images", {1: 3}, "val"),
    ("kitchen_images", KF / "datasets" / "kitchen-images", None, {0: 0, 2: 2, 3: 3}, "train"),
    ("cctv_clean", FD / "dataset_yolo" / "images", None, {0: 0, 1: 1}, "clean"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dest",
        type=Path,
        default=Path(
            r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\data\test_suite_v1"
        ),
    )
    args = ap.parse_args()
    dest: Path = args.dest

    if dest.exists():
        raise SystemExit(f"目标已存在，拒绝覆盖：{dest}")
    (dest / "images").mkdir(parents=True)
    (dest / "labels").mkdir(parents=True)

    manifest: list[dict] = []
    per_source: dict[str, dict] = {}

    for tag, root, sub, cmap, leakage in SOURCES:
        base = root if sub is None else root / sub
        if not base.exists():
            print(f"  ! 跳过不存在的来源 {tag}: {base}")
            continue
        imgs = list_images(base)
        kept = 0
        for img in imgs:
            lbl = resolve_label(img)
            if lbl is None:
                continue
            lines = convert_label(lbl, cmap)
            out_name = f"{tag}__{img.stem}"
            shutil.copy2(img, dest / "images" / f"{out_name}{img.suffix.lower()}")
            (dest / "labels" / f"{out_name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            counts: dict[int, int] = defaultdict(int)
            for ln in lines:
                counts[int(ln.split()[0])] += 1
            manifest.append(
                {
                    "file": f"{out_name}{img.suffix.lower()}",
                    "source": tag,
                    "src": str(img),
                    "leaked": leakage,
                    "md5": md5(img),
                    "boxes": {str(k): v for k, v in sorted(counts.items())},
                    "empty": len(lines) == 0,
                }
            )
            kept += 1
        per_source[tag] = {"dir": str(base), "candidates": len(imgs), "kept": kept, "leakage": leakage}
        print(f"  {tag:<16} 候选 {len(imgs):>5}  收进 {kept:>5}  泄漏级别={leakage}")

    if not manifest:
        raise SystemExit("未收集到任何图片。")

    # 统计
    boxes: dict[int, int] = defaultdict(int)
    empty = 0
    clean_n = 0
    for m in manifest:
        for k, v in m["boxes"].items():
            boxes[int(k)] += v
        if m["empty"]:
            empty += 1
        if m["leaked"] == "clean":
            clean_n += 1

    print()
    print("=" * 72)
    print(f"test_suite_v1：{len(manifest)} 张图，{empty} 张空标签")
    print(f"  其中泄漏级别 clean（真正无重叠）：{clean_n} 张")
    print(f"  泄漏级别 val（图像在 combined/val）：{len(manifest) - clean_n - sum(1 for m in manifest if m['leaked'] == 'train')} 张")
    print(f"  泄漏级别 train（图像在 combined/train，完全泄漏）：{sum(1 for m in manifest if m['leaked'] == 'train')} 张")
    for i, n in enumerate(NAMES):
        print(f"    {n:<6} 框数 = {boxes.get(i, 0)}")
    print("=" * 72)

    (dest / "data.yaml").write_text(
        f"path: {str(dest).replace(chr(92), '/')}\n"
        "train: images\nval: images\ntest: images\nnc: 4\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)),
        encoding="utf-8",
    )

    with (dest / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["file", "source", "leaked", "empty", "md5", "boxes", "src"])
        for m in manifest:
            w.writerow(
                [
                    m["file"],
                    m["source"],
                    m["leaked"],
                    int(m["empty"]),
                    m["md5"],
                    json.dumps(m["boxes"], sort_keys=True),
                    m["src"],
                ]
            )

    audit = {
        "frozen_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 仅供参考，需人工复核",
        "date": "2026-09-25",
        "dest": str(dest),
        "total": len(manifest),
        "clean_images": clean_n,
        "empty_label_images": empty,
        "boxes_per_class": {NAMES[i]: boxes.get(i, 0) for i in range(4)},
        "per_source": per_source,
        "leakage_levels": {
            "clean": "MD5 与 combined 无任何重叠 —— 唯一可作独立评测的部分",
            "val": "图像已在 combined/images/val —— 可用于相对比较，非独立",
            "train": "图像已在 combined/images/train —— 完全泄漏，仅作回归追踪",
        },
        "critical_findings": [
            "kitchen-images 64 张被 prepare_data.py 全部复制进 combined/images/train（前缀 manual_*），"
            "100% 泄漏。results/ 下那套按场景分类的回归集就是这批图，不能当独立测试集。",
            "fire-flame/stove/pan 的 test 分片被 prepare_data.py:26 的 ('test','val') 映射进 val，"
            "共 1064 张候选全部已在 combined 中。",
            "combined 的 17431 张图只有 10222 个不同 MD5 —— 过采样 _dN 复制造成大量内容重复。",
            "smoke 类在 combined 中始终为 0 正例；本套件的 smoke 正例全部来自 cctv_clean（另一个域）。",
            "【未证实】cctv_clean 的类别语义假设为 data.yaml 的 0=fire、1=smoke；"
            "该源的 dataset2/README.md 提到 [smoke,fire]，两者矛盾，需人工看图确认。",
        ],
    }
    (dest / "MANIFEST_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已写入：{dest}")
    print("  data.yaml / manifest.csv / MANIFEST_AUDIT.json / images/ / labels/")


if __name__ == "__main__":
    main()
