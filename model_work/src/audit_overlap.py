"""只读审计：regression 集（kitchen-images）与 combined 训练集的重叠情况。

用途：判断 results/ 下那 64 张按场景分类的回归集能否作为「独立测试集」。
只读：不写入任何被检查的目录；结果写到 stdout 与 --out 指定的新文件。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

PROJ = Path(r"C:\Users\ASUS\Documents\Codex\kitchen-fire-detection")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in IMG_EXT:
                out.append(Path(dirpath) / fn)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None, help="写入审计 JSON 的新文件")
    args = ap.parse_args()

    # 1. 回归集（带真值标注的原始目录）
    reg_root = PROJ / "datasets" / "kitchen-images"
    reg = list_images(reg_root)
    # 排除 datasets.lnk 之类的非图片（list_images 已按扩展名过滤）

    print(f"[1] 回归集 kitchen-images: {len(reg)} 张图")

    # 2. 训练集 combined
    comb_root = PROJ / "datasets" / "combined"
    comb = list_images(comb_root)
    print(f"[2] combined: {len(comb)} 张图（train+val）")

    # 3. 逐源数据集（用于定位重叠来源）
    src_roots = {
        "fire-flame": PROJ / "datasets" / "fire-flame.v1i.yolov11",
        "stove-detection": PROJ / "datasets" / "stove-detection",
        "pan-detection": PROJ / "datasets" / "pan-detection",
    }

    print("[3] 计算回归集内容哈希 ...")
    reg_hash: dict[str, list[Path]] = defaultdict(list)
    for p in reg:
        reg_hash[md5(p)].append(p)

    print("[4] 计算 combined 内容哈希 ...")
    comb_hash: dict[str, list[Path]] = defaultdict(list)
    for p in comb:
        comb_hash[md5(p)].append(p)

    print("[5] 计算各源数据集内容哈希 ...")
    src_hash: dict[str, dict[str, list[Path]]] = {}
    for name, root in src_roots.items():
        d: dict[str, list[Path]] = defaultdict(list)
        if root.exists():
            for p in list_images(root):
                d[md5(p)].append(p)
        src_hash[name] = d
        print(f"    {name}: {sum(len(v) for v in d.values())} 张")

    # 6. 重叠分析
    leaked_exact: dict[str, dict] = {}
    for h, paths in reg_hash.items():
        if h in comb_hash:
            leaked_exact[str(paths[0].relative_to(reg_root))] = {
                "scene": paths[0].parent.name,
                "combined_matches": [
                    str(q.relative_to(comb_root)) for q in comb_hash[h]
                ],
                "source_matches": {
                    name: [str(q.relative_to(src_roots[name])) for q in d[h]]
                    for name, d in src_hash.items()
                    if h in d
                },
            }

    total_reg = len(reg)
    leaked = len(leaked_exact)
    print()
    print("=" * 68)
    print(f"精确内容重叠（MD5 相同）：{leaked} / {total_reg} 张")
    print("=" * 68)

    by_scene: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in reg:
        s = p.parent.name
        by_scene[s][1] += 1
    for rel, info in leaked_exact.items():
        by_scene[info["scene"]][0] += 1

    print(f"{'场景':<18}{'重叠':>6}{'总数':>6}   可作独立评测")
    for scene in sorted(by_scene):
        lk, tot = by_scene[scene]
        print(f"{scene:<18}{lk:>6}{tot:>6}   {tot - lk} 张")

    if leaked_exact:
        print()
        print("重叠明细（前 10 条）：")
        for rel, info in list(leaked_exact.items())[:10]:
            print(f"  {rel}")
            print(f"     -> combined: {info['combined_matches'][:2]}")
            if info["source_matches"]:
                for name, m in info["source_matches"].items():
                    print(f"     -> 源 {name}: {m[:2]}")

    report = {
        "regression_root": str(reg_root),
        "regression_total": total_reg,
        "combined_total": len(comb),
        "exact_leaked": leaked,
        "exact_leaked_detail": leaked_exact,
        "per_scene": {
            s: {"leaked": v[0], "total": v[1], "clean": v[1] - v[0]}
            for s, v in sorted(by_scene.items())
        },
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print()
        print(f"审计结果已写入：{args.out}")


if __name__ == "__main__":
    main()
