"""只读：找出磁盘上所有「从未进入 combined」的火焰/厨房相关图像（潜在干净数据）。

判据：内容 MD5 不在 combined 中，也不是 combined 的近重复。
近重复用 64-bit dHash（汉明距离 <= 6）粗筛，用于发现重压缩/缩放副本。

结果只打印与写入 --out 指定的新文件；不修改任何既有文件。
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

# 扫描这些根目录，找可能的干净数据
SCAN_ROOTS = [
    PROJ / "datasets",
    PROJ / "results",
    Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\artifacts"),
    Path(r"D:\Codes\flame-detection"),
]


def md5(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dhash(p: Path, size: int = 8) -> int:
    """极简 dHash：不依赖 PIL，用自带解码。失败返回 -1。"""
    try:
        from PIL import Image
    except Exception:
        return -1
    try:
        with Image.open(p) as im:
            g = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
            px = list(g.getdata())
    except Exception:
        return -1
    bits = 0
    idx = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            if left > right:
                bits |= 1 << idx
            idx += 1
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def list_images(root: Path):
    if not root.exists():
        return
    for dirpath, _d, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in IMG_EXT:
                yield Path(dirpath) / fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--hamming", type=int, default=6)
    args = ap.parse_args()

    comb_root = PROJ / "datasets" / "combined"
    print("[1] 建立 combined 指纹库 ...")
    comb_md5: set[str] = set()
    comb_dhash: list[tuple[int, str]] = []
    n_comb = 0
    for p in list_images(comb_root):
        n_comb += 1
        if n_comb % 4000 == 0:
            print(f"    {n_comb} ...")
        try:
            comb_md5.add(md5(p))
        except OSError:
            continue
        h = dhash(p)
        if h >= 0:
            comb_dhash.append((h, str(p.relative_to(comb_root))))
    print(f"    combined 指纹：{len(comb_md5)} 个 MD5，{len(comb_dhash)} 个 dHash")

    print("[2] 扫描候选根目录 ...")
    results = []
    per_root: dict[str, dict[str, int]] = defaultdict(
        lambda: {"scanned": 0, "clean_exact": 0, "clean_near": 0}
    )
    for root in SCAN_ROOTS:
        if not root.exists():
            print(f"    (跳过不存在的) {root}")
            continue
        for p in list_images(root):
            # 跳过 combined 自身
            try:
                p.relative_to(comb_root)
                continue
            except ValueError:
                pass
            st = per_root[str(root)]
            st["scanned"] += 1
            try:
                m = md5(p)
            except OSError:
                continue
            is_exact = m in comb_md5
            h = dhash(p)
            near = False
            if not is_exact and h >= 0:
                for hh, _rel in comb_dhash:
                    if hamming(h, hh) <= args.hamming:
                        near = True
                        break
            if is_exact:
                continue
            if near:
                st["clean_near"] += 1
                results.append(
                    {"path": str(p), "md5": m, "near_dup": True, "root": str(root)}
                )
            else:
                st["clean_exact"] += 1
                results.append(
                    {"path": str(p), "md5": m, "near_dup": False, "root": str(root)}
                )

    print()
    print("=" * 72)
    for root, st in per_root.items():
        print(
            f"{root}\n    扫描 {st['scanned']:>6}  精确干净 {st['clean_exact']:>6}  "
            f"近重复 {st['clean_near']:>6}"
        )
    print("=" * 72)

    truly_clean = [r for r in results if not r["near_dup"]]
    print(f"\n与 combined 既非精确重复、也非近重复（dHash<= {args.hamming}）的图：{len(truly_clean)} 张")
    for r in truly_clean[:40]:
        print("   ", r["path"])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "scanned_roots": [str(r) for r in SCAN_ROOTS],
                    "combined_images": n_comb,
                    "per_root": dict(per_root),
                    "clean_exact_count": sum(1 for r in results if not r["near_dup"]),
                    "clean_near_count": sum(1 for r in results if r["near_dup"]),
                    "clean_exact": truly_clean,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n已写入：{args.out}")


if __name__ == "__main__":
    main()
