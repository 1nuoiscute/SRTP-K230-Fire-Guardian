"""按「视觉重复族」重新切分，彻底消除跨 split 的近重复泄漏。

为什么需要这一步
----------------
1) Roboflow 在导出时会给**同一张源照片**派生多个 `.rf.<hash>` 身份。
   实测：DSC_4779 / DSC_4793 / DSC_4800 / DSC_4801 / DSC_4808 …
   与 DSC_4780 的 dHash 完全相同（距离 0），但文件名与 MD5 都不同。
   因此按 `.rf.` 前缀分组**抓不全**同源，遗留 512 处跨 split 近重复（其中 212 处 d=0）。
2) 同一拍摄段（如 DSC_#### 连拍）的相邻帧几乎必然相似，对视频/连拍数据集
   无法靠阈值区分「相邻帧」与「同图派生」。最稳妥的做法是
   **把互为近重复的图当作一个不可分割的族，整体只进一个 split**。

做法
----
  1. 对全数据集算 64-bit dHash。
  2. 以「距离 <= 阈值」建图，取连通分量作为视觉重复族（union-find）。
  3. 每个族确定唯一目标 split：
       - 含 gc_kitchen_v2 的族 -> 跟随 GC 的源 split（train/valid/test），
         以保住目标域在 val/test 的代表性；
       - 其余族 -> 多数决（平票时按哈希确定性选择）。
  4. 物理移动 images/labels 到目标 split，重写 manifest 的 split 列。
  5. 复算并输出核实报告。

就地修改数据集；不删除任何文件，只移动。判定阈值默认 4。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

D_DEFAULT = Path(r"D:\SRTP_Datasets\kitchen_vision_v2_20260926")


def dhash(p: Path, size: int = 8) -> int:
    with Image.open(p) as im:
        g = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        px = list(g.getdata())
    bits = 0
    idx = 0
    for row in range(size):
        for col in range(size):
            if px[row * (size + 1) + col] > px[row * (size + 1) + col + 1]:
                bits |= 1 << idx
            idx += 1
    return bits


def ham(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=D_DEFAULT,
                    help="数据集目录（须含 manifest.csv）")
    ap.add_argument("--threshold", type=int, default=4)
    ap.add_argument("--apply", action="store_true", help="真正执行移动；缺省只预演")
    args = ap.parse_args()
    D = args.data
    if not (D / "manifest.csv").exists():
        raise SystemExit(f"未找到 manifest.csv：{D}")

    rows = list(csv.DictReader((D / "manifest.csv").open(encoding="utf-8")))
    fields = list(rows[0].keys())
    print(f"manifest {len(rows)} 行；阈值 dHash <= {args.threshold}；"
          f"模式={'执行' if args.apply else '预演'}")

    # 1. 计算哈希
    print("计算 dHash ...")
    hashes: list[int] = []
    for i, r in enumerate(rows):
        p = D / "images" / r["split"] / r["out_file"]
        try:
            hashes.append(dhash(p))
        except Exception:  # noqa: BLE001
            hashes.append(-1)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(rows)}")

    # 2. 近重复族（union-find）
    print("聚类近重复族 ...")
    dsu = DSU(len(rows))
    idx_by_hash: dict[int, list[int]] = defaultdict(list)
    for i, h in enumerate(hashes):
        if h >= 0:
            idx_by_hash[h].append(i)
    # 先按完全相同的哈希快速合并
    for h, ids in idx_by_hash.items():
        for k in range(1, len(ids)):
            dsu.union(ids[0], ids[k])
    # 再对每个代表做阈值内比较（用族代表，避免 O(N^2)）
    reps: list[int] = []
    seen: set[int] = set()
    for i in range(len(rows)):
        r = dsu.find(i)
        if r not in seen:
            seen.add(r)
            reps.append(i)
    print(f"  完全同哈希族：{len(idx_by_hash)}；代表点 {len(reps)}")
    for a in range(len(reps)):
        ia = reps[a]
        if hashes[ia] < 0:
            continue
        for b in range(a + 1, len(reps)):
            ib = reps[b]
            if hashes[ib] < 0:
                continue
            if ham(hashes[ia], hashes[ib]) <= args.threshold:
                dsu.union(ia, ib)

    fams: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        fams[dsu.find(i)].append(i)
    multi = {k: v for k, v in fams.items() if len(v) > 1}
    print(f"  视觉重复族总数 {len(fams)}；其中多成员族 {len(multi)}")

    # 3. 每个族决定唯一目标 split
    def target_for(ids: list[int]) -> str:
        gc = [rows[i] for i in ids if rows[i]["source"] == "gc_kitchen_v2"]
        if gc:
            # 跟随 GC 源 split，保住目标域在 val/test 的代表性
            ss = {r["source_split"] for r in gc}
            for pref, tgt in (("train", "train"), ("valid", "val"), ("test", "test")):
                if pref in ss:
                    return tgt
            return "train"
        c = Counter(rows[i]["split"] for i in ids)
        if len(c) == 1:
            return next(iter(c))
        top = max(c.values())
        tied = sorted(k for k, v in c.items() if v == top)
        if len(tied) == 1:
            return tied[0]
        # 平票：按族内文件名哈希确定性选择
        key = hashlib.sha256("".join(sorted(rows[i]["out_file"] for i in ids)).encode()).hexdigest()
        return tied[int(key, 16) % len(tied)]

    moves = 0
    changed_fams = 0
    for k, ids in fams.items():
        if len(ids) == 1:
            continue
        cur = {rows[i]["split"] for i in ids}
        if len(cur) == 1:
            continue
        tgt = target_for(ids)
        changed_fams += 1
        for i in ids:
            if rows[i]["split"] != tgt:
                moves += 1
    print(f"  需要改 split 的族：{changed_fams}；需移动的图：{moves}")

    if not args.apply:
        print("\n（预演模式，未做任何改动。加 --apply 执行。）")
        return

    # 4. 执行移动
    print("执行移动 ...")
    moved = 0
    for k, ids in fams.items():
        if len(ids) == 1:
            continue
        cur = {rows[i]["split"] for i in ids}
        if len(cur) == 1:
            continue
        tgt = target_for(ids)
        for i in ids:
            r = rows[i]
            if r["split"] == tgt:
                continue
            src_img = D / "images" / r["split"] / r["out_file"]
            src_lbl = D / "labels" / r["split"] / (Path(r["out_file"]).stem + ".txt")
            dst_img = D / "images" / tgt / r["out_file"]
            dst_lbl = D / "labels" / tgt / (Path(r["out_file"]).stem + ".txt")
            if dst_img.exists():
                stem = Path(r["out_file"]).stem
                suf = Path(r["out_file"]).suffix
                k2 = 1
                while (D / "images" / tgt / f"{stem}_{k2}{suf}").exists():
                    k2 += 1
                dst_img = D / "images" / tgt / f"{stem}_{k2}{suf}"
                dst_lbl = D / "labels" / tgt / f"{stem}_{k2}.txt"
                r["out_file"] = dst_img.name
            if src_img.exists():
                shutil.move(str(src_img), str(dst_img))
            if src_lbl.exists():
                shutil.move(str(src_lbl), str(dst_lbl))
            r["split"] = tgt
            moved += 1
    print(f"  实际移动 {moved} 张")

    with (D / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # 5. 复核
    print("复核 ...")
    h2: list[int] = []
    for r in rows:
        p = D / "images" / r["split"] / r["out_file"]
        try:
            h2.append(dhash(p))
        except Exception:  # noqa: BLE001
            h2.append(-1)
    tr = [i for i, r in enumerate(rows) if r["split"] == "train"]
    ev = [i for i, r in enumerate(rows) if r["split"] in ("val", "test")]
    remain = 0
    for i in ev:
        if h2[i] < 0:
            continue
        for j in tr:
            if h2[j] < 0:
                continue
            if ham(h2[i], h2[j]) <= args.threshold:
                remain += 1
                break
    grp_splits: dict[str, set[str]] = defaultdict(set)
    vers: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        gk = f"{r['source']}::{r['group_id']}"
        grp_splits[gk].add(r["split"])
        vers[gk][r["split"]] += 1
    summary = {}
    for s in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == s]
        summary[s] = {"images": len(rs),
                      "empty": sum(int(r["empty_label_post"]) for r in rs),
                      "groups": len({f"{r['source']}::{r['group_id']}" for r in rs})}
    out = {
        "fixed_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 仅供参考",
        "method": "按 dHash 视觉重复族（union-find, 阈值 %d）整体重分配 split" % args.threshold,
        "families_total": len(fams),
        "multi_member_families": len(multi),
        "families_reassigned": changed_fams,
        "images_moved": moved,
        "remaining_near_dup_eval_vs_train": remain,
        "remaining_cross_split_group_leaks": sum(1 for v in grp_splits.values() if len(v) > 1),
        "eval_multi_version_groups": sum(
            1 for v in vers.values() if any(v.get(s, 0) > 1 for s in ("val", "test"))),
        "per_split_after": summary,
    }
    (D / "manifest_visual_family_fix.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 74)
    for s, d in summary.items():
        print(f"  {s:<6} 图={d['images']:<6} 空标签={d['empty']:<5} 源组={d['groups']}")
    print(f"  残留 近重复(val/test vs train) : {remain}")
    print(f"  残留 源组跨 split             : {out['remaining_cross_split_group_leaks']}")
    print(f"  残留 val/test 多版本组        : {out['eval_multi_version_groups']}")
    print("=" * 74)


if __name__ == "__main__":
    main()
