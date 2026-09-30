"""给 v3 manifest 补一列 GC 原始子类，用于严格的「锅下火 / 燃气灶火」分层。

背景：本项目的四槽映射把 GC 的 `1 flame-under-pot` 与 `2 gas-stove-flame`
都并入 `0 fire`，因此 manifest 里丢失了这个子类区分，导致错例报告无法
把「锅下火漏检」与「裸露灶火漏检」自动分开。

做法：只**新增列**，不动任何图片/标签，也不改 data.yaml。
      训练进程只读 data.yaml 与 images/labels，不读 manifest.csv，
      且 label cache 已生成，因此本操作对正在运行的训练无影响。

新增列：
  gc_orig_subclasses  该图在原 GC 标签里出现的原始类名（如 flame-under-pot）
  gc_boxes_flame_under_pot / gc_boxes_gas_stove_flame / gc_boxes_flame_reflection / gc_boxes_no_flame
  fire_subclass_tag   用于报告分层的标签：
                        "under_pot"      仅含锅下火
                        "gas_stove"      仅含裸露灶火
                        "both"           两者都有
                        "reflection_only" GC 图但只有反光（本构建映射后为 0 张）
                        "(non-gc)"       非 GC 来源
原文件先备份为 manifest.before_subclass.csv。
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

GC_NAMES = {0: "flame-reflection", 1: "flame-under-pot",
            2: "gas-stove-flame", 3: "no-flame"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path(r"D:\SRTP_Datasets\kitchen_vision_v3_20260926"))
    args = ap.parse_args()
    D = args.data
    mf = D / "manifest.csv"
    if not mf.exists():
        raise SystemExit(f"未找到 {mf}")
    bak = D / "manifest.before_subclass.csv"
    if not bak.exists():
        shutil.copy2(mf, bak)
        print(f"已备份原 manifest -> {bak.name}")
    else:
        print(f"备份已存在，沿用：{bak.name}")

    rows = list(csv.DictReader(bak.open(encoding="utf-8")))
    fields = list(rows[0].keys())
    new_cols = ["gc_orig_subclasses", "gc_boxes_flame_under_pot",
                "gc_boxes_gas_stove_flame", "gc_boxes_flame_reflection",
                "gc_boxes_no_flame", "fire_subclass_tag"]
    for c in new_cols:
        if c not in fields:
            fields.append(c)

    tag_count = Counter()
    for r in rows:
        sub = Counter()
        if r["source"] == "gc_kitchen_v2" and r["source_path"]:
            lp = Path(r["source_path"]).parent.parent / "labels" / \
                 (Path(r["source_path"]).stem + ".txt")
            if lp.exists():
                for ln in lp.read_text(encoding="utf-8", errors="replace").splitlines():
                    f = ln.split()
                    if len(f) >= 5:
                        try:
                            sub[int(f[0])] += 1
                        except ValueError:
                            pass
        if sub:
            r["gc_orig_subclasses"] = "|".join(GC_NAMES[k] for k in sorted(sub))
            r["gc_boxes_flame_under_pot"] = sub.get(1, 0)
            r["gc_boxes_gas_stove_flame"] = sub.get(2, 0)
            r["gc_boxes_flame_reflection"] = sub.get(0, 0)
            r["gc_boxes_no_flame"] = sub.get(3, 0)
            has_up, has_gs = sub.get(1, 0) > 0, sub.get(2, 0) > 0
            tag = ("both" if has_up and has_gs else
                   "under_pot" if has_up else
                   "gas_stove" if has_gs else "reflection_only")
        else:
            r["gc_orig_subclasses"] = ""
            r["gc_boxes_flame_under_pot"] = 0
            r["gc_boxes_gas_stove_flame"] = 0
            r["gc_boxes_flame_reflection"] = 0
            r["gc_boxes_no_flame"] = 0
            tag = "(non-gc)"
        r["fire_subclass_tag"] = tag
        tag_count[tag] += 1

    with mf.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\n已写回 {mf.name}（新增 {len(new_cols)} 列）")
    print("fire_subclass_tag 分布：")
    for k, v in tag_count.most_common():
        print(f"  {k:<18} {v:>5}")

    # 分层 × split 交叉表，供报告直接引用
    cross: dict[tuple[str, str], int] = Counter()
    for r in rows:
        cross[(r["fire_subclass_tag"], r["split"])] += 1
    print("\ntag × split：")
    splits = ("train", "val", "test")
    print(f"  {'tag':<18}" + "".join(f"{s:>8}" for s in splits))
    for tag in sorted({t for t, _ in cross}):
        print(f"  {tag:<18}" + "".join(f"{cross.get((tag, s), 0):>8}" for s in splits))


if __name__ == "__main__":
    main()
