"""修复已构建数据集中的一个验收不合规项：val/test 内存在同源组多版本。

背景：stove-detection 的 24 个源图组在 val/test 里各有多个增强版本，
违反执行单 3.3「val/test 无增强复制」。修法：把这些组整体移回 train，
保证 val/test 里每个源组只出现一次。

就地修改数据集（只动 images/labels 的 split 目录 + manifest + report）。
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

D = Path(r"D:\SRTP_Datasets\kitchen_vision_v2_20260926")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path(r"D:\SRTP_Datasets\kitchen_vision_v2_20260926"))
    args = ap.parse_args()
    D = args.data
    if not (D / "manifest.csv").exists():
        raise SystemExit(f"未找到 manifest.csv：{D}")

    rows = list(csv.DictReader((D / "manifest.csv").open(encoding="utf-8")))
    fields = list(rows[0].keys())

    vers: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        vers[(r["source"], r["group_id"])][r["split"]] += 1
    bad = {k for k, v in vers.items() if any(v.get(s, 0) > 1 for s in ("val", "test"))}
    print(f"val/test 内多版本组：{len(bad)} 个 -> 整体移回 train")

    moved = 0
    for r in rows:
        if (r["source"], r["group_id"]) in bad and r["split"] in ("val", "test"):
            old_img = D / "images" / r["split"] / r["out_file"]
            old_lbl = D / "labels" / r["split"] / (Path(r["out_file"]).stem + ".txt")
            new_img = D / "images" / "train" / r["out_file"]
            new_lbl = D / "labels" / "train" / (Path(r["out_file"]).stem + ".txt")
            if old_img.exists():
                old_img.replace(new_img)
            if old_lbl.exists():
                old_lbl.replace(new_lbl)
            r["split"] = "train"
            moved += 1
    print(f"实际移动 {moved} 张")

    with (D / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # 复核
    vers2: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    grp_splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        vers2[(r["source"], r["group_id"])][r["split"]] += 1
        grp_splits[f"{r['source']}::{r['group_id']}"].add(r["split"])
    still = {k for k, v in vers2.items() if any(v.get(s, 0) > 1 for s in ("val", "test"))}
    leaks = {k: sorted(v) for k, v in grp_splits.items() if len(v) > 1}

    summary = {}
    for s in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == s]
        summary[s] = {"images": len(rs),
                      "empty": sum(int(r["empty_label_post"]) for r in rs),
                      "groups": len({f"{r['source']}::{r['group_id']}" for r in rs})}

    (D / "manifest_split_fix.json").write_text(json.dumps({
        "fix": "val/test 内同源组多版本 -> 整体移回 train",
        "groups_moved": len(bad), "images_moved": moved,
        "remaining_multi_version_in_eval": len(still),
        "remaining_cross_split_group_leaks": len(leaks),
        "per_split_after": summary,
        "fixed_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 仅供参考",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"修复后：val/test 多版本组 = {len(still)}   跨 split 组泄漏 = {len(leaks)}")
    for s, d in summary.items():
        print(f"  {s:<6} 图={d['images']:<6} 空标签={d['empty']:<5} 源组={d['groups']}")


if __name__ == "__main__":
    main()
