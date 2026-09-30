"""只读：把 baseline_*.json 汇总成可读报告（含 Markdown）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NAMES = ["fire", "smoke", "stove", "pan"]
TIER_DESC = {
    "clean": "干净（与 combined 零重叠）",
    "val": "图像在 combined/val",
    "train": "图像在 combined/train（完全泄漏）",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args()

    d = json.loads(args.report.read_text(encoding="utf-8"))
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out(f"报告文件   : {args.report}")
    out(f"权重       : {d['weights']}")
    out(f"SHA-256    : {d['weights_sha256']}")
    out(f"imgsz      : {d['imgsz']}    conf 下限 {d['conf_floor']}    NMS IoU {d['nms_iou']}")
    out(f"套件       : {d['suite']}")
    out()

    out("| 泄漏层级 | 说明 | 图数 | Box P | Box R | mAP50 | mAP50-95 | 跳过(含smoke正例) |")
    out("|---|---|---:|---:|---:|---:|---:|---:|")
    order = ["clean", "val", "train"]
    for tier in order:
        v = d["tiers"].get(tier)
        if not v:
            continue
        out(
            f"| **{tier}** | {TIER_DESC.get(tier, '')} | {v['images']} | {v['precision']} | "
            f"{v['recall']} | {v['mAP50']} | {v['mAP50-95']} | {v['skipped_smoke_positive_images']} |"
        )
    out()

    for tier in order:
        v = d["tiers"].get(tier)
        if not v:
            continue
        out(f"### 层级 {tier} 的逐类结果")
        out()
        pc = v.get("per_class")
        if not isinstance(pc, dict):
            out(f"（无非逐类数据：{pc}）")
            out()
            continue
        out("| 类别 | P | R | mAP50 | mAP50-95 |")
        out("|---|---:|---:|---:|---:|")
        for c in NAMES:
            m = pc.get(c)
            if not isinstance(m, dict):
                continue
            if m.get("precision") is None:
                out(f"| {c} | — | — | — | — |")
            else:
                out(
                    f"| {c} | {m['precision']} | {m['recall']} | "
                    f"{m['mAP50']} | {m['mAP50-95']} |"
                )
        out()

    s = d.get("scene_level_clean")
    if s:
        out("### 场景级：火焰有无判定（clean 层，conf=0.5，对齐板端阈值）")
        out()
        out(f"- 有火图 {s['gt_with_fire']} 张：命中 {s['true_positive_imgs']}，漏检 {s['false_negative_imgs']}"
            f"  → **图像级召回 {s['image_level_recall']}**")
        out(f"- 无火图 {s['gt_without_fire']} 张：正确 {s['true_negative_imgs']}，误报 {s['false_positive_imgs']}"
            f"  → **图像级误报率 {s['image_level_false_alarm_rate']}**")
        out()
        if s.get("examples"):
            out("失败样例（前若干条）：")
            out()
            out("| 类型 | 最高 fire 置信度 | 真值框 | 图片 |")
            out("|---|---:|---|---|")
            for kind, f, cf, box in s["examples"]:
                out(f"| {kind} | {cf} | `{box}` | `{f}` |")
            out()

    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nMarkdown 已写入：{args.md}")


if __name__ == "__main__":
    main()
