#!/usr/bin/env python3
"""v21 对照结果汇总。

只读 experiment_meta.json（training）与 comparison_complete.json（comparison），
写出 summary.json 与 summary.md。不执行预测、不修改模型、不自动晋升。

用法：
    python3 summarize_v21.py \
        --training experiment_meta.json \
        --comparison comparison_complete.json \
        --out ./v21_summary_out
"""

import argparse
import json
import os
import sys


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_json(path: str, label: str):
    if not os.path.isfile(path):
        die(f"{label} 文件不存在: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        die(f"{label} 文件不是合法 JSON ({path}): {e}")


def require(obj, keys, ctx):
    """按键路径逐级取值，缺失即报错，绝不默认填 0。"""
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            die(f"缺少字段 {ctx}: {'.'.join(keys)}")
        cur = cur[k]
    return cur


def require_int(obj, keys, ctx) -> int:
    v = require(obj, keys, ctx)
    if not isinstance(v, int) or isinstance(v, bool):
        die(f"字段类型应为整数 {'.'.join(keys)} (ctx={ctx})，实际为 {type(v).__name__}")
    return v


def require_float_like(obj, keys, ctx) -> float:
    v = require(obj, keys, ctx)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        die(f"字段类型应为数值 {'.'.join(keys)} (ctx={ctx})，实际为 {type(v).__name__}")
    return float(v)


def build_row(models, label, entry, ctx):
    """提取 Markdown 表格行所需的全部计数。"""
    vis = require(entry, ["visible_flame", "supplemental", "0.5"], ctx)
    indoor = require(entry, ["sources", "nofire_real_indoor"], ctx)
    queue = require(entry, ["new_training_sources", "by_origin", "reviewed_commons_queue"], ctx)
    return {
        "label": label,
        "weights_sha256": require(entry, ["weights_sha256"], ctx),
        "blue_localized_iou50": require_int(
            entry, ["blue", "primary_localized_iou50"], ctx),
        "blue_images": require_int(entry, ["blue", "images"], ctx),
        "flame_tp": require_int(vis, ["tp"], ctx),
        "flame_fp": require_int(vis, ["fp"], ctx),
        "flame_fn": require_int(vis, ["fn"], ctx),
        "indoor_images": require_int(indoor, ["images"], ctx),
        "indoor_fp": require_int(indoor, ["fp"], ctx),
        "queue_images": require_int(queue, ["images"], ctx),
        "queue_tp": require_int(queue, ["tp"], ctx),
        "queue_fp": require_int(queue, ["fp"], ctx),
        "queue_fn": require_int(queue, ["fn"], ctx),
        "legacy_map50": require_float_like(entry, ["legacy", "map50"], ctx),
    }


def fmt_diff(delta: int) -> str:
    return f"{delta:+d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="v21 对照结果汇总（只读、不晋升）")
    ap.add_argument("--training", required=True, help="experiment_meta.json 路径")
    ap.add_argument("--comparison", required=True, help="comparison_complete.json 路径")
    ap.add_argument("--out", required=True, help="输出目录（必须不存在）")
    args = ap.parse_args()

    if os.path.exists(args.out):
        die(f"拒绝覆盖：--out 目录已存在: {args.out}")

    training = load_json(args.training, "training")
    comparison = load_json(args.comparison, "comparison")

    # training 顶层字段完整性检查（完整保留）
    for key in ["actual_optimizer_steps", "actual_batches", "config",
                "checkpoints", "serialized_frozen_state_equal",
                "serialized_adaptation", "environment"]:
        if key not in training:
            die(f"training 缺少顶层字段: {key}")

    models = comparison.get("models")
    if not isinstance(models, dict) or not models:
        die("comparison.models 缺失或为空")

    rows = []
    for label, entry in models.items():
        rows.append(build_row(models, label, entry, f"models[{label!r}]"))

    base_label = "v16_best"
    if base_label not in models:
        die(f"对照基线模型不存在: {base_label}（无晋升判断，仅作差值参照）")
    base = next(r for r in rows if r["label"] == base_label)

    # summary.json：保留模型完整原始子结构 + 提取行 + 相对差值 + training 全量
    summary = {
        "notes": {
            "evidence_status": "全部是已曝光开发反馈，未完成独立测试或上板验证",
            "promotion_policy": "不设晋升阈值，本脚本不自动晋升",
            "baseline_for_diffs": base_label,
        },
        "training": training,
        "comparison_plan": require(comparison, ["plan"], "comparison"),
        "models_raw": models,
        "extracted": {
            r["label"]: r for r in rows
        },
        "diff_vs_baseline": {
            r["label"]: {
                "blue_localized_iou50": r["blue_localized_iou50"] - base["blue_localized_iou50"],
                "flame_tp": r["flame_tp"] - base["flame_tp"],
                "flame_fp": r["flame_fp"] - base["flame_fp"],
                "flame_fn": r["flame_fn"] - base["flame_fn"],
                "indoor_fp": r["indoor_fp"] - base["indoor_fp"],
                "queue_tp": r["queue_tp"] - base["queue_tp"],
                "queue_fp": r["queue_fp"] - base["queue_fp"],
                "queue_fn": r["queue_fn"] - base["queue_fn"],
                "legacy_map50": r["legacy_map50"] - base["legacy_map50"],
            }
            for r in rows
        },
    }

    out_json = os.path.join(args.out, "summary.json")
    out_md = os.path.join(args.out, "summary.md")
    os.makedirs(args.out, exist_ok=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # summary.md：紧凑对照表
    lines = []
    lines.append("# v21 对照结果汇总")
    lines.append("")
    lines.append("> **全部是已曝光开发反馈，未完成独立测试或上板验证。**")
    lines.append("> 相对 `v16_best` 的差值列（Δ）仅为计数差，不设晋升阈值，不做晋升。")
    lines.append("")
    lines.append("| 标签 | 蓝焰定位数（IoU≥.5） | 蓝焰图数 | 紧火焰TP/FP/FN | 室内无火FP(/图数) | 14图训练曝光TP/FP/FN | 旧mAP50 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            "| {label} | {iou:d} (Δ{iou_d:+d}) | {bi} | {tp}/{fp}/{fn} "
            "(Δ{tp_d}/{fp_d}/{fn_d}) | {ifp}/{iimg} (Δ{ifp_d}) | "
            "{qtp}/{qfp}/{qfn} (Δ{qtp_d}/{qfp_d}/{qfn_d}) | {map:.4f} (Δ{map_d:+.4f}) |".format(
                label=r["label"],
                iou=r["blue_localized_iou50"],
                iou_d=r["blue_localized_iou50"] - base["blue_localized_iou50"],
                bi=r["blue_images"],
                tp=r["flame_tp"], fp=r["flame_fp"], fn=r["flame_fn"],
                tp_d=fmt_diff(r["flame_tp"] - base["flame_tp"]),
                fp_d=fmt_diff(r["flame_fp"] - base["flame_fp"]),
                fn_d=fmt_diff(r["flame_fn"] - base["flame_fn"]),
                ifp=r["indoor_fp"], iimg=r["indoor_images"],
                ifp_d=fmt_diff(r["indoor_fp"] - base["indoor_fp"]),
                qtp=r["queue_tp"], qfp=r["queue_fp"], qfn=r["queue_fn"],
                qtp_d=fmt_diff(r["queue_tp"] - base["queue_tp"]),
                qfp_d=fmt_diff(r["queue_fp"] - base["queue_fp"]),
                qfn_d=fmt_diff(r["queue_fn"] - base["queue_fn"]),
                map=r["legacy_map50"],
                map_d=r["legacy_map50"] - base["legacy_map50"],
            )
        )
    lines.append("")
    lines.append("training 元数据（actual_optimizer_steps、actual_batches、config、"
                 "checkpoints、serialized_frozen_state_equal、serialized_adaptation、"
                 "environment）完整保留于 `summary.json`。")
    lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"OK: 写入 {out_json}")
    print(f"OK: 写入 {out_md}")


if __name__ == "__main__":
    main()
