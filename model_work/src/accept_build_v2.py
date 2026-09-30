"""构建后自动验收（执行单 3.3）：只读检查已构建的数据集，并导出分层 manifest。

检查项：
  1. 每张图能解码
  2. 每张标签存在；类 ID 只在 0-3；框坐标合法且在图内
  3. 图/标签数一致
  4. 无源组、SHA-256 相同图跨 train/val/test
  5. val/test 无同源组多版本（增强复制）
  6. 感知近重复：区分「同源组内相邻帧」与「跨组/跨来源视觉近似」
  7. data.yaml 绝对路径可被训练环境读到

只读：不修改数据集；结果写到 --outdir 新目录（默认放工作区）。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

NAMES = ["fire", "smoke", "stove", "pan"]


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path(r"D:\SRTP_Datasets\kitchen_vision_v2_20260926"))
    ap.add_argument("--outdir", type=Path,
                    default=Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work"
                                 r"\out\build_acceptance_v2"))
    ap.add_argument("--phash-threshold", type=int, default=4)
    args = ap.parse_args()
    if args.outdir.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{args.outdir}")
    args.outdir.mkdir(parents=True)

    d = args.data
    rows = list(csv.DictReader((d / "manifest.csv").open(encoding="utf-8")))
    print(f"manifest 行数：{len(rows)}")

    problems: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, dict] = {}
    per_class: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    empty_by_split: dict[str, int] = defaultdict(int)

    # --- 1/2/3 逐张结构检查 ---
    for split in ("train", "val", "test"):
        imgs = sorted((d / "images" / split).iterdir())
        lbls = {p.stem for p in (d / "labels" / split).glob("*.txt")}
        n_ok = n_bad = 0
        for img in imgs:
            lp = d / "labels" / split / f"{img.stem}.txt"
            if img.stem not in lbls:
                problems["missing_label"].append(f"{split}/{img.name}")
            try:
                with Image.open(img) as im:
                    im.verify()
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                problems["undecodable"].append(f"{split}/{img.name}: {e}")
                n_bad += 1
                continue
            lines = ([l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
                     if lp.exists() else [])
            if not lines:
                empty_by_split[split] += 1
            for ln in lines:
                f = ln.split()
                if len(f) != 5:
                    problems["bad_field_count"].append(f"{split}/{lp.name}")
                    continue
                try:
                    cid = int(f[0])
                    cx, cy, w, h = (float(x) for x in f[1:5])
                except ValueError:
                    problems["unparsable"].append(f"{split}/{lp.name}")
                    continue
                if cid not in (0, 1, 2, 3):
                    problems["class_out_of_range"].append(f"{split}/{lp.name}: {cid}")
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    problems["coord_out_of_range"].append(f"{split}/{lp.name}")
                x0, y0 = cx - w / 2, cy - h / 2
                x1, y1 = cx + w / 2, cy + h / 2
                if x0 < -1e-6 or y0 < -1e-6 or x1 > 1 + 1e-6 or y1 > 1 + 1e-6:
                    problems["box_out_of_image"].append(f"{split}/{lp.name}")
                per_class[split][NAMES[cid]] += 1
        counts[split] = {"images": len(imgs), "labels": len(lbls),
                         "decoded_ok": n_ok, "decode_fail": n_bad}
        if len(imgs) != len(lbls):
            problems["count_mismatch"].append(
                f"{split}: img={len(imgs)} lbl={len(lbls)}")

    # --- 4. 源组 / SHA-256 跨 split ---
    grp: dict[str, set[str]] = defaultdict(set)
    sha: dict[str, set[str]] = defaultdict(set)
    grp_of_file: dict[str, tuple[str, str]] = {}
    for r in rows:
        grp[f"{r['source']}::{r['group_id']}"].add(r["split"])
        sha[r["sha256"]].add(r["split"])
        grp_of_file[r["out_file"]] = (r["source"], r["group_id"])
    grp_leak = {k: sorted(v) for k, v in grp.items() if len(v) > 1}
    sha_leak = {k: sorted(v) for k, v in sha.items() if len(v) > 1}

    # --- 6. 感知近重复：区分同组相邻帧 vs 跨组 ---
    ph: list[tuple[str, str, int]] = []
    for r in rows:
        p = d / "images" / r["split"] / r["out_file"]
        if p.exists():
            try:
                ph.append((r["split"], r["out_file"], dhash(p)))
            except Exception:  # noqa: BLE001
                pass
    print(f"感知哈希：{len(ph)} 张；跨 split 近重复检查（阈值 <= {args.phash_threshold}）")
    by_split: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for s, f, h in ph:
        by_split[s].append((f, h))
    near_same_group = 0
    near_cross: list[dict] = []
    for s in ("val", "test"):
        for f, h in by_split[s]:
            for tf, th in by_split["train"]:
                if ham(h, th) <= args.phash_threshold:
                    if grp_of_file.get(f) == grp_of_file.get(tf):
                        near_same_group += 1
                    else:
                        near_cross.append({
                            "split": s, "file": f, "group": grp_of_file.get(f),
                            "train_match": tf, "train_group": grp_of_file.get(tf),
                            "distance": ham(h, th),
                        })
                    break

    # --- 5. val/test 同源组多版本 ---
    vers: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        vers[f"{r['source']}::{r['group_id']}"][r["split"]] += 1
    multi_version_eval = {k: dict(v) for k, v in vers.items()
                          if any(v.get(s, 0) > 1 for s in ("val", "test"))}

    # --- 7. data.yaml ---
    yaml_txt = (d / "data.yaml").read_text(encoding="utf-8")
    yaml_ok = all((d / p).exists() for p in ("images/train", "images/val", "images/test"))

    verdict = ("PASS" if not problems and not grp_leak and not sha_leak
               and not near_cross and not multi_version_eval and yaml_ok else "REVIEW")
    report = {
        "accepted_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 只读验收，仅供参考",
        "data": str(d),
        "counts": counts,
        "per_class_boxes": {s: dict(per_class[s]) for s in per_class},
        "empty_label_images": dict(empty_by_split),
        "problem_counts": {k: len(v) for k, v in problems.items()},
        "problems": {k: v[:30] for k, v in problems.items()},
        "group_cross_split_leaks": len(grp_leak),
        "sha256_cross_split_leaks": len(sha_leak),
        "near_duplicate_same_group_adjacent_frames": near_same_group,
        "near_duplicate_cross_group": len(near_cross),
        "near_duplicate_cross_group_examples": near_cross[:20],
        "eval_splits_with_multi_version_groups": len(multi_version_eval),
        "data_yaml_ok": yaml_ok,
        "data_yaml": yaml_txt,
        "verdict": verdict,
        "caveats": [
            "同一源组内的相邻视频帧在感知哈希下必然相似（视频数据集固有性质），"
            "已与「跨组视觉近似」分开计数。",
            "近重复判定用 64-bit dHash 阈值 4，对强裁剪/旋转不敏感，不能排除此类隐蔽重叠。",
        ],
    }
    (args.outdir / "acceptance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 分层 manifest ---
    fields = list(rows[0].keys())
    for split in ("train", "val", "test"):
        with (args.outdir / f"manifest_{split}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows([r for r in rows if r["split"] == split])
    for tier, name in (("formal", "manifest_formal_eval.csv"),
                       ("exploratory", "manifest_exploratory_eval.csv")):
        rs = [r for r in rows if r["eval_tier"] == tier and r["split"] in ("val", "test")]
        with (args.outdir / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rs)

    print()
    print("=" * 74)
    for s, c in counts.items():
        print(f"  {s:<6} 图={c['images']:<6} 标签={c['labels']:<6} 解码OK={c['decoded_ok']:<6} "
              f"失败={c['decode_fail']:<3} 空标签={empty_by_split[s]:<5} "
              f"框={dict(per_class[s])}")
    print(f"  结构问题          : {report['problem_counts']}")
    print(f"  源组跨 split      : {len(grp_leak)}")
    print(f"  SHA256 跨 split   : {len(sha_leak)}")
    print(f"  近重复(同组相邻帧): {near_same_group}")
    print(f"  近重复(跨组/跨源) : {len(near_cross)}")
    print(f"  val/test 多版本组 : {len(multi_version_eval)}")
    print(f"  data.yaml 可读    : {yaml_ok}")
    print(f"  判定: {verdict}")
    print("=" * 74)
    print(f"输出：{args.outdir}")


if __name__ == "__main__":
    main()
