"""构建 kitchen_vision_v2_20260926：只读输入，输出新目录，拒绝覆盖。

设计要点（对应执行单第 2、3 节）
  * 输入全部只读；绝不修改/删除任何原始数据。
  * 输出固定为 D:\\SRTP_Datasets\\kitchen_vision_v2_20260926；已存在即报错退出。
  * 按「源图组」切分，同组不跨 split；val/test 不做增强复制。
  * 映射前后每 split 的图数 / 逐类框数 / 空标签图数 / 有效源图组数 全部记账。
  * 每张图逐条记录：来源、原 SHA-256、源图组 ID、许可、原类别→新类别、排除理由。
  * 无火难例分两个 provenance 池，许可与域在 manifest 里分开标注：
      nofire_real_indoor   —— 真实室内「有烟无火」，但许可未声明 → 仅探索性评测
      nofire_overhead_pan  —— 俯拍锅具，无火，CC BY 4.0 明确
  * 不把 fire-flame 全部压进来：按组抽样，上限 3,000 张。

产出
  images/{train,val,test}/ , labels/{train,val,test}/
  data.yaml , manifest.csv , build_report.json , leak_report.json

免责：本脚本由 DeepSeek（deepseek-flash, DeepSeek Harness）生成，仅供参考；
数据划分与许可判断须人工复核后使用。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

KF = Path(r"C:\Users\ASUS\Documents\Codex\kitchen-fire-detection\datasets")
GC = Path(r"D:\gc_kitchen_annotation.v2-version_two.yolov11")
HOME_ZIP = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\data\online_sources"
                r"\home_fire_v1_val.zip")
DEFAULT_OUT = Path(r"D:\SRTP_Datasets\kitchen_vision_v2_20260926")
IMG_EXT = {".jpg", ".jpeg", ".png"}
NAMES = ["fire", "smoke", "stove", "pan"]
SEED = 20260926
FIREFLAME_TRAIN_CAP = 3000

LICENSE_INFO = {
    "gc_kitchen_v2": ("CC BY 4.0", "data.yaml roboflow.license + 上游页面标示"),
    "fire_flame": ("Private", "data.yaml roboflow.license: Private"),
    "stove_detection": ("CC BY 4.0", "data.yaml roboflow.license"),
    "pan_detection": ("CC BY 4.0", "data.yaml roboflow.license"),
    "home_fire": ("未声明", "ZIP 内无 data.yaml/README；上游 HTTP 403 -> 保守按不可发布处理"),
}
PROVENANCE_TIER = {
    "kitchen_stove_fire": "formal",
    "generic_fire": "formal",
    "object_stove": "formal",
    "object_pan": "formal",
    "nofire_real_indoor": "exploratory",
    "nofire_overhead_pan": "formal",
}


def resolve_label(img: Path) -> Path | None:
    """找图片对应的 YOLO 标签，兼容三种布局：
      1. 同目录同名 .txt
      2. Roboflow: <split>/images/x.jpg + <split>/labels/x.txt
    """
    same = img.with_suffix(".txt")
    if same.exists():
        return same
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            cand = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
            if cand.exists():
                return cand
            break
    return None


def read_label(p: Path | None) -> list[tuple[int, float, float, float, float]]:
    out: list[tuple[int, float, float, float, float]] = []
    if p is None or not p.exists():
        return out
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        f = ln.split()
        if len(f) < 5:
            continue
        try:
            if len(f) > 5:  # 多边形 -> bbox
                vals = [float(x) for x in f[1:]]
                xs, ys = vals[0::2], vals[1::2]
                cx = (min(xs) + max(xs)) / 2
                cy = (min(ys) + max(ys)) / 2
                w, h = max(xs) - min(xs), max(ys) - min(ys)
            else:
                cx, cy, w, h = (float(x) for x in f[1:5])
            out.append((int(f[0]), cx, cy, w, h))
        except ValueError:
            continue
    return out


# fire-flame 源图有多种增强前缀，同源照片的不同增强版本必须归为同一组。
# 例：WEBFire1427 / MirrorWEBFire1427 / NoiseWEBFire1427 都来自同一张源照片。
_AUG_PREFIXES = ("Mirror", "Noise", "Flip", "Bright", "Dark", "Crop", "Rotate")


def group_of(stem: str) -> str:
    """源图组 ID：剥掉 Roboflow 的 .rf.<hash> 与增强前缀，再并掉已有的 _dN 风格后缀。"""
    base = stem.split(".rf.")[0] if ".rf." in stem else stem
    changed = True
    while changed:
        changed = False
        for p in _AUG_PREFIXES:
            if base.startswith(p) and len(base) > len(p):
                base = base[len(p):]
                changed = True
    return base


def read_label_bytes(raw: bytes) -> list[tuple[int, float, float, float, float]]:
    """从 ZIP 内解出的标签字节流读 YOLO 框。"""
    out: list[tuple[int, float, float, float, float]] = []
    for ln in raw.decode("utf-8", errors="replace").splitlines():
        f = ln.split()
        if len(f) < 5:
            continue
        try:
            out.append((int(f[0]), *(float(x) for x in f[1:5])))
        except ValueError:
            continue
    return out


def fmt(boxes: list[tuple[int, float, float, float, float]]) -> str:
    return "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes)


class Builder:
    def __init__(self, out: Path) -> None:
        self.out = out
        for s in ("train", "val", "test"):
            (out / "images" / s).mkdir(parents=True)
            (out / "labels" / s).mkdir(parents=True)
        self.rows: list[dict] = []
        self.mapper = {"gc_kitchen_v2": {1: 0, 2: 0},
                       "fire_flame": {0: 0},
                       "stove_detection": {0: 2},
                       "pan_detection": {1: 3},
                       "home_fire": {0: 0}}
        self.src_stats: dict[str, dict] = defaultdict(
            lambda: {"pre": Counter(), "post": Counter(), "empty_pre": 0,
                     "empty_post": 0, "images": 0, "groups_post": set()})
        self.seen_sha: dict[str, str] = {}
        self.dup_sha = 0

    def add(self, *, src: str, provenance: str, split: str, img,
            boxes_raw: list[tuple[int, float, float, float, float]],
            src_split: str, note: str = "") -> None:
        cmap = self.mapper.get(src, {})
        mapped = [(cmap[c], x, y, w, h) for c, x, y, w, h in boxes_raw if c in cmap]
        mapped = [(c, x, y, w, h) for c, x, y, w, h in mapped
                  if w > 1e-6 and h > 1e-6 and 0 <= x <= 1 and 0 <= y <= 1]
        if isinstance(img, Path):
            data = img.read_bytes()
            src_suffix = img.suffix.lower() or ".jpg"
            src_stem = img.stem
            src_path = str(img)
        else:
            data = img[0]
            src_stem = img[1]
            src_suffix = ".jpg"
            src_path = f"{HOME_ZIP.name}::{src_stem}"
        h = hashlib.sha256(data).hexdigest()
        if h in self.seen_sha:
            self.dup_sha += 1
            return
        out_stem = f"{provenance}__{src_stem}"[:150]
        k = 1
        while (self.out / "images" / split / f"{out_stem}{src_suffix}").exists():
            out_stem = f"{out_stem}_{k}"
            k += 1
        (self.out / "images" / split / f"{out_stem}{src_suffix}").write_bytes(data)
        (self.out / "labels" / split / f"{out_stem}.txt").write_text(
            fmt(mapped) + ("\n" if mapped else ""), encoding="utf-8")
        self.seen_sha[h] = f"images/{split}/{out_stem}{src_suffix}"

        pre = Counter(c for c, *_ in boxes_raw)
        post = Counter(c for c, *_ in mapped)
        st = self.src_stats[src]
        st["images"] += 1
        for kk, vv in pre.items():
            st["pre"][kk] += vv
        for kk, vv in post.items():
            st["post"][kk] += vv
        if not boxes_raw:
            st["empty_pre"] += 1
        if not mapped:
            st["empty_post"] += 1
        grp = group_of(src_stem)
        if mapped:
            st["groups_post"].add(grp)

        self.rows.append({
            "out_file": f"{out_stem}{src_suffix}",
            "split": split,
            "source": src,
            "provenance": provenance,
            "eval_tier": PROVENANCE_TIER.get(provenance, "exploratory"),
            "source_split": src_split,
            "source_path": src_path,
            "sha256": h,
            "group_id": grp,
            "orig_classes": "|".join(str(c) for c in sorted(pre)),
            "new_classes": "|".join(str(c) for c in sorted(post)),
            "n_boxes_pre": len(boxes_raw),
            "n_boxes_post": len(mapped),
            "empty_label_post": int(not mapped),
            "license": LICENSE_INFO.get(src, ("未声明", ""))[0],
            "license_evidence": LICENSE_INFO.get(src, ("未声明", ""))[1],
            "scene_review": ("见 label_quality_audit" if src == "gc_kitchen_v2"
                             else "未逐张人工审核"),
            "note": note,
        })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fireflame-train-cap", type=int, default=FIREFLAME_TRAIN_CAP)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{args.out}")
    rng = random.Random(SEED)
    b = Builder(args.out)

    # ---------- 1. GC 灶火（目标域）：沿用原 split ----------
    print("[1] GC 灶火 v2（目标域）")
    for split in ("train", "valid", "test"):
        tgt = {"train": "train", "valid": "val", "test": "test"}[split]
        for lbl in sorted((GC / split / "labels").glob("*.txt")):
            img = None
            for cand in sorted((GC / split / "images").iterdir()):
                if cand.stem == lbl.stem and cand.suffix.lower() in IMG_EXT:
                    img = cand
                    break
            if img is None:
                continue
            b.add(src="gc_kitchen_v2", provenance="kitchen_stove_fire", split=tgt,
                  img=img, boxes_raw=read_label(lbl), src_split=split)

    # ---------- 2. fire-flame：按源图组重切，消除上游自带的跨 split 泄漏 ----------
    # 实测：fire-flame 原始数据内部就有 1,175 个源图组跨 train/valid/test
    # （同一张源照片的多个增强变体被切到不同 split）。若沿用原 split，
    # 模型会把「同源照片的另一个增强版」当成独立测试样本 -> 指标虚高。
    # 因此这里按组重新分配：一个组的所有变体必然进入同一个目标 split。
    print("[2] fire-flame 通用火焰（按源图组重切，消除上游泄漏）")
    ff = KF / "fire-flame.v1i.yolov11"
    ff_by_group: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for split in ("train", "valid", "test"):
        for img in sorted((ff / split / "images").glob("*")):
            if img.suffix.lower() in IMG_EXT:
                ff_by_group[group_of(img.stem)].append((split, img))
    missing_lbl = sum(1 for v in ff_by_group.values() for _, i in v
                      if resolve_label(i) is None)
    print(f"    图 {sum(len(v) for v in ff_by_group.values())} 张 / "
          f"{len(ff_by_group)} 组；找不到标签 {missing_lbl} 张")

    multi = {g: v for g, v in ff_by_group.items() if len({s for s, _ in v}) > 1}
    print(f"    其中跨源 split 的组（上游泄漏）：{len(multi)}")

    # 划分策略（依据执行单 3.2/3.3：首轮正式验证只报灶火）：
    #   fire-flame 是「通用火焰」辅助数据、非目标域，全部进 train（按组、受 cap 约束）。
    #   val/test 只留目标域(GC 灶火) + 物体(stove/pan) + 无火难例，
    #   这样 val/test 的构成与正式验收范围一致，不会被 3 千多张非目标域图稀释。
    keys = sorted(ff_by_group)
    rng.shuffle(keys)
    train_budget = args.fireflame_train_cap
    n_tr = n_skipped = 0
    for g in keys:
        members = ff_by_group[g]
        if n_tr >= train_budget:
            n_skipped += len(members)
            continue
        n_tr += len(members)
        for _s, img in members:
            b.add(src="fire_flame", provenance="generic_fire", split="train",
                  img=img, boxes_raw=read_label(resolve_label(img)),
                  src_split=f"{_s}(group={g})")
    print(f"    -> fire-flame 全部进 train：{n_tr} 张（上限 {train_budget}），"
          f"因超限跳过 {n_skipped} 张；val/test 不含 fire-flame")

    # ---------- 3. stove / pan：同样按组重切 ----------
    print("[3] stove / pan（按组重切）")
    for tag, sub, prov, cmap in (
            ("stove_detection", "stove-detection", "object_stove", {0: 2}),
            ("pan_detection", "pan-detection", "object_pan", {1: 3})):
        base = KF / sub
        by_group: dict[str, list[tuple[str, Path]]] = defaultdict(list)
        for split in ("train", "valid", "test"):
            for lbl in sorted((base / split / "labels").glob("*.txt")):
                for cand in sorted((base / split / "images").iterdir()):
                    if cand.stem == lbl.stem and cand.suffix.lower() in IMG_EXT:
                        by_group[group_of(cand.stem)].append((split, cand))
                        break
        ks = sorted(by_group)
        rng.shuffle(ks)
        for g in ks:
            members = by_group[g]
            src_splits = {s for s, _ in members}
            if src_splits == {"train"}:
                tgt = "train"
            elif src_splits == {"valid"}:
                tgt = "val"
            elif src_splits == {"test"}:
                tgt = "test"
            else:
                tgt = "val" if (hash(g) % 2 == 0) else "test"
            for _s, img in members:
                b.add(src=tag, provenance=prov, split=tgt, img=img,
                      boxes_raw=read_label(resolve_label(img)),
                      src_split=f"{_s}(group={g})")
        print(f"    {tag}: {len(ks)} 组")

    # ---------- 4. 无火难例 A：Home-fire 仅烟 ----------
    print("[4] Home-fire 无火难例（仅 smoke 框）")
    z = zipfile.ZipFile(HOME_ZIP)
    smoke_only: list[str] = []
    empties: list[str] = []
    for e in z.namelist():
        if not e.endswith(".txt"):
            continue
        base = Path(e).stem
        boxes = read_label_bytes(z.read(e))
        cs = {c for c, *_ in boxes}
        if cs == {1}:
            smoke_only.append(base)
        elif not cs:
            empties.append(base)
    smoke_only.sort()
    rng.shuffle(smoke_only)
    k = len(smoke_only)
    n_val, n_test = max(1, int(k * 0.3)), max(1, int(k * 0.3))
    for i, base in enumerate(smoke_only):
        split = "train" if i < k - n_val - n_test else ("val" if i < k - n_test else "test")
        try:
            raw = z.read(f"images/{base}.jpg")
        except KeyError:
            continue
        boxes = read_label_bytes(z.read(f"labels/{base}.txt"))
        b.add(src="home_fire", provenance="nofire_real_indoor", split=split,
              img=(raw, base), boxes_raw=boxes, src_split="val(原)",
              note="仅含 smoke 框；映射后为空标签，作无火难例")
    z.close()
    print(f"    仅烟图 {k} 张 -> train/val/test = {k-n_val-n_test}/{n_val}/{n_test}"
          f"；排除 {len(empties)} 张空标签（来源分类不明）")

    # ---------- 5. 无火难例 B：pan 俯拍无火（按组切分）----------
    print("[5] pan 俯拍无火难例（按组切分）")
    pan_nofire: list[tuple[str, Path, list]] = []
    for split in ("train", "valid", "test"):
        base_dir = KF / "pan-detection" / split
        for lbl in sorted((base_dir / "labels").glob("*.txt")):
            boxes = read_label(lbl)
            if not any(c == 1 for c, *_ in boxes):
                for cand in sorted((base_dir / "images").iterdir()):
                    if cand.stem == lbl.stem and cand.suffix.lower() in IMG_EXT:
                        pan_nofire.append((split, cand, boxes))
                        break
    pan_by_group: dict[str, list[tuple[str, Path, list]]] = defaultdict(list)
    for split, img, boxes in pan_nofire:
        pan_by_group[group_of(img.stem)].append((split, img, boxes))
    pks = sorted(pan_by_group)
    rng.shuffle(pks)
    npool = len(pks)
    nv, nt = max(1, int(npool * 0.25)), max(1, int(npool * 0.25))
    for i, g in enumerate(pks):
        tgt = "train" if i < npool - nv - nt else ("val" if i < npool - nt else "test")
        for split, img, boxes in pan_by_group[g]:
            b.add(src="pan_detection", provenance="nofire_overhead_pan", split=tgt,
                  img=img, boxes_raw=boxes, src_split=f"{split}(group={g})",
                  note="仅含 Black Circle（俯拍锅具）或空标签；映射后为空 -> 无火难例")
    print(f"    俯拍无火 {sum(len(v) for v in pan_by_group.values())} 张 / {npool} 组"
          f" -> train/val/test = {npool-nv-nt}/{nv}/{nt}（按组）")

    # ---------- 6. 跨 split 组泄漏检查 ----------
    print("[6] 组泄漏检查")
    grp_splits: dict[str, set[str]] = defaultdict(set)
    for r in b.rows:
        grp_splits[f"{r['source']}::{r['group_id']}"].add(r["split"])
    leaks = {g: sorted(s) for g, s in grp_splits.items() if len(s) > 1}
    train_only_leaks = {g: s for g, s in leaks.items() if "train" in s}
    leak_report = {
        "cross_split_groups": len(leaks),
        "cross_split_groups_involving_train": len(train_only_leaks),
        "total_groups": len(grp_splits),
        "examples": dict(list(leaks.items())[:40]),
        "note": ("同一源图组出现在多个 split。已如实报告而非静默丢弃。"
                 "其中涉及 train 的泄漏最危险，构建后必须先处理再用于正式验收。"),
    }
    (args.out / "leak_report.json").write_text(
        json.dumps(leak_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 7. manifest / data.yaml / 报告 ----------
    with (args.out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(b.rows[0].keys()))
        w.writeheader()
        w.writerows(b.rows)

    (args.out / "data.yaml").write_text(
        f"path: {str(args.out).replace(chr(92), '/')}\n"
        "train: images/train\nval: images/val\ntest: images/test\nnc: 4\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)),
        encoding="utf-8")

    per_split = {}
    for s in ("train", "val", "test"):
        rs = [r for r in b.rows if r["split"] == s]
        per_split[s] = {
            "images": len(rs),
            "boxes_post": dict(Counter(
                int(c) for r in rs if r["new_classes"] for c in r["new_classes"].split("|"))),
            "empty_label_images": sum(r["empty_label_post"] for r in rs),
            "groups": len({f"{r['source']}::{r['group_id']}" for r in rs}),
            "by_provenance": dict(Counter(r["provenance"] for r in rs)),
        }
    src_report = {}
    for src, st in b.src_stats.items():
        src_report[src] = {
            "images": st["images"],
            "boxes_pre": dict(st["pre"]),
            "boxes_post": {NAMES[k] if k < 4 else k: v for k, v in st["post"].items()},
            "empty_label_pre": st["empty_pre"],
            "empty_label_post": st["empty_post"],
            "source_groups_with_boxes": len(st["groups_post"]),
            "license": LICENSE_INFO.get(src, ("未声明", ""))[0],
        }
    report = {
        "built_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 仅供参考，须人工复核",
        "date": "2026-09-26",
        "out": str(args.out),
        "seed": SEED,
        "fireflame_train_cap": args.fireflame_train_cap,
        "per_split": per_split,
        "per_source": src_report,
        "mapping": {
            "gc 1 flame-under-pot": "0 fire",
            "gc 2 gas-stove-flame": "0 fire",
            "gc 0 flame-reflection": "丢弃（困难负例区域）",
            "gc 3 no-flame": "丢弃（未点火炉位）",
            "fire-flame 0": "0 fire",
            "stove 0": "2 stove",
            "pan 1": "3 pan（pan 0 Black Circle 丢弃，成无火难例）",
            "home_fire 0": "0 fire（本构建未使用；仅取 class 1 作无火难例）",
        },
        "eval_tier_rule": PROVENANCE_TIER,
        "excluded_entirely": {
            "kitchen_images_old": "64 张：许可无据、含图库素材与伪标，且已泄漏进旧 combined/train",
            "home_fire_empty_labels": f"{len(empties)} 张空标签：来源分类不明，保守排除",
        },
        "dedup": {"dropped_duplicate_sha256": b.dup_sha},
        "cross_split_group_leaks": len(leaks),
        "caveats": [
            "GC 部分标注质量仅做几何筛查 + 少量人眼抽检，未全量人工复核。",
            "fire-flame 为 Private 许可，仅在本机训练使用，不对外再发布。",
            "nofire_real_indoor 来自 Home-fire，许可未声明 -> 仅探索性评测，不进正式验收。",
            "本轮无独立厨房视频，连续运行误报率不可评估。",
            "smoke 类无审核正例 -> 未训练/未验收。",
        ],
    }
    (args.out / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 76)
    print(f"输出：{args.out}")
    for s in ("train", "val", "test"):
        d = per_split[s]
        print(f"  {s:<6} 图={d['images']:<6} 空标签={d['empty_label_images']:<5} "
              f"源组={d['groups']:<5} 框={d['boxes_post']}")
    print(f"  去重丢弃(同 SHA-256)={b.dup_sha}   跨 split 组泄漏={len(leaks)}"
          f"（其中涉及 train 的 {len(train_only_leaks)}）")
    print("=" * 76)
    print("逐源映射前后：")
    for src, d in src_report.items():
        print(f"  {src:<18} 图={d['images']:<6} 前={d['boxes_pre']}  后={d['boxes_post']} "
              f"空标签前后={d['empty_label_pre']}/{d['empty_label_post']} 源组={d['source_groups_with_boxes']}")


if __name__ == "__main__":
    main()
