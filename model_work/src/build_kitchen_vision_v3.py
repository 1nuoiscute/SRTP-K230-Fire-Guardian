"""构建 kitchen_vision_v3_20260926：Kitchen Safety v3 + 已验证的 v2 组件。

依据（执行单 8.2 / 用户 2026-09-26 指示）
  * 只把审核通过的 Kitchen Safety `1 Flame` 映射为 `0 fire`；
    Burner / Smoke / Spill / safe 一律不映射。
  * 按拍摄段切分；同一段整体进一个 split；原 val/test 不用于验收。
  * 逐图确认「无火负例」：只纳入**已目视确认**的空标签帧（seg006 的 75 帧）。
    其余「有非火框但无 Flame 框」的图一律 pending，**不纳入本轮**。
  * 沿用 v2 已验收的组件：GC 灶火、fire-flame（按组重切）、stove/pan、
    Home-fire 仅烟（exploratory 层）、pan 俯拍无火。
  * 不沿用任何原数据集的 split 作为验收依据。

只读所有原数据；输出新目录，拒绝覆盖。
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
KS = Path(r"D:\Kitchen Safety.v3i.yolov11")
HOME_ZIP = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\data\online_sources"
                r"\home_fire_v1_val.zip")
SEG_ROWS = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out\ks_segments"
                r"\segment_rows.csv")
DEFAULT_OUT = Path(r"D:\SRTP_Datasets\kitchen_vision_v3_20260926")
IMG_EXT = {".jpg", ".jpeg", ".png"}
NAMES = ["fire", "smoke", "stove", "pan"]
KS_NAMES = ["Burner", "Flame", "Smoke", "Spill", "safe"]
SEED = 20260926

# 已验证无火的 KC 拍摄段（人眼确认：seg006 空标签帧 75 张 + 其余为无火灶台）
KS_CONFIRMED_NOFIRE_SEGMENTS = {"seg006"}
# 拍到 09-15 / 09-06 的整段留作评测，避免与 09-02 的 train 同源
KS_VAL_SESSIONS = {"20240906"}
KS_TEST_SESSIONS = {"20240915"}

FIREFLAME_TRAIN_CAP = 3000

LICENSE_INFO = {
    "gc_kitchen_v2": ("CC BY 4.0", "data.yaml roboflow.license"),
    "ks_v3": ("CC BY 4.0", "data.yaml roboflow.license（已人工抽图确认类别语义）"),
    "fire_flame": ("Private", "data.yaml roboflow.license: Private"),
    "stove_detection": ("CC BY 4.0", "data.yaml roboflow.license"),
    "pan_detection": ("CC BY 4.0", "data.yaml roboflow.license"),
    "home_fire": ("未声明", "ZIP 内无 yaml/README；上游 403"),
}
TIER = {
    "kitchen_stove_fire": "formal",
    "ks_flame": "formal",
    "ks_nofire_confirmed": "formal",
    "generic_fire": "formal",
    "object_stove": "formal",
    "object_pan": "formal",
    "nofire_real_indoor": "exploratory",
    "nofire_overhead_pan": "formal",
}
AUG_PREFIXES = ("Mirror", "Noise", "Flip", "Bright", "Dark", "Crop", "Rotate")


def sha256b(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def group_of(stem: str) -> str:
    base = stem.split(".rf.")[0] if ".rf." in stem else stem
    changed = True
    while changed:
        changed = False
        for p in AUG_PREFIXES:
            if base.startswith(p) and len(base) > len(p):
                base = base[len(p):]
                changed = True
    return base


def resolve_label(img: Path) -> Path | None:
    same = img.with_suffix(".txt")
    if same.exists():
        return same
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            c = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
            if c.exists():
                return c
            break
    return None


def read_label(p: Path | None):
    out = []
    if p is None or not p.exists():
        return out
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        f = ln.split()
        if len(f) < 5:
            continue
        try:
            if len(f) > 5:
                v = [float(x) for x in f[1:]]
                xs, ys = v[0::2], v[1::2]
                cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                w, h = max(xs) - min(xs), max(ys) - min(ys)
            else:
                cx, cy, w, h = (float(x) for x in f[1:5])
            out.append((int(f[0]), cx, cy, w, h))
        except ValueError:
            continue
    return out


def fmt(bs) -> str:
    return "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in bs)


class B:
    def __init__(self, out: Path):
        self.out = out
        for s in ("train", "val", "test"):
            (out / "images" / s).mkdir(parents=True)
            (out / "labels" / s).mkdir(parents=True)
        self.rows: list[dict] = []
        self.seen: dict[str, str] = {}
        self.dup = 0
        self.stats: dict[str, dict] = defaultdict(
            lambda: {"pre": Counter(), "post": Counter(), "img": 0,
                     "empty_pre": 0, "empty_post": 0, "groups": set()})

    def add(self, *, src, prov, split, img, boxes_raw, src_split, note="", cmap=None):
        cmap = cmap or {}
        mapped = [(cmap[c], x, y, w, h) for c, x, y, w, h in boxes_raw if c in cmap]
        mapped = [(c, x, y, w, h) for c, x, y, w, h in mapped
                  if w > 1e-6 and h > 1e-6 and 0 <= x <= 1 and 0 <= y <= 1]
        if isinstance(img, Path):
            data = img.read_bytes()
            stem, suf, spath = img.stem, (img.suffix.lower() or ".jpg"), str(img)
        else:
            data, stem = img[0], img[1]
            suf, spath = ".jpg", f"{HOME_ZIP.name}::{stem}"
        h = sha256b(data)
        if h in self.seen:
            self.dup += 1
            return
        out_stem = f"{prov}__{stem}"[:150]
        k = 1
        while (self.out / "images" / split / f"{out_stem}{suf}").exists():
            out_stem = f"{out_stem}_{k}"
            k += 1
        (self.out / "images" / split / f"{out_stem}{suf}").write_bytes(data)
        (self.out / "labels" / split / f"{out_stem}.txt").write_text(
            fmt(mapped) + ("\n" if mapped else ""), encoding="utf-8")
        self.seen[h] = f"{split}/{out_stem}{suf}"
        pre = Counter(c for c, *_ in boxes_raw)
        post = Counter(c for c, *_ in mapped)
        st = self.stats[src]
        st["img"] += 1
        for kk, vv in pre.items():
            st["pre"][kk] += vv
        for kk, vv in post.items():
            st["post"][kk] += vv
        if not boxes_raw:
            st["empty_pre"] += 1
        if not mapped:
            st["empty_post"] += 1
        g = group_of(stem)
        if mapped:
            st["groups"].add(g)
        self.rows.append({
            "out_file": f"{out_stem}{suf}", "split": split, "source": src,
            "provenance": prov, "eval_tier": TIER.get(prov, "exploratory"),
            "source_split": src_split, "source_path": spath, "sha256": h,
            "group_id": g,
            "orig_classes": "|".join(str(c) for c in sorted(pre)),
            "new_classes": "|".join(str(c) for c in sorted(post)),
            "n_boxes_pre": len(boxes_raw), "n_boxes_post": len(mapped),
            "empty_label_post": int(not mapped),
            "license": LICENSE_INFO.get(src, ("未声明", ""))[0],
            "license_evidence": LICENSE_INFO.get(src, ("未声明", ""))[1],
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
    b = B(args.out)

    # ---------- 1. Kitchen Safety v3（目标域新数据）----------
    print("[1] Kitchen Safety v3")
    seg = {r["file"]: r for r in csv.DictReader(SEG_ROWS.open(encoding="utf-8"))}

    def ks_target_split(session: str) -> str:
        if session in KS_TEST_SESSIONS:
            return "test"
        if session in KS_VAL_SESSIONS:
            return "val"
        return "train"

    ks_rows = []
    for split in ("train", "valid", "test"):
        for img in sorted((KS / split / "images").iterdir()):
            if img.suffix.lower() in IMG_EXT:
                ks_rows.append((split, img))
    n_flame = n_nofire = n_pending = 0
    pending_rows = []
    for orig_split, img in ks_rows:
        sr = seg.get(img.name)
        if sr is None:
            n_pending += 1
            pending_rows.append({"file": img.name, "reason": "无法匹配拍摄段记录"})
            continue
        segment = sr["segment"]
        # 会话 = 日期或 snap 组
        m = segment
        session = ("20240915" if segment in ("seg006", "seg007", "seg008")
                   else "20240906" if segment == "seg005"
                   else "20240902" if segment in ("seg001", "seg002", "seg003", "seg004")
                   else "snapshot")
        boxes = read_label(resolve_label(img))
        has_flame = any(c == 1 for c, *_ in boxes)
        if has_flame:
            tgt = ks_target_split(session)
            b.add(src="ks_v3", prov="ks_flame", split=tgt, img=img, boxes_raw=boxes,
                  src_split=f"{orig_split}/{segment}",
                  cmap={1: 0}, note="Kitchen Safety 1 Flame -> 0 fire")
            n_flame += 1
        elif not boxes and segment in KS_CONFIRMED_NOFIRE_SEGMENTS:
            # 已目视确认无火的空标签帧
            tgt = ks_target_split(session)
            b.add(src="ks_v3", prov="ks_nofire_confirmed", split=tgt, img=img,
                  boxes_raw=boxes, src_split=f"{orig_split}/{segment}", cmap={},
                  note="空标签且已目视确认无可见火焰（人眼确认 seg006）")
            n_nofire += 1
        else:
            # 有非火框但无 Flame，或未确认的空标签 -> 一律 pending，不纳入
            n_pending += 1
            pending_rows.append({
                "file": img.name, "segment": segment,
                "orig_classes": sr.get("classes", ""),
                "reason": ("含非火框（Burner/Smoke/Spill/safe）但无 Flame 框；"
                           "尚未逐张确认是否漏标火焰，本轮不纳入"),
            })
    print(f"    Flame 正例 {n_flame}；已确认无火 {n_nofire}；pending（不纳入）{n_pending}")
    with (args.out / "ks_pending_review.csv").open("w", encoding="utf-8",
                                                  newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "segment", "orig_classes", "reason"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(pending_rows)

    # ---------- 2. GC 灶火（沿用 v2 结论：按视觉族已重切过，这里直接用 v2 的划分）----------
    print("[2] GC 灶火 v2（沿用 v2 的 split 划分）")
    V2 = Path(r"D:\SRTP_Datasets\kitchen_vision_v2_20260926")
    v2rows = list(csv.DictReader((V2 / "manifest.csv").open(encoding="utf-8")))
    n = 0
    for r in v2rows:
        if r["source"] != "gc_kitchen_v2":
            continue
        src = Path(r["source_path"])
        if not src.exists():
            continue
        b.add(src="gc_kitchen_v2", prov="kitchen_stove_fire", split=r["split"],
              img=src, boxes_raw=read_label(Path(str(src).replace("images", "labels"))
                                            .with_suffix(".txt")),
              src_split=r["source_split"], cmap={1: 0, 2: 0},
              note="沿用 v2 的视觉族重切结果")
        n += 1
    print(f"    {n} 张")

    # ---------- 3. fire-flame（按组重切，全部进 train）----------
    print("[3] fire-flame（按组重切，全部进 train）")
    ff = KF / "fire-flame.v1i.yolov11"
    g: dict[str, list] = defaultdict(list)
    for sp in ("train", "valid", "test"):
        for img in sorted((ff / sp / "images").glob("*")):
            if img.suffix.lower() in IMG_EXT:
                g[group_of(img.stem)].append((sp, img))
    keys = sorted(g)
    rng.shuffle(keys)
    cap, taken = args.fireflame_train_cap, 0
    for k in keys:
        if taken >= cap:
            break
        for sp, img in g[k]:
            b.add(src="fire_flame", prov="generic_fire", split="train", img=img,
                  boxes_raw=read_label(resolve_label(img)), src_split=f"{sp}(g={k})",
                  cmap={0: 0})
            taken += 1
    print(f"    train {taken} 张")

    # ---------- 4. stove / pan（按组）----------
    print("[4] stove / pan（按组）")
    for tag, sub, prov, cmap in (("stove_detection", "stove-detection", "object_stove", {0: 2}),
                                 ("pan_detection", "pan-detection", "object_pan", {1: 3})):
        by = defaultdict(list)
        for sp in ("train", "valid", "test"):
            for lbl in sorted((KF / sub / sp / "labels").glob("*.txt")):
                for c in sorted((KF / sub / sp / "images").iterdir()):
                    if c.stem == lbl.stem and c.suffix.lower() in IMG_EXT:
                        by[group_of(c.stem)].append((sp, c))
                        break
        ks_ = sorted(by)
        rng.shuffle(ks_)
        for k in ks_:
            members = by[k]
            ss = {s for s, _ in members}
            tgt = ("train" if ss == {"train"} else "val" if ss == {"valid"}
                   else "test" if ss == {"test"} else ("val" if hash(k) % 2 else "test"))
            for sp, img in members:
                b.add(src=tag, prov=prov, split=tgt, img=img,
                      boxes_raw=read_label(resolve_label(img)),
                      src_split=f"{sp}(g={k})", cmap=cmap)
        print(f"    {tag}: {len(ks_)} 组")

    # ---------- 5. 无火难例：Home-fire 仅烟 + pan 俯拍 ----------
    print("[5] 无火难例")
    z = zipfile.ZipFile(HOME_ZIP)
    smoke_only = []
    for e in z.namelist():
        if not e.endswith(".txt"):
            continue
        base = Path(e).stem
        raw = z.read(e).decode("utf-8", errors="replace")
        cs = set()
        for ln in raw.splitlines():
            f = ln.split()
            if len(f) >= 5:
                cs.add(int(f[0]))
        if cs == {1}:
            smoke_only.append(base)
    smoke_only.sort()
    rng.shuffle(smoke_only)
    kk = len(smoke_only)
    nv, nt = max(1, int(kk * 0.3)), max(1, int(kk * 0.3))
    for i, base in enumerate(smoke_only):
        sp = "train" if i < kk - nv - nt else ("val" if i < kk - nt else "test")
        try:
            raw = z.read(f"images/{base}.jpg")
        except KeyError:
            continue
        boxes = []
        for ln in z.read(f"labels/{base}.txt").decode("utf-8", errors="replace").splitlines():
            f = ln.split()
            if len(f) >= 5:
                boxes.append((int(f[0]), *(float(x) for x in f[1:5])))
        b.add(src="home_fire", prov="nofire_real_indoor", split=sp, img=(raw, base),
              boxes_raw=boxes, src_split="val(原)", cmap={0: 0},
              note="仅 smoke 框 -> 映射后空标签；许可未声明，仅探索性")
    z.close()
    print(f"    Home-fire 仅烟 {kk} 张")

    pan_no = []
    for sp in ("train", "valid", "test"):
        for lbl in sorted((KF / "pan-detection" / sp / "labels").glob("*.txt")):
            boxes = read_label(lbl)
            if not any(c == 1 for c, *_ in boxes):
                for c in sorted((KF / "pan-detection" / sp / "images").iterdir()):
                    if c.stem == lbl.stem and c.suffix.lower() in IMG_EXT:
                        pan_no.append((sp, c, boxes))
                        break
    byg = defaultdict(list)
    for sp, img, bx in pan_no:
        byg[group_of(img.stem)].append((sp, img, bx))
    pks = sorted(byg)
    rng.shuffle(pks)
    nv2, nt2 = max(1, int(len(pks) * .25)), max(1, int(len(pks) * .25))
    for i, k in enumerate(pks):
        tgt = "train" if i < len(pks) - nv2 - nt2 else ("val" if i < len(pks) - nt2 else "test")
        for sp, img, bx in byg[k]:
            b.add(src="pan_detection", prov="nofire_overhead_pan", split=tgt, img=img,
                  boxes_raw=bx, src_split=f"{sp}(g={k})", cmap={},
                  note="俯拍锅具无火 -> 空标签难例")
    print(f"    pan 俯拍无火 {len(pks)} 组")

    # ---------- 6. 泄漏检查 ----------
    gs: dict[str, set] = defaultdict(set)
    for r in b.rows:
        gs[f"{r['source']}::{r['group_id']}"].add(r["split"])
    leaks = {k: sorted(v) for k, v in gs.items() if len(v) > 1}
    (args.out / "leak_report.json").write_text(json.dumps({
        "cross_split_groups": len(leaks), "total_groups": len(gs),
        "examples": dict(list(leaks.items())[:40]),
        "note": "同一源图组跨 split；应为 0（KS 按拍摄段，其余按组）。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[6] 跨 split 组泄漏: {len(leaks)}")

    # ---------- 7. 输出 ----------
    with (args.out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(b.rows[0].keys()))
        w.writeheader()
        w.writerows(b.rows)
    (args.out / "data.yaml").write_text(
        f"path: {str(args.out).replace(chr(92), '/')}\n"
        "train: images/train\nval: images/val\ntest: images/test\nnc: 4\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)), encoding="utf-8")

    per = {}
    for s in ("train", "val", "test"):
        rs = [r for r in b.rows if r["split"] == s]
        per[s] = {
            "images": len(rs),
            "boxes": dict(Counter(int(c) for r in rs if r["new_classes"]
                                  for c in r["new_classes"].split("|"))),
            "empty": sum(r["empty_label_post"] for r in rs),
            "groups": len({f"{r['source']}::{r['group_id']}" for r in rs}),
            "by_provenance": dict(Counter(r["provenance"] for r in rs)),
        }
    src_rep = {}
    for s, st in b.stats.items():
        src_rep[s] = {"images": st["img"], "boxes_pre": dict(st["pre"]),
                      "boxes_post": {NAMES[k] if k < 4 else k: v
                                     for k, v in st["post"].items()},
                      "empty_pre": st["empty_pre"], "empty_post": st["empty_post"],
                      "groups_with_boxes": len(st["groups"])}
    (args.out / "build_report.json").write_text(json.dumps({
        "built_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 仅供参考",
        "date": "2026-09-26", "out": str(args.out), "seed": SEED,
        "per_split": per, "per_source": src_rep,
        "mapping": {"ks 1 Flame": "0 fire（唯一映射）",
                    "ks 0 Burner/2 Smoke/3 Spill/4 safe": "不映射",
                    "gc 1 flame-under-pot / 2 gas-stove-flame": "0 fire",
                    "gc 0 flame-reflection / 3 no-flame": "丢弃",
                    "fire-flame 0": "0 fire", "stove 0": "2 stove", "pan 1": "3 pan",
                    "home_fire 0": "0 fire（本构建仅取 class1 作无火难例）"},
        "ks_pending_excluded": n_pending,
        "ks_pending_reason": "有非火框但无 Flame 框的图尚未逐张确认是否漏标火焰；"
                             "已在 KS Spill 组发现 1 张明显蓝焰漏标，故保守排除。",
        "cross_split_group_leaks": len(leaks),
        "caveats": [
            "KS 的 smoke 类已确认主要是蒸汽，不映射。",
            "无火负例仅纳入了目视确认的 KS 空标签帧 75 张 + Home-fire 仅烟 + pan 俯拍。",
            "无独立厨房视频，连续运行误报率不可评估。",
            "smoke 类零正例 -> 未训练/未验收。",
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 76)
    for s, d in per.items():
        print(f"  {s:<6} 图={d['images']:<6} 空标签={d['empty']:<5} "
              f"源组={d['groups']:<5} 框={d['boxes']}")
    print(f"  去重丢弃={b.dup}  跨 split 泄漏={len(leaks)}")
    print("=" * 76)
    for s, d in src_rep.items():
        print(f"  {s:<18} 图={d['images']:<6} 前={d['boxes_pre']} 后={d['boxes_post']} "
              f"空标签 {d['empty_pre']}->{d['empty_post']}")


if __name__ == "__main__":
    main()
