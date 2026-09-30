"""只读审计：为 kitchen_vision_v2 构建收集「来源 / 许可 / 排除理由」逐项证据。

覆盖执行单第 1、2 项要求的：
  A. Home-fire 上游类别定义核实（Roboflow 403 环境下用数据本身判定）
  B. 旧 kitchen-images 64 张的标签与使用权限抽查
  C. 本机 stove / pan 数据中是否存在「合格纯无火画面」
  D. 所有来源的来源/许可/排除理由逐张记录 → CSV

只读：不修改任何被检查的数据；输出只写 --outdir 新目录，拒绝覆盖。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

KF = Path(r"C:\Users\ASUS\Documents\Codex\kitchen-fire-detection\datasets")
HOME_ZIP = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\data\online_sources"
                r"\home_fire_v1_val.zip")
GC = Path(r"D:\gc_kitchen_annotation.v2-version_two.yolov11")
IMG_EXT = {".jpg", ".jpeg", ".png"}

# 许可（来自各自 data.yaml / 上游页面，见 audit.json 的 license_evidence）
LICENSE = {
    "gc_kitchen_v2": ("CC BY 4.0", "data.yaml roboflow.license + 上游页面标示"),
    "fire_flame": ("Private", "data.yaml roboflow.license: Private（保留原使用边界，不对外再发布）"),
    "stove_detection": ("CC BY 4.0", "data.yaml roboflow.license"),
    "pan_detection": ("CC BY 4.0", "data.yaml roboflow.license"),
    "home_fire": ("未声明", "ZIP 内无 data.yaml/README，上游页面 403，无法确认许可 → 保守视为不可发布"),
    "kitchen_images_old": ("未声明", "目录内无 README/yaml；含图库素材，版权存疑"),
    "cctv_simuletic": ("CC BY-NC-4.0", "dataset/README.md: license cc-by-nc-4.0（非商业）"),
}


def md5(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        try:
            out.append((int(p[0]), *(float(x) for x in p[1:5])))
        except ValueError:
            continue
    return out


def audit_home_fire(rows: list[dict], facts: dict) -> None:
    """A. Home-fire 类别判定 + 逐张记录。"""
    if not HOME_ZIP.exists():
        facts["home_fire"] = {"status": "ZIP 缺失"}
        return
    z = zipfile.ZipFile(HOME_ZIP)
    per_img: dict[str, list[tuple[int, float, float, float, float]]] = {}
    for e in z.namelist():
        if e.endswith(".txt"):
            b = Path(e).stem
            txt = z.read(e).decode("utf-8", errors="replace")
            boxes = []
            for ln in txt.splitlines():
                p = ln.split()
                if len(p) >= 5:
                    try:
                        boxes.append((int(p[0]), *(float(x) for x in p[1:5])))
                    except ValueError:
                        pass
            per_img[b] = boxes
    z.close()

    comp = Counter()
    for b, boxes in per_img.items():
        cs = {c for c, *_ in boxes}
        if not cs:
            comp["empty"] += 1
        elif cs == {0}:
            comp["only_class0"] += 1
        elif cs == {1}:
            comp["only_class1"] += 1
        else:
            comp["both"] += 1

    facts["home_fire"] = {
        "zip": str(HOME_ZIP),
        "images": len(per_img),
        "boxes_class0": sum(1 for v in per_img.values() for c, *_ in v if c == 0),
        "boxes_class1": sum(1 for v in per_img.values() for c, *_ in v if c == 1),
        "composition": dict(comp),
        "class_name_evidence": (
            "Roboflow 上游页面返回 HTTP 403，ZIP 内亦无 data.yaml/README，"
            "故改用本机图像直接判定："
            "val_1（仅 class 0）画面为壁炉内明火 → class 0 = fire；"
            "val_100（仅 class 1）画面为空调机上方灰白烟羽 → class 1 = smoke。"
            "判定人：DeepSeek（模型），仅供参考，须人工复核。"
        ),
        "class_names": {0: "fire", 1: "smoke"},
        "usable_as_no_fire": (
            "仅 class 1 的 373 张 + 空标签 33 张 = 406 张不含 fire 框的图，"
            "可作第一轮『图像级无火误报测试』素材；但均为室内明火/烟气场景、"
            "非厨房，且含人工实验视频帧。"
        ),
    }

    for b, boxes in sorted(per_img.items()):
        rows.append({
            "source": "home_fire",
            "file": f"{b}.jpg",
            "split": "val(原)",
            "orig_classes": "|".join(str(c) for c in sorted({c for c, *_ in boxes})),
            "n_boxes": len(boxes),
            "mapped_classes": "1" if any(c == 0 for c, *_ in boxes) else "",
            "license": LICENSE["home_fire"][0],
            "license_evidence": LICENSE["home_fire"][1],
            "keep": "pending",
            "exclude_reason": "类别语义已本地判定，但上游许可无法确认；且非厨房场景",
        })


def audit_gc(rows: list[dict], facts: dict) -> None:
    """GC v2：逐张记录映射前后 + 排除理由。"""
    if not GC.exists():
        facts["gc_kitchen_v2"] = {"status": "缺失"}
        return
    stats = {}
    for split in ("train", "valid", "test"):
        ld, id_ = GC / split / "labels", GC / split / "images"
        if not ld.exists():
            continue
        pre = Counter(); post = Counter(); empty_pre = empty_post = 0
        groups_post: set[str] = set()
        for lbl in sorted(ld.glob("*.txt")):
            boxes = read_labels(lbl)
            stem = lbl.stem
            grp = stem.split(".rf.")[0] if ".rf." in stem else stem
            pre_c = Counter(c for c, *_ in boxes)
            for k, v in pre_c.items():
                pre[k] += v
            # 映射：1,2 -> fire(0)；0(反光)/3(无火炉位) 丢弃
            mapped = []
            for c, cx, cy, w, h in boxes:
                if c in (1, 2):
                    mapped.append((0, cx, cy, w, h))
            if not mapped:
                empty_post += 1
            else:
                groups_post.add(grp)
            if not boxes:
                empty_pre += 1
            post[0] += len(mapped)
            rows.append({
                "source": "gc_kitchen_v2",
                "file": lbl.name,
                "split": split,
                "orig_classes": "|".join(str(c) for c in sorted(pre_c)),
                "n_boxes": len(boxes),
                "mapped_classes": "0" if mapped else "",
                "license": LICENSE["gc_kitchen_v2"][0],
                "license_evidence": LICENSE["gc_kitchen_v2"][1],
                "keep": "yes" if mapped else "no",
                "exclude_reason": "" if mapped else "映射后无 fire 框（本数据集中实际为 0 张）",
            })
        stats[split] = {
            "images": sum(1 for _ in ld.glob("*.txt")),
            "boxes_pre": dict(pre),
            "boxes_post_fire": post[0],
            "empty_label_images_pre": empty_pre,
            "empty_label_images_post": empty_post,
            "source_groups_with_fire_post": len(groups_post),
        }
    facts["gc_kitchen_v2"] = {
        "root": str(GC),
        "per_split": stats,
        "mapping": {"1 flame-under-pot": "0 fire", "2 gas-stove-flame": "0 fire",
                    "0 flame-reflection": "丢弃（困难负例区域，不生成 fire 框）",
                    "3 no-flame": "丢弃（未点火炉位，不映射到 stove）"},
    }


def audit_old_kitchen(rows: list[dict], facts: dict) -> None:
    """B. 旧 kitchen-images 64 张：标签抽查 + 权限。"""
    ki = KF / "kitchen-images"
    if not ki.exists():
        facts["kitchen_images_old"] = {"status": "缺失"}
        return
    per_cat = {}
    for cat in sorted(p.name for p in ki.iterdir() if p.is_dir()):
        cd = ki / cat
        imgs = [p for p in cd.iterdir() if p.suffix.lower() in IMG_EXT]
        n_empty = 0
        cls_used = Counter()
        for img in imgs:
            lbl = img.with_suffix(".txt")
            boxes = read_labels(lbl) if lbl.exists() else []
            if not boxes:
                n_empty += 1
            for c, *_ in boxes:
                cls_used[c] += 1
            rows.append({
                "source": "kitchen_images_old",
                "file": f"{cat}/{img.name}",
                "split": "train(旧,已泄漏)",
                "orig_classes": "|".join(str(c) for c in sorted(cls_used)) if boxes else "",
                "n_boxes": len(boxes),
                "mapped_classes": "",
                "license": LICENSE["kitchen_images_old"][0],
                "license_evidence": LICENSE["kitchen_images_old"][1],
                "keep": "no",
                "exclude_reason": "执行单要求全部暂缓：有空标/伪标/图库素材，版权与场景名均存疑；且已泄漏进旧 combined/train",
            })
        per_cat[cat] = {
            "images": len(imgs),
            "empty_label": n_empty,
            "boxes_by_class": dict(cls_used),
            "avg_md5": None,
        }
    facts["kitchen_images_old"] = {
        "root": str(ki),
        "per_category": per_cat,
        "license": LICENSE["kitchen_images_old"][0],
        "license_evidence": LICENSE["kitchen_images_old"][1],
        "verdict": "全部暂缓，不纳入第一轮；需逐张重标并核实版权后方可考虑",
    }


def audit_stove_pan_for_no_fire(rows: list[dict], facts: dict) -> None:
    """C. 本机 stove / pan 中是否有合格纯无火画面。

    判据：我们把 stove→2、pan→3 映射进四槽。若某图只含 stove/pan 框而无 fire，
    它就是「无火但有关联物体」的候选难例（不是纯空图，比空图更有价值）。
    """
    facts["stove_pan_no_fire_candidates"] = {}
    for tag, sub, cmap in (("stove_detection", "stove-detection", {0: 2}),
                           ("pan_detection", "pan-detection", {1: 3})):
        root = KF / sub
        if not root.exists():
            continue
        info = {}
        for split in ("train", "valid", "test"):
            ld = root / split / "labels"
            if not ld.exists():
                continue
            n = n_empty = 0
            cls = Counter()
            cand = []
            for lbl in ld.glob("*.txt"):
                n += 1
                boxes = read_labels(lbl)
                if not boxes:
                    n_empty += 1
                for c, *_ in boxes:
                    cls[c] += 1
                kept = [c for c, *_ in boxes if c in cmap]
                if not kept:
                    cand.append(lbl.stem)
                rows.append({
                    "source": tag,
                    "file": f"{split}/{lbl.name}",
                    "split": split,
                    "orig_classes": "|".join(str(c) for c in sorted({c for c, *_ in boxes})),
                    "n_boxes": len(boxes),
                    "mapped_classes": "|".join(str(cmap[c]) for c in kept),
                    "license": LICENSE[tag][0],
                    "license_evidence": LICENSE[tag][1],
                    "keep": "yes" if kept else "candidate_no_fire",
                    "exclude_reason": "" if kept else "映射后无框 → 可作『有关联物体但无火』候选难例",
                })
            info[split] = {"labels": n, "empty_label": n_empty,
                           "boxes_by_orig_class": dict(cls),
                           "no_mapped_box_count": len(cand),
                           "no_mapped_box_examples": cand[:10]}
        facts["stove_pan_no_fire_candidates"][tag] = info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path,
                    default=Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work"
                                 r"\out\source_audit_v2"))
    args = ap.parse_args()
    if args.outdir.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{args.outdir}")
    args.outdir.mkdir(parents=True)

    rows: list[dict] = []
    facts: dict = {
        "audited_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 只读审计，仅供参考",
        "license_table": {k: {"license": v[0], "evidence": v[1]} for k, v in LICENSE.items()},
    }

    print("[A] Home-fire 类别判定 ...")
    audit_home_fire(rows, facts)
    print("[B] 旧 kitchen-images 标签与权限 ...")
    audit_old_kitchen(rows, facts)
    print("[C] 本机 stove/pan 无火候选 ...")
    audit_stove_pan_for_no_fire(rows, facts)
    print("[D] GC v2 映射前后统计 ...")
    audit_gc(rows, facts)

    cols = ["source", "file", "split", "orig_classes", "n_boxes",
            "mapped_classes", "license", "license_evidence", "keep", "exclude_reason"]
    with (args.outdir / "source_records.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    (args.outdir / "source_audit.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 控制台摘要
    print()
    hf = facts.get("home_fire", {})
    if hf:
        print(f"Home-fire: {hf.get('images')} 图  class0={hf.get('boxes_class0')}框 "
              f"class1={hf.get('boxes_class1')}框  {hf.get('composition')}")
        print(f"          类名判定: {hf.get('class_names')}")
    for tag, info in facts.get("stove_pan_no_fire_candidates", {}).items():
        for split, d in info.items():
            print(f"{tag}/{split}: 标签 {d['labels']} 空标签 {d['empty_label']} "
                  f"无映射框={d['no_mapped_box_count']} 原类框={d['boxes_by_orig_class']}")
    gc = facts.get("gc_kitchen_v2", {}).get("per_split", {})
    for split, d in gc.items():
        print(f"GC/{split}: 图 {d['images']}  映射前框 {d['boxes_pre']}  "
              f"映射后 fire={d['boxes_post_fire']}  空标签(后)={d['empty_label_images_post']}  "
              f"含火源组={d['source_groups_with_fire_post']}")
    print(f"\n逐项记录 {len(rows)} 行 → {args.outdir / 'source_records.csv'}")
    print(f"汇总 → {args.outdir / 'source_audit.json'}")


if __name__ == "__main__":
    main()
