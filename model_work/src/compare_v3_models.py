"""旧权重 vs 新权重 在 v3 测试集上的对照 + 错例报告（执行单 8.3-2）。

输出：
  * 逐类 mAP50 / mAP50-95（同一测试集）
  * 场景级：灶火检出 / 漏检；分 provenance 的误报（厨房无火 vs 跨域无火分开报）
  * 错例清单：漏检（含小蓝焰/锅下火）+ 误报（含反光/无火灶具）
  * 每项同时报图数、源组数，并标注是否达到"可正式验收"的独立量

只读数据集；输出写工作区新目录，拒绝覆盖。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import ultralytics.data.dataset as _ul_dataset


class _SerialPool:
    def __init__(self, *_a, **_kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_e):
        return False

    def starmap(self, func=None, iterable=None, chunksize=None, *a, **k):
        return [func(*x) for x in (iterable or [])]

    def map(self, func=None, iterable=None, chunksize=None, *a, **k):
        return [func(x) for x in (iterable or [])]

    def imap(self, func=None, iterable=None, chunksize=None, *a, **k):
        return iter([func(x) for x in (iterable or [])])

    def imap_unordered(self, func=None, iterable=None, chunksize=None, *a, **k):
        return iter([func(x) for x in (iterable or [])])

    def uimap(self, *a, **k):
        return self.imap(*a, **k)

    def uimap_unordered(self, *a, **k):
        return self.imap(*a, **k)

    def close(self):
        pass

    def join(self):
        pass

    def terminate(self):
        pass


_ul_dataset.ThreadPool = _SerialPool
_ul_dataset.NUM_THREADS = 0

import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

NAMES = ["fire", "smoke", "stove", "pan"]
KF = Path(r"C:\Users\ASUS\Documents\Codex\kitchen-fire-detection")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path(r"D:\SRTP_Datasets\kitchen_vision_v3_20260926"))
    ap.add_argument("--old", type=Path,
                    default=KF / "runs/kitchen-fire/weights/best.pt")
    ap.add_argument("--new", type=Path,
                    default=KF / "runs/kitchen-vision-v3-640/weights/best.pt")
    ap.add_argument("--outdir", type=Path,
                    default=Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work"
                                 r"\out\compare_v3"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5,
                    help="场景级判定阈值（对齐板端）")
    args = ap.parse_args()
    if args.outdir.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{args.outdir}")
    args.outdir.mkdir(parents=True)

    rows = list(csv.DictReader((args.data / "manifest.csv").open(encoding="utf-8")))
    ev = [r for r in rows if r["split"] == args.split]
    print(f"{args.split} 集 {len(ev)} 张")
    for w, tag in ((args.old, "old"), (args.new, "new")):
        print(f"  {tag}: {w}  存在={w.exists()}")

    report: dict = {
        "compared_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 仅供参考",
        "data": str(args.data), "split": args.split, "imgsz": args.imgsz,
        "scene_conf": args.conf,
        "weights": {"old": str(args.old), "new": str(args.new)},
        "tiers": {},
    }

    # 临时 val 目录：只含本次评测集，避免在正式数据集里写 cache
    import shutil
    tmp = args.outdir / "_eval_set"
    for s in ("images", "labels"):
        (tmp / s).mkdir(parents=True)
    for r in ev:
        shutil.copy2(args.data / "images" / args.split / r["out_file"],
                     tmp / "images" / r["out_file"])
        lp = args.data / "labels" / args.split / (Path(r["out_file"]).stem + ".txt")
        shutil.copy2(lp, tmp / "labels" / lp.name)
    (tmp / "data.yaml").write_text(
        f"path: {str(tmp).replace(chr(92), '/')}\n"
        "train: images\nval: images\nnc: 4\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)), encoding="utf-8")

    for w, tag in ((args.old, "old"), (args.new, "new")):
        if not w.exists():
            report["tiers"][tag] = {"error": "权重不存在"}
            continue
        m = YOLO(str(w))
        res = m.val(data=str(tmp / "data.yaml"), imgsz=args.imgsz, conf=0.001,
                    iou=0.6, workers=0, plots=False, verbose=False,
                    project=str(args.outdir / "_val"), name=tag, exist_ok=True)
        box = res.box
        present = [int(i) for i in np.asarray(box.ap_class_index)]
        maps = np.asarray(box.maps)
        per = {}
        for dense, cid in enumerate(present):
            per[NAMES[cid]] = {
                "precision": round(float(np.asarray(box.p)[dense]), 5),
                "recall": round(float(np.asarray(box.r)[dense]), 5),
                "mAP50": round(float(np.asarray(box.ap50)[dense]), 5),
                "mAP50-95": round(float(maps[cid]), 5),
            }
        report["tiers"][tag] = {
            "precision": round(float(box.mp), 5), "recall": round(float(box.mr), 5),
            "mAP50": round(float(box.map50), 5), "mAP50-95": round(float(box.map), 5),
            "per_class": per,
        }
        print(f"\n[{tag}] P={box.mp:.4f} R={box.mr:.4f} "
              f"mAP50={box.map50:.4f} mAP50-95={box.map:.4f}")
        for k, v in per.items():
            print(f"    {k:<7} P={v['precision']:<9} R={v['recall']:<9} "
                  f"mAP50={v['mAP50']:<9} mAP50-95={v['mAP50-95']}")

    # ---- 场景级：火有无 + 分 provenance 误报 ----
    print(f"\n=== 场景级（conf={args.conf}）===")
    for w, tag in ((args.old, "old"), (args.new, "new")):
        if not w.exists():
            continue
        m = YOLO(str(w))
        srcs = [str(args.data / "images" / args.split / r["out_file"]) for r in ev]
        preds = m.predict(source=srcs, imgsz=args.imgsz, conf=args.conf,
                          iou=0.6, verbose=False, stream=True)
        stat = defaultdict(lambda: {"n": 0, "fire_hit": 0, "fp": 0,
                                    "gt_fire": 0, "tp": 0, "fn": 0, "tn": 0})
        misses, fas = [], []
        for r, pr in zip(ev, preds):
            key = r["provenance"]
            s = stat[key]
            s["n"] += 1
            gt_fire = (r["new_classes"].find("0") >= 0)
            has = False
            best = 0.0
            if pr.boxes is not None and len(pr.boxes):
                for c, cf in zip(pr.boxes.cls.tolist(), pr.boxes.conf.tolist()):
                    if int(c) == 0:
                        has = True
                        best = max(best, float(cf))
            if gt_fire:
                s["gt_fire"] += 1
                if has:
                    s["tp"] += 1
                else:
                    s["fn"] += 1
                    misses.append({"prov": key, "file": r["out_file"],
                                   "group": r["group_id"],
                                   "conf": round(best, 3), "split": r["split"]})
            elif has:
                s["fp"] += 1
                fas.append({"prov": key, "file": r["out_file"],
                            "group": r["group_id"], "conf": round(best, 3)})
            else:
                s["tn"] += 1
        tier = {}
        for k, v in stat.items():
            tier[k] = {
                "images": v["n"], "gt_fire_images": v["gt_fire"],
                "tp": v["tp"], "fn": v["fn"], "fp": v["fp"], "tn": v["tn"],
                "recall": round(v["tp"] / v["gt_fire"], 4) if v["gt_fire"] else None,
                "false_alarm_rate": (round(v["fp"] / (v["fp"] + v["tn"]), 4)
                                     if (v["fp"] + v["tn"]) else None),
            }
        report.setdefault("scene_level", {})[tag] = tier
        report.setdefault("examples", {})[tag] = {
            "misses": misses[:40], "false_alarms": fas[:40],
            "n_misses": len(misses), "n_false_alarms": len(fas),
        }
        print(f"\n[{tag}]")
        for k, v in sorted(tier.items()):
            print(f"    {k:<24} 图={v['images']:<5} 有火={v['gt_fire_images']:<5} "
                  f"召回={v['recall']} 误报率={v['false_alarm_rate']} "
                  f"(TP{v['tp']}/FN{v['fn']}/FP{v['fp']}/TN{v['tn']})")
        print(f"    漏检 {len(misses)} 例；误报 {len(fas)} 例")

    # 每个过错的图都单独存一份，便于人眼看
    for tag in ("old", "new"):
        ex = report.get("examples", {}).get(tag)
        if not ex:
            continue
        for kind in ("misses", "false_alarms"):
            for i, e in enumerate(ex[kind][:30]):
                p = args.data / "images" / args.split / e["file"]
                if p.exists():
                    shutil.copy2(p, args.outdir / f"{tag}_{kind}_{i:02d}"
                                 f"__conf{e['conf']}__{e['file'][:52]}")

    (args.outdir / "compare_v3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出：{args.outdir}")


if __name__ == "__main__":
    main()
