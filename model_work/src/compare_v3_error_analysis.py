"""旧/新权重错例对照（执行单 8.3-2）——聚焦用户指定的三类：
  ① 蓝焰（小蓝焰）漏检
  ② 锅下火（被锅遮挡）漏检
  ③ 无火误报（分 provenance：厨房无火 vs 跨域无火）

所有指标标为**探索性**；不用于替换板端模型。

只读数据集；输出写工作区新目录，拒绝覆盖。
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import ultralytics.data.dataset as _ul_dataset


class _SerialPool:
    """沙箱禁止命名进程管道：替换 ultralytics 的 ThreadPool。"""

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
from PIL import Image, ImageDraw  # noqa: E402
from ultralytics import YOLO  # noqa: E402

NAMES = ["fire", "smoke", "stove", "pan"]
KF = Path(r"C:\Users\ASUS\Documents\Codex\kitchen-fire-detection")

# 蓝焰/橙焰的几何与颜色判据（用于给错例**归类**，不用于自动裁决）
BLUE_FLAME_RGB = (60, 170, 255)
ORANGE_FLAME_RGB = (255, 140, 40)


def boxes_of(lbl: Path):
    out = []
    if not lbl.exists():
        return out
    for ln in lbl.read_text(encoding="utf-8", errors="replace").splitlines():
        f = ln.split()
        if len(f) < 5:
            continue
        try:
            out.append((int(f[0]), *(float(x) for x in f[1:5])))
        except ValueError:
            continue
    return out


def box_hue(img: Image.Image, box, W: int, H: int) -> str:
    """判断框内主色偏向蓝焰还是橙焰（粗略）。"""
    _, cx, cy, w, h = box
    x0 = max(0, int((cx - w / 2) * W))
    y0 = max(0, int((cy - h / 2) * H))
    x1 = min(W, int((cx + w / 2) * W))
    y1 = min(H, int((cy + h / 2) * H))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return "unknown"
    crop = img.convert("RGB").crop((x0, y0, x1, y1))
    px = list(crop.getdata())
    n = len(px)
    if not n:
        return "unknown"
    blue = sum(1 for r, g, b in px if b > 130 and b - r > 50 and b - g > 25) / n
    orange = sum(1 for r, g, b in px if r > 150 and r - b > 70 and g > 60) / n
    if blue > orange * 1.4 and blue > 0.02:
        return "blue"
    if orange > blue * 1.4 and orange > 0.02:
        return "orange"
    return "mixed_or_dim"


def annotate(src: Path, lbl: Path, dst: Path, pred_boxes, title: str) -> None:
    """存一张带真值(绿)与预测(红)的对照图。"""
    try:
        with Image.open(src) as im0:
            im = im0.convert("RGB")
        im.thumbnail((900, 900))
        W, H = im.size
        dr = ImageDraw.Draw(im)
        for c, cx, cy, w, h in boxes_of(lbl):
            x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
            x1, y1 = (cx + w / 2) * W, (cy + h / 2) * H
            dr.rectangle([x0, y0, x1, y1], outline=(60, 230, 120), width=3)
            dr.text((x0 + 3, max(0, y0 - 12)), f"GT:{NAMES[c]}", fill=(60, 230, 120))
        for c, cf, x0, y0, x1, y1 in pred_boxes:
            dr.rectangle([x0, y0, x1, y1], outline=(255, 60, 60), width=2)
            dr.text((x0 + 3, min(H - 12, y1 + 1)), f"P:{NAMES[c]} {cf:.2f}",
                    fill=(255, 60, 60))
        dr.text((5, 5), title[:110], fill=(255, 255, 0))
        im.save(dst, quality=88)
    except Exception:  # noqa: BLE001
        pass


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
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()
    if args.outdir.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{args.outdir}")
    args.outdir.mkdir(parents=True)

    rows = list(csv.DictReader((args.data / "manifest.csv").open(encoding="utf-8")))
    ev = [r for r in rows if r["split"] == args.split]
    print(f"{args.split} 集 {len(ev)} 张")
    for w, t in ((args.old, "old"), (args.new, "new")):
        print(f"  {t}: {w.name} 存在={w.exists()}")

    # 评测用临时目录（避免在正式数据集里写 cache）
    tmp = args.outdir / "_evalset"
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

    report: dict = {
        "compared_by": "DeepSeek (deepseek-flash, DeepSeek Harness)",
        "status": "EXPLORATORY — 探索性结果，不得据以替换板端模型",
        "data": str(args.data), "split": args.split, "scene_conf": args.conf,
        "weights": {"old": str(args.old), "new": str(args.new)},
    }

    for w, tag in ((args.old, "old"), (args.new, "new")):
        if not w.exists():
            report[tag] = {"error": "权重不存在（训练尚未完成？）"}
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
                "P": round(float(np.asarray(box.p)[dense]), 4),
                "R": round(float(np.asarray(box.r)[dense]), 4),
                "mAP50": round(float(np.asarray(box.ap50)[dense]), 4),
                "mAP50-95": round(float(maps[cid]), 4),
            }
        report[tag] = {"overall": {
            "P": round(float(box.mp), 4), "R": round(float(box.mr), 4),
            "mAP50": round(float(box.map50), 4),
            "mAP50-95": round(float(box.map), 4)}, "per_class": per}
        print(f"\n[{tag}] P={box.mp:.4f} R={box.mr:.4f} "
              f"mAP50={box.map50:.4f} mAP50-95={box.map:.4f}")
        for k, v in per.items():
            print(f"    {k:<7} P={v['P']:<8} R={v['R']:<8} "
                  f"mAP50={v['mAP50']:<8} mAP50-95={v['mAP50-95']}")

    # ---- 场景级 + 三类错例 ----
    exdir = args.outdir / "examples"
    exdir.mkdir(parents=True)
    for w, tag in ((args.old, "old"), (args.new, "new")):
        if not w.exists():
            continue
        m = YOLO(str(w))
        srcs = [str(args.data / "images" / args.split / r["out_file"]) for r in ev]
        preds = m.predict(source=srcs, imgsz=args.imgsz, conf=args.conf,
                          iou=0.6, verbose=False, stream=True)
        by_prov = defaultdict(lambda: {"n": 0, "gt_fire": 0, "tp": 0, "fn": 0,
                                       "fp": 0, "tn": 0})
        miss_blue, miss_underpot, fp_kitchen, fp_cross = [], [], [], []
        for r, pr in zip(ev, preds):
            pv = by_prov[r["provenance"]]
            pv["n"] += 1
            img_path = args.data / "images" / args.split / r["out_file"]
            lbl_path = args.data / "labels" / args.split / (Path(r["out_file"]).stem + ".txt")
            gt = boxes_of(lbl_path)
            gt_fire = [(c, x, y, w, h) for c, x, y, w, h in gt if c == 0]
            pred_fire = []
            if pr.boxes is not None and len(pr.boxes):
                W, H = pr.orig_shape[1], pr.orig_shape[0]
                for c, cf, xyxy in zip(pr.boxes.cls.tolist(),
                                       pr.boxes.conf.tolist(),
                                       pr.boxes.xyxy.tolist()):
                    if int(c) == 0:
                        pred_fire.append((int(c), float(cf), xyxy[0], xyxy[1],
                                          xyxy[2], xyxy[3]))
            if gt_fire:
                pv["gt_fire"] += 1
                if pred_fire:
                    pv["tp"] += 1
                else:
                    pv["fn"] += 1
                    # 归类：蓝焰 / 橙焰 / 锅下（用真值框颜色与面积判断）
                    try:
                        with Image.open(img_path) as im:
                            im = im.convert("RGB")
                            W, H = im.size
                        kinds = [box_hue(im, b, W, H) for b in gt_fire]
                    except Exception:  # noqa: BLE001
                        kinds = ["unknown"]
                    rec = {"prov": r["provenance"], "file": r["out_file"],
                           "group": r["group_id"],
                           "gt_boxes": len(gt_fire), "hues": kinds}
                    if "blue" in kinds:
                        miss_blue.append(rec)
                    else:
                        miss_underpot.append(rec)
                    annotate(img_path, lbl_path,
                             exdir / f"{tag}_MISS_"
                                     f"{'blue' if 'blue' in kinds else 'other'}"
                                     f"__{r['out_file'][:56]}.jpg",
                             pred_fire, f"{tag} MISS | hues={kinds}")
            elif pred_fire:
                pv["fp"] += 1
                rec = {"prov": r["provenance"], "file": r["out_file"],
                       "group": r["group_id"],
                       "best_conf": round(max(p[1] for p in pred_fire), 3)}
                (fp_kitchen if r["provenance"].startswith(("ks_", "kitchen"))
                 else fp_cross).append(rec)
                annotate(img_path, lbl_path,
                         exdir / f"{tag}_FP_{r['provenance']}__{r['out_file'][:52]}.jpg",
                         pred_fire, f"{tag} FALSE ALARM | {r['provenance']}")
            else:
                pv["tn"] += 1

        scene = {}
        for k, v in by_prov.items():
            scene[k] = {
                "images": v["n"], "gt_fire_images": v["gt_fire"],
                "recall": round(v["tp"] / v["gt_fire"], 4) if v["gt_fire"] else None,
                "false_alarm_rate": (round(v["fp"] / (v["fp"] + v["tn"]), 4)
                                     if (v["fp"] + v["tn"]) else None),
                "tp": v["tp"], "fn": v["fn"], "fp": v["fp"], "tn": v["tn"],
            }
        report[f"{tag}_scene"] = scene
        report[f"{tag}_errors"] = {
            "miss_blue_flame": miss_blue,
            "miss_other_fire_candidates(含锅下火/橙焰)": miss_underpot,
            "false_alarm_kitchen_nofire": fp_kitchen,
            "false_alarm_cross_domain": fp_cross,
            "counts": {"miss_blue": len(miss_blue),
                       "miss_other": len(miss_underpot),
                       "fp_kitchen": len(fp_kitchen),
                       "fp_cross": len(fp_cross)},
        }
        print(f"\n[{tag}] 蓝焰漏检 {len(miss_blue)}；其他火漏检 {len(miss_underpot)}；"
              f"厨房无火误报 {len(fp_kitchen)}；跨域无火误报 {len(fp_cross)}")
        for k, v in sorted(scene.items()):
            print(f"    {k:<24} 图={v['images']:<5} 有火={v['gt_fire_images']:<5} "
                  f"召回={v['recall']} 误报率={v['false_alarm_rate']}")

    (args.outdir / "compare_v3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n对照图与报告：{args.outdir}")


if __name__ == "__main__":
    main()
