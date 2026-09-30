"""基线评测：对冻结套件 test_suite_v1 跑 4 类 best.pt，并按泄漏级别分层报告。

为什么分层：套件里只有 cctv_clean 是真正无重叠的；val 与 train 两级已泄漏。
把三层的数字分开报，才能看出"看起来很好"有多少来自记忆。

只读：不修改任何既有文件；结果写入 --outdir 指定的新目录。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

# Ultralytics 缓存标签时会开 multiprocessing.pool.ThreadPool，而本机沙箱禁止进程间
# 命名管道（CreateFile 报 WinError 5）。注意 dataset.py 里 `ThreadPool` 与 `NUM_THREADS`
# 都是直接 import 进模块命名空间的，所以要 patch 该模块自己的名字，并提供一个串行替身。
import ultralytics.data.dataset as _ul_dataset  # noqa: E402
from ultralytics.utils import NUM_THREADS as _UL_NUM_THREADS  # noqa: E402


class _SerialPool:
    """最小串行替身：只用 map(starmap) 接口，不开任何进程或管道。"""

    def __init__(self, *_a, **_kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]

    def map(self, func, iterable):
        return [func(x) for x in iterable]

    def imap(self, func, iterable):
        return iter([func(x) for x in iterable])


_ul_dataset.ThreadPool = _SerialPool
_ul_dataset.NUM_THREADS = 0
print(
    f"[init] 已把 ultralytics ThreadPool 替换为串行实现 "
    f"（原 NUM_THREADS={_UL_NUM_THREADS}）；沙箱禁止命名管道。"
)

from ultralytics import YOLO  # noqa: E402

NAMES = ["fire", "smoke", "stove", "pan"]


def md5(p: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_manifest(suite: Path) -> list[dict]:
    rows = []
    with (suite / "manifest.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["boxes"] = json.loads(r["boxes"])
            r["empty"] = r["empty"] == "1"
            rows.append(r)
    return rows


def write_subset(suite: Path, rows: list[dict], dest: Path) -> Path:
    """把某个子集写成一份独立的 YOLO 数据集（软链接/复制标签，图片用绝对路径列表）。"""
    (dest / "images").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)
    for r in rows:
        src_img = suite / "images" / r["file"]
        dst_img = dest / "images" / r["file"]
        if not dst_img.exists():
            try:
                os.link(src_img, dst_img)  # 硬链接，省空间
            except OSError:
                import shutil

                shutil.copy2(src_img, dst_img)
        src_lbl = suite / "labels" / (Path(r["file"]).stem + ".txt")
        dst_lbl = dest / "labels" / (Path(r["file"]).stem + ".txt")
        if not dst_lbl.exists():
            dst_lbl.write_text(
                src_lbl.read_text(encoding="utf-8") if src_lbl.exists() else "",
                encoding="utf-8",
            )
    (dest / "data.yaml").write_text(
        f"path: {str(dest).replace(chr(92), '/')}\n"
        "train: images\nval: images\ntest: images\nnc: 4\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)),
        encoding="utf-8",
    )
    return dest / "data.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--suite",
        type=Path,
        default=Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\data\test_suite_v1"),
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out\baseline"),
    )
    ap.add_argument("--imgsz", type=int, default=576)
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO val 的 conf 下限")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU（板端用 0.6）")
    ap.add_argument(
        "--weights",
        type=Path,
        default=Path(
            r"C:\Users\ASUS\Documents\Codex\kitchen-fire-detection\runs\kitchen-fire\weights\best.pt"
        ),
    )
    ap.add_argument("--tag", default="best_pt_576")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.suite)
    print(f"套件 {args.suite.name}: {len(rows)} 张")

    # 只保留含 0/2/3 类框、或空标签的图（smoke 正例单列，避免用它污染 4 类指标）
    tiers: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        tiers[r["leaked"]].append(r)

    print(f"权重: {args.weights}")
    print(f"  SHA-256 = {__import__('hashlib').sha256(args.weights.read_bytes()).hexdigest()}")
    model = YOLO(str(args.weights))
    print(f"  类别 = {model.names}  nc = {model.model.nc}")

    report: dict = {
        "evaluated_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 仅供参考，需人工复核",
        "date": "2026-09-25",
        "weights": str(args.weights),
        "weights_sha256": __import__("hashlib").sha256(args.weights.read_bytes()).hexdigest(),
        "weights_epoch": None,
        "imgsz": args.imgsz,
        "conf_floor": args.conf,
        "nms_iou": args.iou,
        "suite": str(args.suite),
        "tiers": {},
    }

    for tier in sorted(tiers):
        subset_rows = tiers[tier]
        # 4 类指标：排除 smoke 正例图（旧模型把 smoke 当背景，会人为拉低召回）
        eval_rows = [
            r
            for r in subset_rows
            if not any(k == "1" for k in r["boxes"])
        ]
        skipped = len(subset_rows) - len(eval_rows)
        if len(eval_rows) < 5:
            print(f"\n[{tier}] 可评图数 {len(eval_rows)} < 5，跳过")
            continue
        sub_dir = args.outdir / f"_subset_{tier}"
        yaml_path = write_subset(args.suite, eval_rows, sub_dir)

        print(f"\n{'='*70}")
        print(f"[{tier}] 评测 {len(eval_rows)} 张（跳过含 smoke 正例的 {skipped} 张）")
        print(f"{'='*70}")

        metrics = model.val(
            data=str(yaml_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            workers=0,
            plots=False,
            verbose=False,
            project=str(args.outdir / "_val_runs"),
            name=tier,
            exist_ok=True,
        )

        per_class = {}
        try:
            import numpy as np

            names = {int(k): v for k, v in dict(metrics.names).items()}
            box = metrics.box
            # 关键：ap_class_index 是「类别 id」，而 p/r/ap50/ap 是只对齐到
            # ap_class_index 的「稠密数组」；maps 才是按类别 id 索引的（长度 = nc）。
            present_ids = [int(i) for i in getattr(box, "ap_class_index", [])]
            maps = np.asarray(getattr(box, "maps", []))
            p_arr = np.asarray(box.p)
            r_arr = np.asarray(box.r)
            ap50_arr = np.asarray(box.ap50)
            for dense, cid in enumerate(present_ids):
                cname = str(names.get(cid, cid))
                per_class[cname] = {
                    "class_id": cid,
                    "precision": round(float(p_arr[dense]), 5),
                    "recall": round(float(r_arr[dense]), 5),
                    "mAP50": round(float(ap50_arr[dense]), 5),
                    "mAP50-95": round(float(maps[cid]), 5)
                    if maps.size > cid
                    else None,
                }
            for cid, cname in names.items():
                per_class.setdefault(
                    str(cname),
                    {
                        "class_id": cid,
                        "precision": None,
                        "recall": None,
                        "mAP50": None,
                        "mAP50-95": round(float(maps[cid]), 5) if maps.size > cid else None,
                        "note": "该子集无此类别 GT",
                    },
                )
        except Exception as e:  # noqa: BLE001
            per_class = {"error": f"{type(e).__name__}: {e}"}

        overall = {
            "images": len(eval_rows),
            "precision": round(float(metrics.box.mp), 5),
            "recall": round(float(metrics.box.mr), 5),
            "mAP50": round(float(metrics.box.map50), 5),
            "mAP50-95": round(float(metrics.box.map), 5),
            "per_class": per_class,
            "skipped_smoke_positive_images": skipped,
        }
        report["tiers"][tier] = overall

        print(f"  总体: P={overall['precision']} R={overall['recall']} "
              f"mAP50={overall['mAP50']} mAP50-95={overall['mAP50-95']}")
        for cname, m in per_class.items():
            if not isinstance(m, dict):
                continue
            if m.get("precision") is None:
                print(f"    {cname:<7} （该子集无此类 GT）")
            else:
                print(f"    {cname:<7} P={m['precision']:<9} R={m['recall']:<9} "
                      f"mAP50={m['mAP50']:<9} mAP50-95={m['mAP50-95']}")

    # ---- 场景级误报/漏检（在 clean 层，阈值按板端 0.5）----
    print(f"\n{'='*70}")
    print("[场景级] 火焰有无判定（conf=0.5，仅 clean 层）")
    print(f"{'='*70}")
    scene = {}
    clean = [r for r in rows if r["leaked"] == "clean"]
    if clean:
        sub_dir = args.outdir / "_subset_scene_clean"
        yaml_path = write_subset(args.suite, clean, sub_dir)
        results = model.predict(
            source=[str(args.suite / "images" / r["file"]) for r in clean],
            imgsz=args.imgsz,
            conf=0.5,
            iou=args.iou,
            verbose=False,
            stream=True,
        )
        tp = fp = fn = tn = 0
        details = []
        for r, res in zip(clean, results):
            has_gt_fire = r["boxes"].get("0", 0) > 0
            has_pred_fire = False
            best = 0.0
            if res.boxes is not None and len(res.boxes):
                for c, cf in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                    if int(c) == 0 and cf >= 0.5:
                        has_pred_fire = True
                        best = max(best, cf)
            if has_gt_fire and has_pred_fire:
                tp += 1
            elif has_gt_fire and not has_pred_fire:
                fn += 1
                details.append(("miss", r["file"], round(best, 3), r["boxes"]))
            elif not has_gt_fire and has_pred_fire:
                fp += 1
                details.append(("false_alarm", r["file"], round(best, 3), r["boxes"]))
            else:
                tn += 1
        scene = {
            "tier": "clean",
            "images": len(clean),
            "gt_with_fire": tp + fn,
            "gt_without_fire": fp + tn,
            "true_positive_imgs": tp,
            "false_negative_imgs": fn,
            "false_positive_imgs": fp,
            "true_negative_imgs": tn,
            "image_level_recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "image_level_false_alarm_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
            "examples": details[:20],
        }
        report["scene_level_clean"] = scene
        print(f"  有火图 {tp+fn}：检出 {tp}，漏检 {fn}  -> 图像级召回 {scene['image_level_recall']}")
        print(f"  无火图 {fp+tn}：正确 {tn}，误报 {fp}  -> 图像级误报率 {scene['image_level_false_alarm_rate']}")

    (args.outdir / f"baseline_{args.tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n报告已写入：{args.outdir / f'baseline_{args.tag}.json'}")


if __name__ == "__main__":
    main()
