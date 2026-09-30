"""第一轮正式训练：kitchen_vision_v2 微调（执行单第 4 节）。

包含本机沙箱必需的 ThreadPool 串行替身补丁（沙箱禁止进程间命名管道）。
不做任何数据改动；只读 data.yaml，输出新 run 目录，exist_ok=False。

用法：
    python src/train_kitchen_vision_v2.py            # 正式训练
    python src/train_kitchen_vision_v2.py --dry-run  # 只打印将执行的命令与检查
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# --- 沙箱补丁：必须在 import ultralytics 之前 ---
import ultralytics.data.dataset as _ul_dataset  # noqa: E402


class _SerialPool:
    """multiprocessing.pool.ThreadPool 的串行替身（沙箱禁止命名管道）。"""

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

KF = Path(r"C:\Users\ASUS\Documents\Codex\kitchen-fire-detection")
DATA = Path(r"D:\SRTP_Datasets\kitchen_vision_v2_20260926\data.yaml")
OLD_W = KF / "runs" / "kitchen-fire" / "weights" / "best.pt"
RUN_NAME = "kitchen-vision-v2-640"
EXPECT_OLD_SHA = "39d84feaef74d0c8ed9263320c7e60a5fc8e201b89a2c07f2b0feb6428dfc3ee"

TRAIN_ARGS = dict(
    epochs=80, patience=20, imgsz=640, batch=4, device=0, workers=0, seed=20260926,
    optimizer="auto", lr0=0.001, lrf=0.1,
    hsv_h=0.05, hsv_s=0.5, hsv_v=0.4, degrees=5, translate=0.1, scale=0.2,
    fliplr=0.5, mosaic=1.0, mixup=0.1, copy_paste=0.0, close_mosaic=5,
    project=str(KF / "runs"), name=RUN_NAME, exist_ok=False, save=True, plots=True,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=TRAIN_ARGS["batch"])
    args = ap.parse_args()

    # --- 前置检查 ---
    ok = True
    sha = hashlib.sha256(OLD_W.read_bytes()).hexdigest()
    print(f"旧权重 SHA-256 : {sha}")
    if sha != EXPECT_OLD_SHA:
        print("  ✗ 与执行单记录的哈希不一致，停止")
        ok = False
    else:
        print("  ✓ 与执行单记录一致")
    print(f"data.yaml      : {DATA}  存在={DATA.exists()}")
    if not DATA.exists():
        ok = False
    run_dir = KF / "runs" / RUN_NAME
    print(f"输出 run        : {run_dir}  已存在={run_dir.exists()}")
    if run_dir.exists():
        print("  ✗ 输出目录已存在（exist_ok=False 会报错），停止")
        ok = False

    import torch
    import importlib.metadata as md
    print(f"环境            : ultralytics {md.version('ultralytics')} / torch {torch.__version__} "
          f"/ CUDA={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        ok = False

    yaml_txt = DATA.read_text(encoding="utf-8") if DATA.exists() else ""
    names_ok = all(f"{i}: {n}" in yaml_txt for i, n in enumerate(["fire", "smoke", "stove", "pan"]))
    print(f"类别顺序         : {'✓ 0 fire / 1 smoke / 2 stove / 3 pan' if names_ok else '✗ 不符'}")
    ok &= names_ok

    cmd = (f"YOLO(r'{OLD_W}').train(data=r'{DATA}', "
           + ", ".join(f"{k}={v!r}" for k, v in {**TRAIN_ARGS, 'batch': args.batch}.items())
           + ")")
    print("\n将执行的训练调用：")
    print("  " + cmd)

    if not ok:
        raise SystemExit("\n前置检查未通过，未启动训练。")
    if args.dry_run:
        print("\n（dry-run：未启动训练）")
        return

    from ultralytics import YOLO
    model = YOLO(str(OLD_W))
    print("\n开始训练 ...")
    model.train(data=str(DATA), **{**TRAIN_ARGS, "batch": args.batch})

    rep = KF / "runs" / RUN_NAME / "run_meta.json"
    rep.write_text(json.dumps({
        "launched_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 仅供参考",
        "old_weights": str(OLD_W), "old_weights_sha256": sha,
        "data": str(DATA), "args": {**TRAIN_ARGS, "batch": args.batch},
        "note": "本 run 只更换数据一个变量；结构与主要训练设置沿用旧 run。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成。run 目录：{run_dir}")


if __name__ == "__main__":
    main()
