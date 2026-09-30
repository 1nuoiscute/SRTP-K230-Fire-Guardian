"""第一轮 v3 正式训练：kitchen_vision_v3 微调（执行单 8.3）。

包含本机沙箱必需的 ThreadPool 串行替身补丁。
不做数据改动；输出新 run 目录，exist_ok=False，绝不指向 v2 的 YAML。

用法：
    python src/train_kitchen_vision_v3.py --dry-run    # 只检查
    python src/train_kitchen_vision_v3.py --smoke      # 1 epoch 冒烟（独立 run 名）
    python src/train_kitchen_vision_v3.py              # 正式 80 epoch
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
DATA = Path(r"D:\SRTP_Datasets\kitchen_vision_v3_20260926\data.yaml")
OLD_W = KF / "runs" / "kitchen-fire" / "weights" / "best.pt"
EXPECT_OLD_SHA = "39d84feaef74d0c8ed9263320c7e60a5fc8e201b89a2c07f2b0feb6428dfc3ee"
V2_RUN = KF / "runs" / "kitchen-vision-v2-640"

BASE = dict(
    epochs=80, patience=20, imgsz=640, batch=4, device=0, workers=0, seed=20260926,
    optimizer="auto", lr0=0.001, lrf=0.1,
    hsv_h=0.05, hsv_s=0.5, hsv_v=0.4, degrees=5, translate=0.1, scale=0.2,
    fliplr=0.5, mosaic=1.0, mixup=0.1, copy_paste=0.0, close_mosaic=5,
    project=str(KF / "runs"), exist_ok=False, save=True, plots=True,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch", type=int, default=BASE["batch"])
    ap.add_argument("--workers", type=int, choices=[0], default=0,
                    help="当前环境只支持已验证的 workers=0；线程后端不可用。")
    ap.add_argument("--log", type=Path, default=None,
                    help="训练输出同时写入该日志文件（便于实时 tail）")
    args = ap.parse_args()

    workers = 0
    deviation = None

    ok = True
    sha = hashlib.sha256(OLD_W.read_bytes()).hexdigest()
    print(f"旧权重 SHA-256 : {sha[:24]}  {'OK' if sha == EXPECT_OLD_SHA else '✗ 不符'}")
    ok &= sha == EXPECT_OLD_SHA
    print(f"data.yaml      : {DATA}  存在={DATA.exists()}")
    ok &= DATA.exists()
    if DATA.exists():
        t = DATA.read_text(encoding="utf-8")
        names_ok = all(f"{i}: {n}" in t for i, n in
                       enumerate(["fire", "smoke", "stove", "pan"]))
        print(f"类别顺序       : {'OK 0/1/2/3' if names_ok else '✗ 不符'}")
        ok &= names_ok

    name = "kitchen-vision-v3-smokecheck" if args.smoke else "kitchen-vision-v3-640"
    run_dir = KF / "runs" / name
    print(f"输出 run        : {run_dir}  已存在={run_dir.exists()}")
    ok &= not run_dir.exists()
    print(f"v2 正式 run 未被创建? {not V2_RUN.exists()}")

    import torch
    import importlib.metadata as md
    print(f"环境            : ultralytics {md.version('ultralytics')} / "
          f"torch {torch.__version__} / CUDA={torch.cuda.is_available()}")
    ok &= torch.cuda.is_available()

    cfg = dict(BASE)
    cfg["name"] = name
    cfg["batch"] = args.batch
    cfg["workers"] = workers
    if args.smoke:
        cfg.update(epochs=1, fraction=0.05)

    cmd = (f"YOLO('{OLD_W.name}').train(data=r'{DATA}', "
           + ", ".join(f"{k}={v!r}" for k, v in cfg.items()) + ")")
    print("\n将执行：\n  " + cmd)
    if not ok:
        raise SystemExit("\n前置检查未通过，未启动。")
    if args.dry_run:
        print("\n（dry-run，未启动）")
        return

    from ultralytics import YOLO
    model = YOLO(str(OLD_W))
    print("\n开始训练 ...", flush=True)
    model.train(data=str(DATA), **cfg)
    (run_dir / "run_meta.json").write_text(json.dumps({
        "launched_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 仅供参考",
        "mode": "smoke" if args.smoke else "formal",
        "old_weights": str(OLD_W), "old_weights_sha256": sha,
        "data": str(DATA), "args": cfg,
        "deviation_from_order": deviation,
        "note": "本轮唯一主变量 = 新版数据(v3)；模型结构/类别体系/分辨率/阈值均未变。"
                + (f" 偏离说明：{deviation}" if deviation else
                   " 完全按执行单字面执行（workers=0）。"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成：{run_dir}")


if __name__ == "__main__":
    main()
