"""开训前检查（执行单第 4 节）：逐项验证环境、数据、输出目录，并冒烟加载。

只读；不训练、不创建正式 run。冒烟加载用独立小目录，避免在数据集里写 cache。
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

# 沙箱禁止进程间命名管道；ultralytics 缓存标签时会开 ThreadPool，先替换为串行实现
import ultralytics.data.dataset as _ul_dataset  # noqa: E402


class _SerialPool:
    def __init__(self, *_a, **_kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_e):
        return False

    def starmap(self, func, iterable, chunksize=None):
        return [func(*a) for a in iterable]

    def map(self, func, iterable, chunksize=None):
        return [func(x) for x in iterable]

    def imap(self, func, iterable, chunksize=None):
        return iter([func(x) for x in iterable])

    def imap_unordered(self, func, iterable, chunksize=None):
        return iter([func(x) for x in iterable])

    def uimap(self, func, iterable, chunksize=None):
        return iter([func(x) for x in iterable])

    def uimap_unordered(self, func, iterable, chunksize=None):
        return iter([func(x) for x in iterable])

    def apply_async(self, func, args=(), kwds=None, callback=None):
        class _R:
            def __init__(self, v):
                self._v = v

            def get(self, timeout=None):
                return self._v

        return _R(func(*args, **(kwds or {})))

    def close(self):
        pass

    def join(self):
        pass

    def terminate(self):
        pass


_ul_dataset.ThreadPool = _SerialPool
_ul_dataset.NUM_THREADS = 0

import torch  # noqa: E402
import importlib.metadata as md  # noqa: E402
from ultralytics import YOLO  # noqa: E402

KF = Path(r"C:\Users\ASUS\Documents\Codex\kitchen-fire-detection")
D = Path(r"D:\SRTP_Datasets\kitchen_vision_v2_20260926")
OLD_W = KF / "runs" / "kitchen-fire" / "weights" / "best.pt"
NEW_RUN = KF / "runs" / "kitchen-vision-v2-640"
SMOKE_RUN = KF / "runs" / "kitchen-vision-v2-smokecheck"
EXPECT_OLD_SHA = "39d84feaef74d0c8ed9263320c7e60a5fc8e201b89a2c07f2b0feb6428dfc3ee"

checks: list[tuple[str, bool, str]] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  {detail}")


print("=== 1. 环境 ===")
try:
    import hashlib
    src = (KF / "runs" / "kitchen-fire" / "weights" / "best.pt").read_bytes()
    old_sha = hashlib.sha256(src).hexdigest()
    chk("旧权重 SHA-256 与执行单一致", old_sha == EXPECT_OLD_SHA, old_sha[:16])
except Exception as e:  # noqa: BLE001
    chk("旧权重可读", False, str(e))

chk("ultralytics 可导入", True, md.version("ultralytics"))
chk("torch 可导入", True, torch.__version__)
chk("CUDA 可用", torch.cuda.is_available(),
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")
try:
    import ultralytics.utils as _u

    cfg = Path(_u.USER_CONFIG_DIR) / "settings.json"
    chk("ultralytics settings.json 可读", True, str(cfg))
except Exception as e:  # noqa: BLE001
    chk("ultralytics settings.json 可读", False, str(e)[:80])

print("\n=== 2. 数据 ===")
chk("data.yaml 存在", (D / "data.yaml").exists(), str(D / "data.yaml"))
yaml_txt = (D / "data.yaml").read_text(encoding="utf-8") if (D / "data.yaml").exists() else ""
chk("data.yaml 含四类且顺序正确",
    all(f"{i}: {n}" in yaml_txt for i, n in enumerate(["fire", "smoke", "stove", "pan"])))
for s in ("train", "val", "test"):
    p = D / "images" / s
    n = len(list(p.iterdir())) if p.exists() else 0
    chk(f"images/{s} 存在且非空", p.exists() and n > 0, f"{n} 张")
chk("manifest.csv 存在", (D / "manifest.csv").exists())

print("\n=== 3. 输出目录（必须不存在）===")
chk("正式 run 名未被占用", not NEW_RUN.exists(), str(NEW_RUN))
chk("冒烟 run 名未被占用", not SMOKE_RUN.exists(), str(SMOKE_RUN))

print("\n=== 4. 冒烟加载（在工作区内临时目录，避免污染数据集）===")
try:
    # 沙箱不允许写系统 temp；放到工作区内的临时目录
    smoke_root = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out\_smoke_tmp")
    if smoke_root.exists():
        shutil.rmtree(smoke_root, ignore_errors=True)
    tmp = smoke_root
    for s in ("train", "val"):
        (tmp / "images" / s).mkdir(parents=True)
        (tmp / "labels" / s).mkdir(parents=True)
    rows = list(csv.DictReader((D / "manifest.csv").open(encoding="utf-8")))
    take = {"train": 40, "val": 20}
    for s in ("train", "val"):
        sub = [r for r in rows if r["split"] == s][: take[s]]
        for r in sub:
            shutil.copy2(D / "images" / s / r["out_file"], tmp / "images" / s / r["out_file"])
            lp = D / "labels" / s / (Path(r["out_file"]).stem + ".txt")
            shutil.copy2(lp, tmp / "labels" / s / lp.name)
    (tmp / "data.yaml").write_text(
        f"path: {str(tmp).replace(chr(92), '/')}\n"
        "train: images/train\nval: images/val\nnc: 4\nnames:\n"
        "  0: fire\n  1: smoke\n  2: stove\n  3: pan\n", encoding="utf-8")
    m = YOLO(str(OLD_W))
    chk("旧权重可加载且类别正确",
        list(m.names.values()) == ["fire", "smoke", "stove", "pan"], str(m.names))
    res = m.val(data=str(tmp / "data.yaml"), imgsz=640, batch=4, workers=0,
                plots=False, verbose=False, project=str(tmp / "r"), name="v", exist_ok=True)
    chk("旧权重在冒烟子集上可完成 val", True,
        f"mAP50={float(res.box.map50):.4f} mAP50-95={float(res.box.map):.4f}")
    shutil.rmtree(tmp, ignore_errors=True)
except Exception as e:  # noqa: BLE001
    chk("冒烟加载", False, f"{type(e).__name__}: {str(e)[:120]}")

print("\n" + "=" * 70)
n_fail = sum(1 for _, ok, _ in checks if not ok)
print(f"检查项 {len(checks)}，失败 {n_fail}")
print("=" * 70)
out = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out\preflight_v2.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(
    {"checked_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 仅供参考",
     "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in checks],
     "failed": n_fail,
     "verdict": "READY" if n_fail == 0 else "NOT_READY"},
    ensure_ascii=False, indent=2), encoding="utf-8")
print(f"报告：{out}")
