"""测试：能否用线程后端加速 data loader（不改任何训练超参）。

背景：执行单第 4 节要求 workers=0。在 24 核机器上这会让数据加载只用 1 个核，
GPU 利用率仅 22-31%，80 epoch 约需 12.7 小时。

本脚本只探测「ultralytics 的 DataLoader 是否可安全改为线程 model」，
不启动任何训练、不修改任何配置。若 sandbox 拒绝线程创建则如实报告。
"""
from __future__ import annotations

import time
from pathlib import Path

DATA = Path(r"D:\SRTP_Datasets\kitchen_vision_v3_20260926")

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

print("=== 1. 能否创建线程池（sandbox 是否放行线程）===")
try:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as ex:
        t0 = time.time()
        r = list(ex.map(lambda i: i * i, range(200000)))
    print(f"  OK  ThreadPoolExecutor 可用，200k 任务 {time.time()-t0:.2f}s")
    threads_ok = True
except Exception as e:  # noqa: BLE001
    print(f"  FAIL 线程池不可用: {type(e).__name__}: {e}")
    threads_ok = False

print("\n=== 2. 基准：串行加载并解码 N 张图（模拟 workers=0 的数据加载）===")
import cv2
import numpy as np
imgs = sorted((DATA / "images" / "train").iterdir())[:300]
t0 = time.time()
n = 0
for p in imgs:
    a = cv2.imread(str(p))
    if a is not None:
        n += 1
dt = time.time() - t0
print(f"  串行: {n} 张 / {dt:.2f}s -> {dt/n*1000:.1f} ms/张；"
      f"1486 batch x4 张 ≈ {1486*4*dt/n/60:.1f} 分钟/epoch")

if threads_ok:
    print("\n=== 3. 对照：用 8 线程做同样的加载+解码 ===")

    def load(p):
        try:
            a = cv2.imread(str(p))
            return a is not None
        except Exception:  # noqa: BLE001
            return False

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        got = sum(1 for ok in ex.map(load, imgs[:300]) if ok)
    dt2 = time.time() - t0
    print(f"  8 线程: {got} 张 / {dt2:.2f}s -> {dt2/got*1000:.1f} ms/张；"
          f"1486 batch x4 张 ≈ {1486*4*dt2/got/60:.1f} 分钟/epoch")
    print(f"  => 加速比 {dt/dt2:.2f}x")

print("\n=== 结论 ===")
print("  若线程可用且加速显著，则可在不改任何训练超参的前提下，")
print("  仅把 DataLoader 的进程后端换成线程后端来缩短总时长。")
print("  但这会偏离执行单第 4 节字面的 workers=0，须由用户决定。")
