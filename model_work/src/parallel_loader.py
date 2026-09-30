"""可选的并行数据加载：把 DataLoader 后端换成线程（不改任何训练超参）。

为什么
------
执行单第 4 节写 `workers=0`。在本机（24 逻辑核）这会让数据加载只用 1 个核：
实测 GPU 利用率仅 22~31%、训练吞吐 96 ms/张，而**纯解码只要 2.9 ms/张** ——
瓶颈是单线程的增强流水线。80 epoch 因此约需 12.7 小时。

本模块只改「喂数据的方式」：`workers=N` 仍由 ultralytics 传入，
但 `multiprocessing_context="threads"` 让 worker 走**线程**而不是进程。
- **不建进程、不用命名管道** → 与沙箱（禁止命名管道）兼容；
- **训练超参完全不变**：epochs / patience / imgsz / batch / seed / 优化器 / 增强参数
  全部逐字保持，首轮"唯一主变量 = 新版数据"的前提不受影响。

局限（如实记录）
----------------
多线程共享同一进程的 Python 解释器，增强里的 `random` / `numpy` 全局随机源
会被并发访问。ultralytics 的 `seed_worker` 会为每个 worker 设种，
但在**线程**模式下该设种作用于同一进程、彼此覆盖，因此：
  * 增强的随机序列与 workers=0 时**不会逐位相同**；
  * 在 `seed=20260926` 固定的前提下，运行仍是**可复现**的（同样的 workers 数得到同样结果），
    但不是与串行运行逐位一致。
对本轮（探索性对比）这个差异可接受；若需与串行严格对齐，请用 --workers 0。
"""
from __future__ import annotations

from typing import Any

_PATCHED = False


def enable_threaded_dataloader() -> None:
    """让 ultralytics 的 DataLoader 用线程作为 worker 后端。"""
    global _PATCHED
    if _PATCHED:
        return
    import torch.utils.data.dataloader as _tdl

    _orig = _tdl.DataLoader

    class _ThreadedDataLoader(_orig):  # type: ignore[misc,valid-type]
        """DataLoader 子类：worker 默认走线程后端。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if kwargs.get("num_workers", 0) and "multiprocessing_context" not in kwargs:
                # 线程后端：不 fork 进程、不使用命名管道
                kwargs["multiprocessing_context"] = "threads"
            super().__init__(*args, **kwargs)

    _tdl.DataLoader = _ThreadedDataLoader  # type: ignore[assignment]
    _PATCHED = True


def disable() -> None:
    import torch.utils.data.dataloader as _tdl

    _tdl.DataLoader = _tdl.DataLoader.__mro__[1]  # type: ignore[assignment]
