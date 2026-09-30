"""Kitchen Safety v3 拍摄段分析（执行单 8.2：先分组切分）。

按文件名时间戳切分连续拍摄段；snapshot-* 视为视频抽帧，单独成组。
输出每段的日期、时间跨度、图数、逐类框分布、空标签图数。
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(r"D:\Kitchen Safety.v3i.yolov11")
OUT = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out"
           r"\ks_segments")
IMG_EXT = {".jpg", ".jpeg", ".png"}
NAMES = ["Burner", "Flame", "Smoke", "Spill", "safe"]
GAP_MIN = 10          # 相邻照片间隔 > 10 分钟即视为新拍摄段


def read_label(p: Path):
    boxes = []
    if not p.exists():
        return boxes
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        f = ln.split()
        if len(f) < 5:
            continue
        try:
            boxes.append((int(f[0]), *(float(x) for x in f[1:5])))
        except ValueError:
            continue
    return boxes


def parse_ts(stem: str):
    base = stem.split(".rf.")[0]
    m = re.match(r"^PXL_(\d{8})_(\d{6})", base)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    m = re.match(r"^(\d{8})_(\d{6})", base)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return None


def snapshot_num(stem: str):
    base = stem.split(".rf.")[0]
    m = re.match(r"^snapshot-(\d+)", base)
    return int(m.group(1)) if m else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    files = []
    for split in ("train", "valid", "test"):
        d = ROOT / split / "images"
        if d.exists():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    files.append((split, p))
    print(f"总图 {len(files)}")

    # 逐图信息
    rows = []
    for split, p in files:
        boxes = read_label(ROOT / split / "labels" / f"{p.stem}.txt")
        ts = parse_ts(p.stem)
        sn = snapshot_num(p.stem)
        rows.append({
            "split": split, "file": p.name, "stem": p.stem,
            "ts": ts, "snapshot": sn,
            "classes": "|".join(str(c) for c in sorted({c for c, *_ in boxes})),
            "n_boxes": len(boxes), "empty": int(not boxes),
            "has_flame": int(any(c == 1 for c, *_ in boxes)),
        })

    # 分拍摄段
    dated = [r for r in rows if r["ts"] is not None]
    snap = [r for r in rows if r["snapshot"] is not None]
    other = [r for r in rows if r["ts"] is None and r["snapshot"] is None]
    print(f"有时间戳 {len(dated)}；snapshot 抽帧 {len(snap)}；其他 {len(other)}")
    if other:
        print("  其他样例:", [r['file'][:40] for r in other[:5]])

    dated.sort(key=lambda r: r["ts"])
    seg_id = 0
    prev = None
    for r in dated:
        if prev is None or (r["ts"] - prev).total_seconds() > GAP_MIN * 60:
            seg_id += 1
        r["segment"] = f"seg{seg_id:03d}"
        prev = r["ts"]

    # snapshot 按编号分块（每 20 帧一段）
    sn_groups = defaultdict(list)
    for r in snap:
        sn_groups[r["snapshot"] // 20].append(r)
    for k, grp in sn_groups.items():
        for r in grp:
            r["segment"] = f"snap{k:04d}"
    for r in other:
        r["segment"] = "other"

    # 段统计
    segs = defaultdict(lambda: {"rows": [], "cls": Counter(), "empty": 0})
    for r in rows:
        s = segs[r["segment"]]
        s["rows"].append(r)
        for c, *_ in read_label(ROOT / r["split"] / "labels" / f"{r['stem']}.txt"):
            s["cls"][c] += 1
        if r["empty"]:
            s["empty"] += 1

    print(f"\n共 {len(segs)} 个拍摄段")
    print(f"{'段':<10}{'起止时间':<34}{'图数':>5}{'空标签':>7}{'含Flame':>8}  逐类框")
    print("-" * 108)
    summary = []
    for name in sorted(segs):
        s = segs[name]
        rs = s["rows"]
        tss = [r["ts"] for r in rs if r["ts"]]
        span = (f"{min(tss):%m-%d %H:%M}~{max(tss):%H:%M}" if tss else "snapshot/无时间戳")
        nfl = sum(r["has_flame"] for r in rs)
        cls = " ".join(f"{NAMES[c]}={v}" for c, v in sorted(s["cls"].items()))
        print(f"{name:<10}{span:<34}{len(rs):>5}{s['empty']:>7}{nfl:>8}  {cls}")
        summary.append({
            "segment": name, "images": len(rs), "empty_label": s["empty"],
            "with_flame": nfl, "boxes": {NAMES[c]: v for c, v in sorted(s["cls"].items())},
            "splits": dict(Counter(r["split"] for r in rs)),
            "qid": sorted({r["split"] for r in rs}),
        })

    with (OUT / "segment_rows.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["ts"] = r["ts"].isoformat() if r["ts"] else ""
            w.writerow(rr)

    (OUT / "segments.json").write_text(json.dumps({
        "analyzed_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 仅供参考",
        "root": str(ROOT), "total_images": len(files),
        "gap_minutes": GAP_MIN, "n_segments": len(segs),
        "segments": summary,
        "notes": [
            "拍摄段按文件名时间戳相邻间隔 > %d 分钟切分；snapshot-* 无时间戳，按编号每 20 帧一段。" % GAP_MIN,
            "同一段内应整体分配到一个 split，避免相邻帧跨 split。",
            "段是最小不可分单位；段的切分仍需人工确认边界（例如跨日期是否同一厨房）。",
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出：{OUT}")


if __name__ == "__main__":
    main()
