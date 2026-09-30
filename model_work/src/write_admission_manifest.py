"""把 Tugas Akhir 的整源排除裁定写入逐图准入清单（8.1-4 要求含审图结论与排除原因）。

只写工作区内的新文件；不改动任何原数据。Tugas 全部 260 张判为 exclude，
依据是本次人眼确认的图库水印/短视频封面样例 + 全源同批次命名这一结构性事实。
Kitchen Safety 保持 pending（还需按拍摄段分层审图）。
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

AUD = Path(r"C:\Users\ASUS\Documents\ChatGPT\SRTP\model_work\out\audit_v3_sources")
SRC = AUD / "source_manifest_v3.csv"
DEST = AUD / "admission_manifest_v3.csv"

TUGAS_REASON = (
    "整源排除：人眼确认存在图库水印与短视频封面。"
    "样例 Api-Blue-116(新增=新)iStock+Credit zoom-zoom+编号91515926；"
    "Api-Blue-87 iStock+Credit kvkirillov+编号856908552；"
    "Api-Blue-127 / Api-Blue-132 pngtree 重复斜向水印；"
    "Api-Orange-99 印尼语短视频封面+企业logo。"
    "项目页 CC BY 4.0 不覆盖嵌入第三方图库内容；全 260 张同批次 `Api-` 前缀，"
    "无法可靠切出干净子集。执行单 8.1-2：0 张合格即明确排除。"
).replace("(新增=新)", "")

# 已人眼确认的具体样例（单独标注，便于复核）
CONFIRMED = {
    "Api-Blue-116-": "iStock 水印 + Credit: zoom-zoom + 编号 91515926（人眼确认）",
    "Api-Blue-87-": "iStock 水印 + Credit: kvkirillov + 编号 856908552（人眼确认）",
    "Api-Blue-127-": "pngtree 重复斜向水印（人眼确认）",
    "Api-Blue-132-": "pngtree 重复斜向水印（人眼确认）",
    "Api-Orange-99-": "印尼语短视频封面 + ADK CV. ANDHY KARYA logo（人眼确认）",
}

rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
print(f"读入逐图清单 {len(rows)} 行")

vc: Counter[str] = Counter()
for r in rows:
    if r["source"] == "tugas":
        hit = next((v for k, v in CONFIRMED.items() if r["orig_path"].find(k) >= 0), None)
        r["review_verdict"] = "exclude"
        r["exclude_reason"] = (f"人眼确认：{hit}；整源排除（详见 AUDIT_REPORT_v3_sources.md §2）"
                               if hit else TUGAS_REASON)
        r["target_split"] = ""
        vc["tugas_exclude"] += 1
    else:
        r["review_verdict"] = "pending_review"
        r["exclude_reason"] = "待按拍摄段分层审图后裁定"
        r["target_split"] = ""
        vc["ks_pending"] += 1

with DEST.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"裁定分布: {dict(vc)}")
print(f"写入: {DEST}")

(AUD / "admission_summary.json").write_text(json.dumps({
    "prepared_by": "DeepSeek (deepseek-flash, DeepSeek Harness) — 仅供参考，须人工复核",
    "source_manifest": str(SRC),
    "admission_manifest": str(DEST),
    "tugas": {
        "verdict": "exclude_entire_source",
        "images": sum(1 for r in rows if r["source"] == "tugas"),
        "approved_for_training": 0,
        "human_confirmed_examples": CONFIRMED,
        "reason": TUGAS_REASON,
    },
    "kitchen_safety": {
        "verdict": "pending_review",
        "images": sum(1 for r in rows if r["source"] == "kitchen_safety"),
        "note": "需按拍摄段分层审图；2 Smoke 已确认混入蒸汽；0 Burner 是炉头局部；"
                "4 safe 是安全物体而非无火区域。",
    },
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"写入: {AUD / 'admission_summary.json'}")
