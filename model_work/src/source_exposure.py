"""Track Commons original works across download sizes; not a scene-independence proof."""
import unicodedata
from collections import Counter, defaultdict
from urllib.parse import unquote, urlparse


def commons_source_key(url):
    parts=urlparse(url)
    if parts.hostname != "commons.wikimedia.org" or not parts.path.startswith("/wiki/"):
        raise ValueError("Expected a Wikimedia Commons file description URL")
    title=unicodedata.normalize("NFC",unquote(parts.path[len("/wiki/"):])).replace("_"," ")
    title=" ".join(title.split())
    if not title.startswith("File:") or not title[5:]:
        raise ValueError("Expected a Commons File: original-work identity")
    name=title[5:]
    return "File:"+name[0].upper()+name[1:]


def training_source_keys(manifest):
    lineage=manifest.get("lineage",manifest.get("new_source_lineage",[]))
    return {commons_source_key(row["source_page"]) for row in lineage if row.get("source_page")}


def diagnostic_summary(rows):
    summaries={}
    for conf,suffix in ((0.25,"025"),(0.5,"050")):
        counts=defaultdict(Counter)
        for row in rows:
            category=("blue_flame" if "blue" in row["intended"] else "fire" if int(row["gt_boxes"]) else "no_fire")
            counts[category]["images"]+=1
            counts[category]["detected_or_false_positive"]+=int(row[f"det_{suffix}"])
            counts[category]["localized"]+=int(row[f"localized_{suffix}"])
        summaries[str(conf)]={k:dict(v) for k,v in counts.items()}
    return summaries


def source_exposure_summary(rows):
    return {name:{"images":len(group),"original_works":len({v["source_key"] for v in group}),
                  "summary":diagnostic_summary(group)}
            for name in ("training_source_in_comparison", "untrained_source_development_diagnostic")
            for group in [[v for v in rows if v["source_exposure"]==name]]}
