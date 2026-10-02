"""Reproducible source caps on existing train data; no labels or splits change."""
import hashlib
from collections import Counter


def select_source_caps(lineage, base_rows, caps, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if not caps or any(not isinstance(k, str) or not k or type(v) is not int or v < 1
                       for k, v in caps.items()):
        raise ValueError("Caps must be positive integer unique-image counts")
    rows = {}
    for row in base_rows:
        digest = row["sha256"]
        if digest in rows:
            raise ValueError("Ambiguous base image identity")
        rows[digest] = row
    names = set()
    pools = {key: [] for key in caps}
    provenance = {}
    for row in lineage:
        if row["image"] in names:
            raise ValueError("Duplicate lineage name")
        names.add(row["image"])
        if row["origin"] != "legacy_train":
            provenance[row["image"]] = "derived:" + row["origin"]
            continue
        base = rows.get(row["sha256"])
        if not base or base["split"] != "train" or base["file"] != row["image"]:
            raise ValueError("Legacy input must match an original train record")
        source = base["provenance"]
        provenance[row["image"]] = source
        if source in pools:
            pools[source].append(row)
    excluded = set()
    selection = {}
    for source, cap in sorted(caps.items()):
        pool = pools[source]
        if not pool or cap >= len(pool):
            raise ValueError("A source cap must actually reduce an existing source")
        def rank(row):
            token = f"{seed}\0{source}\0{row['image']}\0{row['sha256']}"
            return hashlib.sha256(token.encode("utf-8")).hexdigest(), row["image"]
        ordered = sorted(pool, key=rank)
        excluded.update(row["image"] for row in ordered[cap:])
        selection[source] = {"before": len(pool), "after": cap,
                             "kept_images": [row["image"] for row in ordered[:cap]]}
    kept = [dict(row) for row in lineage if row["image"] not in excluded]
    def counts(items, weighted=False):
        result = Counter()
        for row in items:
            result[provenance[row["image"]]] += row["weight"] if weighted else 1
        return dict(sorted(result.items()))
    return kept, {"method": "sha256(seed, provenance, image name, image SHA); lowest ranks",
                  "seed": seed, "caps": dict(sorted(caps.items())), "selection": selection,
                  "unique_before": counts(lineage), "unique_after": counts(kept),
                  "entries_before": counts(lineage, True), "entries_after": counts(kept, True)}


def verify_source_cap_revision(meta, parent, base_rows):
    plan = meta["source_sampling"]
    kept, audit = select_source_caps(parent["lineage"], base_rows, plan["caps"], plan["seed"])
    if meta["lineage"] != kept or plan["audit"] != audit:
        raise ValueError("Source cap lineage or recorded selection differs")
    for field in ("base", "role", "source_review_sha256", "added_sources"):
        if meta.get(field) != parent.get(field):
            raise ValueError("Source capping changed inherited data provenance")
    if meta["unique_paths"] != len(kept) or meta["train_entries"] != sum(r["weight"] for r in kept):
        raise ValueError("Source cap counts differ")
    return audit
