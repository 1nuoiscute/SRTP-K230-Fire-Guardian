"""pHash radius-4 triage of D-Fire splits and available known exposure sources.

Edges/components are review candidates, not proven duplicates or independent scenes.
The graph can merge unrelated similar pictures transitively. Never auto-resplits.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
from audit_blue_candidate_overlap import phash
from data_integrity import sha256
from hamming_index import HammingIndex


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory-dir", type=Path, required=True)
    p.add_argument("--known-data", type=Path, action="append", required=True)
    p.add_argument("--known-dir", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): raise SystemExit("Refusing overwrite")
    inventory_path = a.inventory_dir / "inventory.json"
    summary = json.loads((a.inventory_dir / "summary.json").read_text(encoding="utf-8"))
    if sha256(inventory_path) != summary["inventory.json_sha256"]: raise SystemExit("Inventory changed")
    records = json.loads(inventory_path.read_text(encoding="utf-8"))
    tree = HammingIndex(); parents = list(range(len(records))); sizes = [1]*len(records)
    def find(i):
        while parents[i] != i: parents[i] = parents[parents[i]]; i = parents[i]
        return i
    def union(i, j):
        x, y = find(i), find(j)
        if x == y: return
        if sizes[x] < sizes[y]: x, y = y, x
        parents[y] = x; sizes[x] += sizes[y]
    pairs = []; counts = Counter(); direct_cross = set()
    for i, row in enumerate(records):
        if "phash64" not in row: continue
        value = int(row["phash64"], 16)
        for distance, j in tree.within(value, 4):
            union(i, j); counts["candidate_pairs"] += 1; counts["distance_" + str(distance)] += 1
            cross = row["split"] != records[j]["split"]
            if cross: counts["cross_split_pairs"] += 1; direct_cross.update((i, j))
            if len(pairs) < 10000: pairs.append({"left": j, "right": i, "distance": distance, "cross_split": cross})
        tree.add(value, i)
        if i % 3000 == 0: print(f"Queried {i}/{len(records)} source hashes", flush=True)
    groups = defaultdict(list)
    for i in range(len(records)): groups[find(i)].append(i)
    families = [{"members": indices, "split_counts": dict(Counter(records[i]["split"] for i in indices))}
                for indices in groups.values() if len(indices) > 1]
    cross_families = [r for r in families if len(r["split_counts"]) > 1]
    known = {}; identities = []
    for folder in a.known_data:
        manifest = folder / "manifest.csv"; identities.append({"manifest": str(manifest), "sha256": sha256(manifest)})
        for r in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            image = folder / "images" / r["split"] / r.get("file", r.get("out_file"))
            if r["sha256"] not in known:
                if sha256(image) != r["sha256"]: raise SystemExit("Known manifest image identity changed")
                known[r["sha256"]] = {"image": str(image), "sha256": r["sha256"], "scope": "manifest:" + folder.name + ":" + r["split"]}
    for folder in a.known_dir:
        for image in sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
            digest = sha256(image)
            if digest not in known: known[digest] = {"image": str(image), "sha256": digest, "scope": "exposure_folder:" + folder.name}
    known_rows = list(known.values()); cv2.setNumThreads(1); known_tree = HammingIndex()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, value in enumerate(pool.map(lambda r: phash(Path(r["image"])), known_rows)):
            known_rows[i]["phash64"] = f"{value:016x}"; known_tree.add(value, i)
    exposure_matches = []; exposed_candidate_images = set()
    for i, row in enumerate(records):
        if "phash64" not in row: continue
        for distance, j in known_tree.within(int(row["phash64"], 16), 4):
            exposed_candidate_images.add(i)
            exposure_matches.append({"candidate": i, "known": j, "distance": distance})
    a.out.mkdir(parents=True)
    report = {"role": "similarity triage only; no independent-scene assertion or split mutation", "radius": 4,
              "inventory_sha256": sha256(inventory_path), "script_sha256": sha256(Path(__file__)),
              "hamming_index_sha256": sha256(Path(__file__).with_name("hamming_index.py")),
              "phash_script_sha256": sha256(Path(__file__).with_name("audit_blue_candidate_overlap.py")),
              "images": len(records), "pair_counts": dict(counts), "recorded_pairs": len(pairs),
              "pair_records_capped_at": 10000, "images_with_direct_cross_split_candidate": len(direct_cross),
              "candidate_components": len(families), "candidate_cross_split_components": len(cross_families),
              "largest_component_images": max([len(r["members"]) for r in families] or [1]),
              "known_images": len(known_rows), "known_manifests": identities,
              "known_similarity_candidate_images": len(exposed_candidate_images), "known_similarity_pairs": len(exposure_matches),
              "limitations": ["pHash closeness can be false positive; radius-4 components can merge unrelated images transitively.",
                              "Different pHash does not prove different capture event or kitchen.",
                              "Known coverage only, not generic pretrained/unrecorded data.",
                              "Visual review required; no source labels or split changed."]}
    for name, value in (("pairs.json", pairs), ("components.json", families), ("known_images.json", known_rows), ("known_similarity_pairs.json", exposure_matches)):
        target = a.out / name; target.write_text(json.dumps(value, indent=2), encoding="utf-8")
        report[name + "_sha256"] = sha256(target)
    (a.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "known_manifests"}, indent=2), flush=True)


if __name__ == "__main__": main()
