"""Scan available local text records for exact candidate source and pixel identities.

This is bounded to declared roots and exact strings; it cannot establish a
complete provenance history or identify transformed/re-encoded frames.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "model_work/out/candidate_exposure_audit_20261029"
BOAT = ROOT / "model_work/out/gimbaled_truth_intake_20261029/intake.json"
VLOG = ROOT / "model_work/out/cooking_vlog_intake_20261029/burner_source_review/manifest.json"
ROOTS = [ROOT / "model_work/out", ROOT / "model_work/runs", ROOT / "model_work/data", ROOT / "docs"]
SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".yaml", ".yml", ".md"}
EXCLUDED = [OUT, BOAT.parent, VLOG.parents[1], ROOT / "model_work/out/gimbaled_v24_lineage_audit_20261029"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise ValueError("Refusing overwrite")
    boat = json.loads(BOAT.read_text(encoding="utf8"))
    vlog = json.loads(VLOG.read_text(encoding="utf8"))
    terms = {
        "boat_original_sha256": boat["source_sha256"],
        "boat_original_sha1": "84f4dbeff7172bf4b9b2ea592abf2de6cc5ab71e",
        "boat_source_name": "gimbaled_stove",
        "boat_local_filename": "gimbaled.webm",
        "vlog_original_sha256": vlog["source_sha256"],
        "vlog_original_sha1": "54fc49da22884fdaa3d4e2b6bdca13552899e763",
        "vlog_source_name": "preparation-du-poulet-realisatrice-marcko",
        "vlog_author": "koffimarcko",
    }
    for sample in boat["samples"]:
        terms[f"boat_pixel_{sample['frame_index']}"] = sample["pixel_sha256"]
    for sample in vlog["samples"]:
        if sample["frame_index"] in (4225, 4250, 4275):
            terms[f"vlog_pixel_{sample['frame_index']}"] = sample["pixel_sha256"]
    needles = {k: value.lower().encode("utf8") for k, value in terms.items()}
    hits, errors, scanned, total_bytes = [], [], 0, 0
    for root in ROOTS:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUFFIXES:
                continue
            if any(path.is_relative_to(excluded) for excluded in EXCLUDED):
                continue
            try:
                payload = path.read_bytes().lower()
            except OSError as error:
                errors.append({"path": str(path), "error": str(error)})
                continue
            scanned += 1
            total_bytes += len(payload)
            matches = [name for name, needle in needles.items() if needle in payload]
            if matches:
                hits.append({"path": str(path.relative_to(ROOT)), "matched_terms": matches, "sha256": sha(path)})
    summary = {"role": "bounded exact-string exposure triage; not proof of no prior exposure",
               "inputs": {"boat_intake_sha256": sha(BOAT), "vlog_review_sha256": sha(VLOG)},
               "script_sha256": sha(Path(__file__)), "roots": [str(p.relative_to(ROOT)) for p in ROOTS],
               "excluded_current_results": [str(p.relative_to(ROOT)) for p in EXCLUDED],
               "extensions": sorted(SUFFIXES), "files_scanned": scanned, "bytes_scanned": total_bytes,
               "read_errors": errors, "hits": hits,
               "limitations": ["Exact strings only; different filenames, transformed images, absent records and model pretraining are not covered.",
                               "Matching a metadata record does not establish a model prediction; inspect each context.",
                               "Only 28 boat and 3 vlog decoded-frame hashes checked, not all frames."]}
    OUT.mkdir(parents=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: summary[k] for k in ("files_scanned", "bytes_scanned", "read_errors", "hits")}, indent=2))


if __name__ == "__main__":
    main()
