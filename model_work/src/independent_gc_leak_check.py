"""Independent read-only dHash and pixel-difference check on raw GC v2 splits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return int(sum(int(pixels[r, c] > pixels[r, c + 1]) << (r * 8 + c) for r in range(8) for c in range(8)))


def compare_pixels(a: Path, b: Path) -> tuple[float, float]:
    with Image.open(a) as image:
        aa = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.int16)
    with Image.open(b) as image:
        bb = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.int16)
    delta = np.abs(aa - bb)
    return float(delta.mean()), float((delta <= 5).mean())


def main(root: Path) -> None:
    train = list((root / "train" / "images").glob("*.jpg"))
    train_hashes = [(dhash(path), path) for path in train]
    indexed: dict[int, list[Path]] = defaultdict(list)
    for value, path in train_hashes:
        indexed[value].append(path)
    results: dict[str, dict] = {}
    for split in ("valid", "test"):
        eval_images = list((root / split / "images").glob("*.jpg"))
        hits = []
        min_distances = []
        for path in eval_images:
            value = dhash(path)
            min_distances.append(min((value ^ train_value).bit_count() for train_value, _ in train_hashes))
            candidates = indexed.get(value, [])
            if not candidates:
                continue
            scored = [(compare_pixels(path, candidate), candidate) for candidate in candidates]
            (mae, close_fraction), match = min(scored, key=lambda item: item[0][0])
            hits.append({
                "eval": path.name,
                "train": match.name,
                "pixel_mae_0_255": round(mae, 3),
                "fraction_pixels_with_absdiff_le_5": round(close_fraction, 4),
            })
        sorted_mae = sorted(item["pixel_mae_0_255"] for item in hits)
        results[split] = {
            "eval_images": len(eval_images),
            "identical_dhash_hits": len(hits),
            "minimum_dhash_distance_le_1": sum(value <= 1 for value in min_distances),
            "minimum_dhash_distance_le_2": sum(value <= 2 for value in min_distances),
            "minimum_dhash_distance_le_4": sum(value <= 4 for value in min_distances),
            "pixel_mae_le_1": sum(value <= 1 for value in sorted_mae),
            "pixel_mae_le_3": sum(value <= 3 for value in sorted_mae),
            "pixel_mae_le_5": sum(value <= 5 for value in sorted_mae),
            "pixel_mae_le_10": sum(value <= 10 for value in sorted_mae),
            "median_pixel_mae": sorted_mae[len(sorted_mae) // 2] if sorted_mae else None,
            "closest_examples": sorted(hits, key=lambda item: item["pixel_mae_0_255"])[:5],
            "farthest_examples": sorted(hits, key=lambda item: item["pixel_mae_0_255"], reverse=True)[:5],
        }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
