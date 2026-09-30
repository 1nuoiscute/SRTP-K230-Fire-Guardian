"""Fetch a fixed, source-reviewed set of Commons photos for local label review.

This does not add images to any training split. Check every image and draw
boxes manually before using it in an experiment.
"""
from __future__ import annotations

import csv
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "commons_blue_review_20260927"
SOURCES = [
    ("blue_cardoner", "Gas Stove Burner Blue Flame (28409479226).jpg", "Federico Cardoner / TheMusicianLab.com", "CC BY 2.0"),
    ("blue_bogtar", "Flame from the burner of a gas stove.jpg", "BogTar201213", "CC BY-SA 4.0"),
    ("blue_stovetop", "Stove Top .jpg", "Askaskari2311", "CC BY-SA 4.0"),
    ("blue_ka23", "GasStoveBurner 20220521 181829.jpg", "Ka23 13", "CC BY-SA 4.0"),
    ("blue_oven", "Gas oven flame.jpg", "Billjones94", "CC BY-SA 4.0"),
]
USER_AGENT = "SRTP-kitchen-fire-research/0.1 (local educational image review)"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite: {OUT}")
    OUT.mkdir(parents=True)
    rows = []
    for slug, title, author, license_name in SOURCES:
        page = "https://commons.wikimedia.org/wiki/File:" + urllib.parse.quote(title.replace(" ", "_"))
        query = urllib.parse.urlencode({
            "action": "query", "format": "json", "titles": f"File:{title}",
            "prop": "imageinfo", "iiprop": "url", "iiurlwidth": "1280",
        })
        metadata = json.loads(fetch("https://commons.wikimedia.org/w/api.php?" + query))
        info = next(iter(metadata["query"]["pages"].values()))["imageinfo"][0]
        download_url = info.get("thumburl", info["url"])
        payload = fetch(download_url)
        path = OUT / f"{slug}.jpg"
        path.write_bytes(payload)
        with Image.open(path) as opened:
            width, height = opened.size
        rows.append({"slug": slug, "title": title, "source_page": page,
                     "author": author, "license": license_name,
                     "download_url": download_url,
                     "sha256": hashlib.sha256(payload).hexdigest(),
                     "width": width, "height": height,
                     "status": "unreviewed_not_in_training"})
        print(slug, width, height, len(payload), flush=True)
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
