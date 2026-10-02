"""Prepare the complete legacy industrial-stove test stratum for supplementary label review."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from data_integrity import sha256, validate_yolo


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Existing label review')
    sources = [r for r in csv.DictReader((a.data / 'manifest.csv').open(encoding='utf-8-sig'))
               if r['split'] == 'test' and r['provenance'] == 'kitchen_stove_fire']
    sources.sort(key=lambda r: r['file'])
    if len(sources) != 14 or sum(int(r['fire_boxes']) for r in sources) != 20:
        raise ValueError('Expected complete 14-image/20-box stratum')
    a.out.mkdir(parents=True)
    rows, tiles, sheets = [], [], []
    for index, row in enumerate(sources, 1):
        image = a.data / 'images/test' / row['file']
        label = a.data / 'labels/test' / (image.stem + '.txt')
        if sha256(image) != row['sha256']:
            raise ValueError('Image identity differs')
        text = label.read_text()
        validate_yolo(text)
        pixels = cv2.imread(str(image))
        if pixels is None:
            raise ValueError('Unreadable source')
        h, w = pixels.shape[:2]
        boxes = []
        for line in text.splitlines():
            if line.strip():
                _, x, y, bw, bh = map(float, line.split())
                boxes.append([(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h])
        if len(boxes) != int(row['fire_boxes']):
            raise ValueError('Label count differs')
        tile = np.full((680, 640, 3), 25, dtype=np.uint8)
        ratio = min(640 / w, 640 / h)
        resized = cv2.resize(pixels, (round(w*ratio), round(h*ratio)))
        rh, rw = resized.shape[:2]
        top, left = 40 + (640-rh)//2, (640-rw)//2
        tile[top:top+rh, left:left+rw] = resized
        cv2.putText(tile, f'Index {index}: original pixels, no predictions', (10, 27), cv2.FONT_HERSHEY_SIMPLEX, .57, (240, 240, 240), 1)
        tiles.append(tile)
        rows.append(dict(index=index, image=str(image.resolve()), image_sha256=row['sha256'],
                         label=str(label.resolve()), label_sha256=sha256(label), size=[w, h],
                         original_boxes_xyxy=boxes, status='pending_visible_flame_review'))
    for start in range(0, len(tiles), 4):
        page = list(tiles[start:start+4])
        while len(page) < 4:
            page.append(np.zeros((680, 640, 3), dtype=np.uint8))
        sheet = np.concatenate([np.concatenate(page[:2], axis=1), np.concatenate(page[2:], axis=1)], axis=0)
        path = a.out / f'sheet_{start//4+1:02}.jpg'
        if not cv2.imwrite(str(path), sheet):
            raise ValueError('Review sheet encoding failed')
        sheets.append(dict(file=path.name, sha256=sha256(path)))
    meta = dict(role='Supplemental semantics review of complete exposed legacy stratum; original test untouched',
                policy='Approximate visible-flame envelope per visible burner; exclude upper pot body and reflected glow. Hold uncertain visibility; no hidden burner/event inference.',
                predictions_shown=False, previous_development_exposure_acknowledged=True,
                source_manifest_sha256=sha256(a.data / 'manifest.csv'), script_sha256=sha256(Path(__file__)),
                images=rows, sheets=sheets, original_images=14, original_boxes=20)
    (a.out / 'review_inputs.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(images=len(rows), original_boxes=20, sheets=sheets), indent=2))


if __name__ == '__main__':
    main()
