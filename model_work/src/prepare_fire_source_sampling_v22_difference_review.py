"""Render v22 samples whose counts differ from the corresponding v21 role."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from data_integrity import sha256


LABELS = ['v21_best', 'v22_best', 'v21_last', 'v22_last']
PAIRS = [('v21_best', 'v22_best'), ('v21_last', 'v22_last')]


def prepare(comparison, out):
    if out.exists():
        raise ValueError('Refusing overwrite')
    plan = json.loads((comparison / 'plan.json').read_text(encoding='utf-8'))
    complete = json.loads(
        (comparison / 'comparison_complete.json').read_text(encoding='utf-8')
    )
    if complete['plan_sha256'] != sha256(comparison / 'plan.json'):
        raise ValueError('Plan changed')
    rows = [
        json.loads(line)
        for line in (comparison / 'per_frame.jsonl')
        .read_text(encoding='utf-8')
        .splitlines()
    ]
    if len(rows) != sum(clip['sampled_frames'] for clip in complete['clips']):
        raise ValueError('Incomplete samples')
    selected = [
        row
        for row in rows
        if any(
            len(row['predictions'][reference]) != len(row['predictions'][candidate])
            for reference, candidate in PAIRS
        )
    ]

    out.mkdir(parents=True)
    rendered = []
    for video_index, source in enumerate(plan['videos'], 1):
        wanted = {
            row['frame_index']: row
            for row in selected
            if row['video_index'] == video_index
        }
        if not wanted:
            continue
        video = Path(source['path'])
        if sha256(video) != source['sha256']:
            raise ValueError('Video changed')
        capture = cv2.VideoCapture(str(video))
        index = 0
        found = set()
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if index in wanted:
                    row = wanted[index]
                    height, width = image.shape[:2]
                    if (
                        (width, height) != (row['width'], row['height'])
                        or hashlib.sha256(
                            np.ascontiguousarray(image).tobytes()
                        ).hexdigest()
                        != row['source_pixel_sha256']
                    ):
                        raise ValueError('Decoded pixels changed')
                    tiles = []
                    ratio = min(640 / width, 420 / height)
                    resized = cv2.resize(
                        image, (round(width * ratio), round(height * ratio))
                    )
                    top = 40 + (420 - resized.shape[0]) // 2
                    left = (640 - resized.shape[1]) // 2
                    for label in LABELS:
                        boxes = row['predictions'][label]
                        tile = np.full((460, 640, 3), 25, np.uint8)
                        tile[
                            top : top + resized.shape[0],
                            left : left + resized.shape[1],
                        ] = resized
                        for box in boxes:
                            x1, y1, x2, y2 = box['xyxy']
                            cv2.rectangle(
                                tile,
                                (
                                    round(x1 * ratio + left),
                                    round(y1 * ratio + top),
                                ),
                                (
                                    round(x2 * ratio + left),
                                    round(y2 * ratio + top),
                                ),
                                (255, 90, 0),
                                2,
                            )
                        cv2.putText(
                            tile,
                            f'{label} v{video_index} sample={row["sample_index"]} '
                            f'boxes={len(boxes)}',
                            (8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (240, 240, 240),
                            1,
                        )
                        tiles.append(tile)
                    rendered.append((row, np.concatenate(tiles, axis=1)))
                    found.add(index)
                index += 1
        finally:
            capture.release()
        if found != set(wanted):
            raise ValueError('Missing selected frames')
        if index != complete['clips'][video_index - 1]['decoded_frames']:
            raise ValueError('Decode count changed')
        if sha256(video) != source['sha256']:
            raise ValueError('Video changed during decode')
    if len(rendered) != len(selected):
        raise ValueError('Incomplete review')

    sheets = []
    for start in range(0, len(rendered), 3):
        page = [image for _, image in rendered[start : start + 3]]
        while len(page) < 3:
            page.append(np.zeros_like(page[0]))
        path = out / f'sheet_{start // 3 + 1:02}.jpg'
        if not cv2.imwrite(str(path), np.concatenate(page, axis=0)):
            raise ValueError('Encoding failed')
        sheets.append(dict(file=path.name, sha256=sha256(path)))
    manifest = dict(
        role='All v22 count-different samples; no count-based correctness inference',
        comparison_sha256=sha256(comparison / 'comparison_complete.json'),
        per_frame_sha256=sha256(comparison / 'per_frame.jsonl'),
        script_sha256=sha256(Path(__file__)),
        compared_pairs=PAIRS,
        labels=LABELS,
        selected_samples=len(selected),
        samples=[row for row, _ in rendered],
        sheets=sheets,
    )
    (out / 'review_inputs.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(f'{len(selected)} count-different samples / {len(sheets)} pages')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.comparison.resolve(), args.out.resolve())
