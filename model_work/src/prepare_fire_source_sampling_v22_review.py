"""Prepare identity-bound v22 review pages without asserting visual review."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from data_integrity import sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )


def prepare(images, previous_images, videos, previous_videos, data, out):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Existing review output')
    current = read(videos / 'comparison_complete.json')
    prior = read(previous_videos / 'comparison_complete.json')
    for folder, complete in [(videos, current), (previous_videos, prior)]:
        for key, name in [
            ('plan_sha256', 'plan.json'),
            ('per_frame_sha256', 'per_frame.jsonl'),
            ('review_manifest_sha256', 'review_manifest.json'),
        ]:
            if sha256(folder / name) != complete[key]:
                raise ValueError('Completed video identity changed')

    old = [
        json.loads(line)
        for line in (previous_videos / 'per_frame.jsonl')
        .read_text(encoding='utf-8')
        .splitlines()
    ]
    new = [
        json.loads(line)
        for line in (videos / 'per_frame.jsonl')
        .read_text(encoding='utf-8')
        .splitlines()
    ]
    if len(old) != 307 or len(new) != 307 or current['total_samples'] != 307:
        raise ValueError('Wrong paired scope')
    identity = [
        'video_index',
        'video_sha256',
        'frame_index',
        'sample_index',
        'time_seconds',
        'width',
        'height',
        'source_pixel_sha256',
    ]
    controls = [model['label'] for model in read(previous_videos / 'plan.json')['models']]
    for before, after in zip(old, new):
        if any(before[key] != after[key] for key in identity):
            raise ValueError('Source frame identity differs')
        if any(
            before['predictions'][label] != after['predictions'][label]
            for label in controls
        ):
            raise ValueError('Reference prediction differs')
    expected = controls + ['v22_best', 'v22_last']
    if [model['label'] for model in read(videos / 'plan.json')['models']] != expected:
        raise ValueError('Unexpected sheet rows')

    complete = read(images / 'comparison_complete.json')
    if sha256(data / 'build_manifest.json') != complete['plan']['data_manifest_sha256']:
        raise ValueError('Data changed')
    selected = [
        row
        for row in read(data / 'build_manifest.json')['lineage']
        if row.get('v20_new_source')
    ]
    if len(selected) != 14 or sum(len(row['boxes_xyxy']) for row in selected) != 4:
        raise ValueError('Wrong development source scope')

    source_labels = ['v16_best', 'v21_best', 'v21_last', 'v22_best', 'v22_last']
    source_roots = {
        label: (images if label.startswith('v22_') else previous_images)
        for label in source_labels
    }
    source_inputs = []
    pages = []
    for label in source_labels:
        summary = read(source_roots[label] / (label + '_new_sources') / 'summary.json')
        if summary != complete['models'][label]['new_training_sources']:
            raise ValueError('Source summary differs')
        if {row['image'] for row in summary['frames']} != {
            row['image'] for row in selected
        }:
            raise ValueError('Source inventory differs')

    out.mkdir(parents=True)

    def save(name, pixels, origin):
        path = out / name
        if not cv2.imwrite(str(path), pixels):
            raise ValueError('Page encoding failed')
        pages.append(dict(file=name, sha256=sha256(path), source=origin))

    rows = []
    tile_width, tile_height = 384, 350
    for source in sorted(selected, key=lambda row: row['image']):
        raw = data / 'images/train' / source['image']
        if sha256(raw) != source['sha256']:
            raise ValueError('Raw source changed')
        row = np.full((tile_height, tile_width * len(source_labels), 3), 24, np.uint8)
        for column, label in enumerate(source_labels):
            path = source_roots[label] / (label + '_new_sources') / source['image']
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Missing annotated source')
            height, width = image.shape[:2]
            scale = min((tile_width - 8) / width, (tile_height - 42) / height)
            resized = cv2.resize(image, (round(width * scale), round(height * scale)))
            top = 38 + (tile_height - 42 - resized.shape[0]) // 2
            left = column * tile_width + (tile_width - resized.shape[1]) // 2
            row[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
            cv2.putText(
                row,
                f'{label} {source["image"]}',
                (column * tile_width + 6, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
            )
            source_inputs.append(dict(path=str(path), sha256=sha256(path)))
        rows.append(row)
    for start in range(0, len(rows), 4):
        page = rows[start : start + 4]
        while len(page) < 4:
            page.append(np.zeros_like(rows[0]))
        save(
            f'new_sources_{start // 4 + 1:02}.jpg',
            np.concatenate(page, axis=0),
            'All 14 training-exposed sources; green GT / blue prediction',
        )

    manifest = read(videos / 'review_manifest.json')
    row_indices = [0, 5, 6, 7, 8]
    for item in manifest['videos']:
        path = videos / item['sheet']
        if sha256(path) != item['sheet_sha256']:
            raise ValueError('Fixed video sheet changed')
        image = cv2.imread(str(path))
        if image.shape != (len(expected) * 460, 1920, 3):
            raise ValueError('Unexpected comparison page geometry')
        save(
            f'video_{item["video_index"]:02}_selected_models.jpg',
            np.concatenate(
                [image[index * 460 : (index + 1) * 460] for index in row_indices],
                axis=0,
            ),
            dict(
                file=str(path),
                sha256=sha256(path),
                rows_zero_based=row_indices,
            ),
        )

    visible_inputs = []
    for label in ['v22_best', 'v22_last']:
        for index in range(1, 4):
            path = images / 'visible_flame' / f'{label}_sheet_{index:02}.jpg'
            if not path.is_file():
                raise ValueError('Missing visible-flame sheet')
            visible_inputs.append(dict(path=str(path), sha256=sha256(path)))

    write(
        out / 'review_inputs.json',
        dict(
            role='Prepared review pages; actual visual inspection remains required',
            script_sha256=sha256(Path(__file__)),
            image_comparison_sha256=sha256(images / 'comparison_complete.json'),
            previous_image_comparison_sha256=sha256(
                previous_images / 'comparison_complete.json'
            ),
            video_comparison_sha256=sha256(videos / 'comparison_complete.json'),
            previous_video_comparison_sha256=sha256(
                previous_videos / 'comparison_complete.json'
            ),
            paired_reference_audit=dict(
                shared_frames=307,
                exact_reference_predictions=307 * len(controls),
                controls=controls,
                identity_fields=identity,
            ),
            new_sources=14,
            new_sources_are_training_exposed=True,
            reserved_sources_predicted=False,
            source_labels=source_labels,
            video_row_labels=[expected[index] for index in row_indices],
            annotated_source_inputs=source_inputs,
            visible_flame_inputs=visible_inputs,
            pages=pages,
        ),
    )
    print(
        'Prepared 4 source pages, 9 fixed video pages and 6 visible-flame sheets; '
        f'{307 * len(controls)} reference predictions match exactly'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        'images',
        'previous-images',
        'videos',
        'previous-videos',
        'data',
        'out',
    ]:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(
        args.images.resolve(),
        args.previous_images.resolve(),
        args.videos.resolve(),
        args.previous_videos.resolve(),
        args.data.resolve(),
        args.out.resolve(),
    )
