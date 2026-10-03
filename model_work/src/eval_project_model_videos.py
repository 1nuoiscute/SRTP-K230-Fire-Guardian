"""Compare declared fire candidates on identical decoded development-video samples.

One decode pass and a shared frame for every model. Sampled box counts are not
recognition accuracy or event performance; preserve source frames for review.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from data_integrity import sha256

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_inputs(models_path, folders, registry_path):
    models = json.loads(models_path.read_text(encoding='utf-8'))
    names = [m['label'] for m in models]
    if not names or len(names) != len(set(names)) or any(not re.fullmatch(r'[a-zA-Z0-9_-]+', n) for n in names):
        raise ValueError('Declare distinct filesystem-safe model labels')
    for model in models:
        if sha256(Path(model['weights'])) != model['expected_sha256']:
            raise ValueError('Declared model identity changed')
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    known = {r['sha256'] for r in registry['videos']}
    paths = [p.resolve() for folder in folders for p in sorted(folder.glob('*.mp4'))]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Missing or repeated input videos')
    videos = [dict(path=str(p), sha256=sha256(p)) for p in paths]
    if len({v['sha256'] for v in videos}) != len(videos) or any(v['sha256'] not in known for v in videos):
        raise ValueError('Repeated bytes or unregistered fresh video: use separate frozen fresh-media protocol')
    return models, videos


def longest_empty_sample_run(rows):
    longest = run = 0
    for count in rows:
        run = 0 if count else run + 1
        longest = max(longest, run)
    return longest


def main():
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    p = argparse.ArgumentParser()
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--videos-dir', type=Path, nargs='+', required=True)
    p.add_argument('--registry', type=Path, default=ROOT / 'docs/data_exposure_registry.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sample-fps', type=float, default=1.)
    a = p.parse_args()
    if a.out.exists() or not 0 < a.sample_fps <= 30:
        raise ValueError('Existing output or invalid sampling rate')
    models, videos = load_inputs(a.models, a.videos_dir, a.registry)
    a.out.mkdir(parents=True)
    source_frames = a.out / 'source_review_frames'
    source_frames.mkdir()
    torch.set_num_threads(2)
    config = dict(imgsz=640, conf=.25, iou=.6, max_det=300, device=0, half=False, rect=True)
    plan = dict(role='Project-scoped paired development video predictions; sampled counts are not accuracy',
                started_utc=datetime.now(timezone.utc).isoformat(), configuration=config,
                requested_sample_fps=a.sample_fps, models=models, videos=videos,
                model_listing_sha256=sha256(a.models), registry_sha256=sha256(a.registry),
                script_sha256=sha256(Path(__file__)),
                policy_sha256=sha256(ROOT / 'docs/MODEL_ADOPTION_CRITERIA_20261002.md'),
                review_selection='First/middle/last sample per video, fixed before inference',
                source_pixel_digest='SHA256 of contiguous uint8 decoded BGR bytes; dimensions recorded separately')
    write(a.out / 'plan.json', plan)
    loaded = {m['label']: YOLO(m['weights']) for m in models}
    if any(m.names != {0: 'fire'} for m in loaded.values()):
        raise ValueError('Expected fire-only candidates')
    clips, gallery = [], []
    total_samples = 0
    try:
        with (a.out / 'per_frame.jsonl').open('w', encoding='utf-8') as stream:
            for number, video in enumerate(videos, 1):
                cap = cv2.VideoCapture(video['path'])
                fps, reported = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if not cap.isOpened() or fps <= 0 or reported <= 0:
                    raise ValueError('Invalid video metadata')
                step = max(1, round(fps / a.sample_fps))
                expected_samples = math.ceil(reported / step)
                review_indices = [0, (expected_samples - 1) // 2, expected_samples - 1]
                counts = {n: [] for n in loaded}
                reviewed, index = [], 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if index % step == 0:
                        sample = index // step
                        predictions = {}
                        for name, model in loaded.items():
                            result = model.predict(frame, **config, verbose=False)[0]
                            predictions[name] = [dict(confidence=float(b.conf.item()), xyxy=b.xyxy[0].tolist()) for b in result.boxes]
                            counts[name].append(len(predictions[name]))
                        record = dict(video_index=number, video_sha256=video['sha256'], frame_index=index,
                                      sample_index=sample, time_seconds=index / fps,
                                      width=frame.shape[1], height=frame.shape[0],
                                      source_pixel_sha256=hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest(),
                                      predictions=predictions)
                        stream.write(json.dumps(record) + '\n')
                        if sample in review_indices:
                            path = source_frames / f'video_{number:02}_sample_{sample:04}.png'
                            if not cv2.imwrite(str(path), frame):
                                raise ValueError('Could not save review source')
                            record['review_source_file'] = str(path.relative_to(a.out))
                            record['review_source_file_sha256'] = sha256(path)
                            reviewed.append(record)
                        total_samples += 1
                    index += 1
                cap.release()
                if index != reported or len(reviewed) != len(set(review_indices)) or any(len(v) != expected_samples for v in counts.values()):
                    raise ValueError('Decoded/sample/review inventory differs from planned video metadata')
                # Common source and timestamps across rows; each row is one model.
                rows = []
                for name in loaded:
                    row = np.full((460, 1920, 3), 24, np.uint8)
                    for column, record in enumerate(reviewed):
                        frame = cv2.imread(str(a.out / record['review_source_file']))
                        for box in record['predictions'][name]:
                            b = [round(v) for v in box['xyxy']]
                            cv2.rectangle(frame, tuple(b[:2]), tuple(b[2:]), (255, 0, 0), 3)
                        height, width = frame.shape[:2]
                        ratio = min(630 / width, 390 / height)
                        frame = cv2.resize(frame, (round(width * ratio), round(height * ratio)))
                        h, w = frame.shape[:2]
                        left, top = column * 640 + (640 - w) // 2, 60 + (390 - h) // 2
                        row[top:top+h, left:left+w] = frame
                        cv2.putText(row, f'{name} v{number} t={record["time_seconds"]:.1f}s',
                                    (column * 640 + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1)
                        cv2.putText(row, f'frame {record["frame_index"]}, boxes {len(record["predictions"][name])}',
                                    (column * 640 + 8, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1)
                    rows.append(row)
                sheet = a.out / f'video_{number:02}_comparison.jpg'
                if not cv2.imwrite(str(sheet), np.concatenate(rows, axis=0)):
                    raise ValueError('Comparison sheet encoding failed')
                clip = dict(video_index=number, source_sha256=video['sha256'], fps=fps, step=step,
                            decoded_frames=index, sampled_frames=expected_samples,
                            model_counts={n: dict(frames_with_any_box=sum(c > 0 for c in values),
                                boxes=sum(values), longest_empty_sample_run=longest_empty_sample_run(values),
                                box_count_histogram=dict(Counter(values))) for n, values in counts.items()})
                clips.append(clip)
                gallery.append(dict(video_index=number, sheet=sheet.name, sheet_sha256=sha256(sheet), frames=reviewed))
                stream.flush()
                write(a.out / 'progress.json', dict(completed_clips=clips, completed_samples=total_samples))
                print(f'Finished common video {number}/{len(videos)}, {expected_samples} samples for {len(loaded)} models', flush=True)
        if any(sha256(Path(m['weights'])) != m['expected_sha256'] for m in models) or any(sha256(Path(v['path'])) != v['sha256'] for v in videos):
            raise ValueError('A source or model changed during evaluation')
        write(a.out / 'review_manifest.json', dict(role='Fixed review samples, no dense event truth', videos=gallery))
        write(a.out / 'comparison_complete.json', dict(plan_sha256=sha256(a.out / 'plan.json'),
              completed_utc=datetime.now(timezone.utc).isoformat(), clips=clips, total_samples=total_samples,
              model_inferences=total_samples * len(models), per_frame_sha256=sha256(a.out / 'per_frame.jsonl'),
              review_manifest_sha256=sha256(a.out / 'review_manifest.json'),
              models_sha256={m['label']: m['expected_sha256'] for m in models},
              source_frame_shared=True, automatic_adoption=False,
              limits='Any-box and empty-run statistics include wrong boxes and lack exhaustive truth; not detection accuracy, alarm rate or real-time latency'))
    except Exception as error:
        write(a.out / 'failed.json', dict(error=type(error).__name__, message=str(error), partial_outputs_preserved=True))
        raise


if __name__ == '__main__':
    main()
