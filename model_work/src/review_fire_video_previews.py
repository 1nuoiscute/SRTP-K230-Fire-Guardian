"""Identity-bound start/middle/end contact sheets of generated video predictions."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from data_integrity import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folders', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Existing visual review')
    records, rows = [], []
    for folder in a.folders:
        meta = json.loads((folder / 'summary.json').read_text())
        for clip in meta['clips']:
            video = folder / (Path(clip['video']).stem + '_boxed.mp4')
            digest = sha256(video)
            cap = cv2.VideoCapture(str(video))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or count != clip['sampled'] or count < 1:
                raise ValueError('Preview frame count differs')
            indices = [0, (count - 1) // 2, count - 1]
            row = np.full((430, 1920, 3), 28, dtype=np.uint8)
            for col, index in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('Cannot read review frame')
                height, width = frame.shape[:2]
                ratio = min(630 / width, 380 / height)
                resized = cv2.resize(frame, (round(width * ratio), round(height * ratio)))
                h, w = resized.shape[:2]
                left, top = col * 640 + (640 - w) // 2, 45 + (380 - h) // 2
                row[top:top+h, left:left+w] = resized
                cv2.putText(row, f'{len(records)+1}: sample {index+1}/{count}', (col*640+8, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (245, 245, 245), 1)
            cap.release()
            if sha256(video) != digest:
                raise ValueError('Preview changed during review')
            rows.append(row)
            records.append(dict(source_sha256=clip['sha256'], model_sha256=meta['weights_sha256'],
                                preview=str(video.resolve()), preview_sha256=digest, sampled_frames=count,
                                review_indices=indices))
    a.out.mkdir(parents=True)
    sheets = []
    for start in range(0, len(rows), 3):
        path = a.out / f'sheet_{start//3+1:02}.jpg'
        if not cv2.imwrite(str(path), np.concatenate(rows[start:start+3], axis=0)):
            raise ValueError('Review sheet encoding failed')
        sheets.append(dict(file=path.name, sha256=sha256(path)))
    record = dict(role='Uniform preview samples for manual review; predictions only, no exhaustive per-frame truth',
                  script_sha256=sha256(Path(__file__)), previews=records, sheets=sheets, review_status='awaiting_visual_inspection')
    (a.out / 'review_manifest.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(videos=len(records), sheets=sheets), indent=2))


if __name__ == '__main__':
    main()
