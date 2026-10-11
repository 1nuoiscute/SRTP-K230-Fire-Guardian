"""Review and freeze supplementary visible-flame boxes without editing source labels."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from data_integrity import sha256


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def validated(review):
    inputs = json.loads((review / 'review_inputs.json').read_text(encoding='utf-8'))
    proposal = json.loads((review / 'manual_proposal.json').read_text(encoding='utf-8'))
    rows = inputs['images']
    decisions = proposal['images']
    if inputs['predictions_shown'] or not inputs['previous_development_exposure_acknowledged']:
        raise ValueError('Review exposure scope missing')
    if len(rows) != 14 or len(decisions) != 14 or [r['index'] for r in decisions] != [r['index'] for r in rows]:
        raise ValueError('Incomplete or reordered complete-source review')
    for sheet in inputs['sheets']:
        if sha256(review / sheet['file']) != sheet['sha256']:
            raise ValueError('Original review sheet changed')
    combined = []
    for source, decision in zip(rows, decisions):
        if sha256(Path(source['image'])) != source['image_sha256'] or sha256(Path(source['label'])) != source['label_sha256']:
            raise ValueError('Original image or label changed')
        w, h = source['size']
        if decision['status'] not in ('proposed', 'held') or not decision['note'].strip():
            raise ValueError('Invalid review decision')
        boxes = decision['boxes_xyxy']
        if decision['status'] == 'held' and boxes:
            raise ValueError('Held image must not be scored')
        if decision['status'] == 'proposed' and (not boxes or len(boxes) != len(source['original_boxes_xyxy'])):
            raise ValueError('Expected complete reviewed burner coverage')
        for box in boxes:
            if len(box) != 4 or not all(isinstance(x, (int, float)) and np.isfinite(x) for x in box):
                raise ValueError('Invalid box')
            x1, y1, x2, y2 = box
            if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
                raise ValueError('Box outside image')
        combined.append({**source, **decision})
    return inputs, proposal, combined


def render(review):
    target = review / 'proposal_sheets'
    if target.exists():
        raise ValueError('Existing proposal sheets')
    inputs, proposal, rows = validated(review)
    target.mkdir()
    tiles, sheets = [], []
    for row in rows:
        im = cv2.imread(row['image'])
        if im is None or list(im.shape[:2][::-1]) != row['size']:
            raise ValueError('Image decoding changed')
        for box in row['original_boxes_xyxy']:
            cv2.rectangle(im, tuple(map(round, box[:2])), tuple(map(round, box[2:])), (170,170,170), 1)
        for box in row['boxes_xyxy']:
            cv2.rectangle(im, tuple(map(round, box[:2])), tuple(map(round, box[2:])), (0,255,0), 2)
        tile = np.full((680,640,3), 25, dtype=np.uint8)
        tile[40:680] = im
        title = f"Index {row['index']}: {row['status']} (green flame / gray original)"
        cv2.putText(tile,title,(8,26),cv2.FONT_HERSHEY_SIMPLEX,.51,(240,240,240),1)
        tiles.append(tile)
    for start in range(0,14,4):
        page = tiles[start:start+4]
        page += [np.zeros((680,640,3),np.uint8)] * (4-len(page))
        pixels = np.concatenate([np.concatenate(page[:2],1),np.concatenate(page[2:],1)],0)
        path = target / f'sheet_{start//4+1:02}.jpg'
        if not cv2.imwrite(str(path), pixels):
            raise ValueError('Review sheet encoding failed')
        sheets.append(dict(file=str(path.relative_to(review)),sha256=sha256(path)))
    write(review / 'proposal_render_manifest.json',dict(inputs_sha256=sha256(review/'review_inputs.json'),proposal_sha256=sha256(review/'manual_proposal.json'),sheets=sheets,script_sha256=sha256(Path(__file__))))


def finalize(review, seen):
    target = review / 'review_complete.json'
    if target.exists():
        raise ValueError('Existing completed review')
    inputs, proposal, rows = validated(review)
    rendered = json.loads((review/'proposal_render_manifest.json').read_text())
    if rendered['inputs_sha256'] != sha256(review/'review_inputs.json') or rendered['proposal_sha256'] != sha256(review/'manual_proposal.json') or rendered['script_sha256'] != sha256(Path(__file__)):
        raise ValueError('Proposal differs from rendered review')
    if sorted(seen) != sorted(s['file'] for s in rendered['sheets']) or len(seen) != len(set(seen)):
        raise ValueError('Every rendered sheet must be visually inspected')
    for sheet in rendered['sheets']:
        if sha256(review / sheet['file']) != sheet['sha256']:
            raise ValueError('Proposal review sheet changed')
    for row in rows:
        if row['status'] == 'proposed':
            row['status'] = 'approved_supplemental_approximate'
    approved = [r for r in rows if r['status'] == 'approved_supplemental_approximate']
    result = dict(role=proposal['role'],policy=proposal['policy'],annotations_are_approximate=True,
                  independent_test=False,training_labels_modified=False,original_metrics_replaced=False,
                  previous_prediction_exposure_acknowledged=True,reviewer='Codex visual inspection of every proposal sheet',
                  inputs_sha256=sha256(review/'review_inputs.json'),proposal_sha256=sha256(review/'manual_proposal.json'),
                  render_manifest_sha256=sha256(review/'proposal_render_manifest.json'),
                  preparation_script_sha256=inputs['script_sha256'],review_script_sha256=sha256(Path(__file__)),
                  visually_seen_sheets=rendered['sheets'],reviewed_images=14,approved_images=len(approved),
                  held_images=14-len(approved),approved_boxes=sum(len(r['boxes_xyxy']) for r in approved),images=rows)
    write(target,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('images','visually_seen_sheets')},indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=('render','finalize'))
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--seen-sheet',action='append',default=[])
    a = p.parse_args()
    if a.action == 'render':
        render(a.review)
    else:
        finalize(a.review,a.seen_sheet)
