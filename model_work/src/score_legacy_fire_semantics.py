"""Rescore identity-bound cached predictions against supplementary reviewed flame envelopes."""
import argparse
import json
import re
from pathlib import Path
import cv2
import numpy as np
from data_integrity import iou, matches, sha256
from legacy_fire_semantic_review import validated, write


def aggregate(rows):
    counts = {k: sum(r[k] for r in rows) for k in ('tp','fp','fn')}
    d = 2*counts['tp'] + counts['fp'] + counts['fn']
    return dict(**counts,micro_f1=2*counts['tp']/d if d else None)


def score(review, cache, labels, out):
    if out.exists():
        raise ValueError('Existing semantics diagnostic; no overwrite')
    approved = json.loads((review/'review_complete.json').read_text(encoding='utf-8'))
    inputs, proposal, rows = validated(review)
    for key, name in [('inputs_sha256','review_inputs.json'),('proposal_sha256','manual_proposal.json'),('render_manifest_sha256','proposal_render_manifest.json')]:
        if approved[key] != sha256(review/name):
            raise ValueError('Completed review identity changed')
    if approved['review_script_sha256'] != sha256(Path(__file__).with_name('legacy_fire_semantic_review.py')) or approved['preparation_script_sha256'] != sha256(Path(__file__).with_name('prepare_legacy_fire_semantic_audit.py')):
        raise ValueError('Review source changed')
    for row in rows:
        if row['status'] == 'proposed':
            row['status'] = 'approved_supplemental_approximate'
    if rows != approved['images'] or approved['independent_test'] or not approved['annotations_are_approximate']:
        raise ValueError('Supplemental scope or reviewed rows differs')
    for sheet in approved['visually_seen_sheets']:
        if sha256(review/sheet['file']) != sheet['sha256']:
            raise ValueError('Reviewed overlay changed')
    summary = json.loads((cache/'summary.json').read_text(encoding='utf-8'))
    if summary['configuration'] != dict(imgsz=640,conf=.25,nms_iou=.6,match_iou=.5):
        raise ValueError('Cached prediction configuration differs')
    identities = json.loads((cache/'local_inputs.json').read_text(encoding='utf-8'))
    by_name = {r['file']:r for r in identities}
    if len(by_name) != len(identities):
        raise ValueError('Duplicate source inputs')
    for row in rows:
        original = by_name[Path(row['image']).name]
        if original['sha256'] != row['image_sha256'] or original['label_sha256'] != row['label_sha256'] or original['gt'] != row['original_boxes_xyxy']:
            raise ValueError('Cached original annotation differs')
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError('Duplicate or empty model selection')
    for label in labels:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',label) or label not in summary['models']:
            raise ValueError('Invalid model label')
    scored = [r for r in rows if r['status'] == 'approved_supplemental_approximate']
    out.mkdir(parents=True)
    result = dict(role='Supplemental approximate visible-flame localization on exposed legacy development source',
                  independent_test=False,original_metrics_replaced=False,training_labels_modified=False,
                  configuration=summary['configuration'],approximate_geometry_sensitivity_iou=[.3,.5],
                  reviewed_images=14,scored_images=len(scored),scored_boxes=sum(len(r['boxes_xyxy']) for r in scored),
                  held_indices=[r['index'] for r in rows if r['status']=='held'],
                  review_complete_sha256=sha256(review/'review_complete.json'),source_summary_sha256=sha256(cache/'summary.json'),
                  source_inputs_sha256=sha256(cache/'local_inputs.json'),script_sha256=sha256(Path(__file__)),models={})
    for label in labels:
        path = cache/(label+'_local_predictions.json')
        predictions = json.loads(path.read_text(encoding='utf-8'))
        indexed = {r['image']:r for r in predictions}
        if len(indexed) != len(predictions) or set(indexed) != set(by_name):
            raise ValueError('Prediction image inventory differs')
        details, tiles = [], []
        for row in scored:
            name = Path(row['image']).name
            record = indexed[name]
            boxes = [r['xyxy'] for r in record['predictions']]
            for box in boxes:
                if len(box)!=4 or not all(np.isfinite(v) for v in box) or not box[0]<box[2] or not box[1]<box[3]:
                    raise ValueError('Invalid cached prediction geometry')
            if any(r['confidence'] < .24995 or r['confidence'] > 1 for r in record['predictions']):
                raise ValueError('Cached confidence outside fixed selection')
            original = matches(row['original_boxes_xyxy'],boxes)
            if original != {k:record[k] for k in ('tp','fp','fn')}:
                raise ValueError('Cached original matching differs')
            revised = {str(t):matches(row['boxes_xyxy'],boxes,threshold=t) for t in (.3,.5)}
            details.append(dict(index=row['index'],image=name,original=original,supplemental=revised,
                                best_prediction_iou_per_visible_flame=[max((iou(g,b) for b in boxes),default=0) for g in row['boxes_xyxy']],
                                supplemental_vs_original_best_iou=[max(iou(g,b) for b in row['original_boxes_xyxy']) for g in row['boxes_xyxy']],
                                visible_flame_boxes=row['boxes_xyxy'],original_boxes=row['original_boxes_xyxy'],predictions=record['predictions']))
            im = cv2.imread(row['image'])
            for b in row['original_boxes_xyxy']:
                cv2.rectangle(im,tuple(map(round,b[:2])),tuple(map(round,b[2:])),(170,170,170),1)
            for b in row['boxes_xyxy']:
                cv2.rectangle(im,tuple(map(round,b[:2])),tuple(map(round,b[2:])),(0,255,0),2)
            for b in boxes:
                cv2.rectangle(im,tuple(map(round,b[:2])),tuple(map(round,b[2:])),(255,140,0),2)
            tile = np.full((680,640,3),25,np.uint8)
            tile[40:680] = im
            cv2.putText(tile,f'{label} index {row["index"]}: green visible / blue prediction',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1)
            tiles.append(tile)
        sheets = []
        for start in range(0,len(tiles),4):
            page = tiles[start:start+4]
            page += [np.zeros((680,640,3),np.uint8)]*(4-len(page))
            im = np.concatenate([np.concatenate(page[:2],1),np.concatenate(page[2:],1)],0)
            sheet = out/f'{label}_sheet_{start//4+1:02}.jpg'
            if not cv2.imwrite(str(sheet),im):
                raise ValueError('Prediction sheet encoding failed')
            sheets.append(dict(file=sheet.name,sha256=sha256(sheet)))
        details_path = out/(label+'_details.json')
        write(details_path,details)
        values = [v for r in details for v in r['best_prediction_iou_per_visible_flame']]
        result['models'][label] = dict(weights_sha256=summary['models'][label]['weights_sha256'],
                                      cached_predictions_sha256=sha256(path),original_same_subset=aggregate([r['original'] for r in details]),
                                      supplemental={str(t):aggregate([r['supplemental'][str(t)] for r in details]) for t in (.3,.5)},
                                      mean_best_prediction_iou=float(np.mean(values)),
                                      details_sha256=sha256(details_path),sheets=sheets)
        write(out/'summary_partial.json',result)
    write(out/'summary.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--source-cache',type=Path,required=True)
    p.add_argument('--models',nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    score(a.review,a.source_cache,a.models,a.out)
