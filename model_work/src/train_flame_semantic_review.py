"""Render and freeze a separate train-only flame-label pilot after visual review."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from data_integrity import sha256


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def validate(review):
    inputs=json.loads((review/'review_inputs.json').read_text(encoding='utf-8'))
    proposal=json.loads((review/'manual_proposal.json').read_text(encoding='utf-8'))
    if inputs['selected_images']!=32 or inputs['predictions_shown'] or not inputs['no_exact_pixel_overlap_with_original_val_test'] or not inputs['smoke_absence_unlabelled']:
        raise ValueError('Train-only pilot scope differs')
    if inputs['script_sha256']!=sha256(Path(__file__).with_name('prepare_train_flame_semantic_review.py')):
        raise ValueError('Selection source changed')
    sources,decisions=inputs['images'],proposal['images']
    if len(sources)!=32 or len(decisions)!=32 or [r['index'] for r in sources]!=list(range(1,33)) or [r['index'] for r in decisions]!=list(range(1,33)):
        raise ValueError('Incomplete or reordered review')
    if len({r['image_sha256'] for r in sources})!=32:
        raise ValueError('Duplicate pilot pixels')
    for sheet in inputs['sheets']:
        if sha256(review/sheet['file'])!=sheet['sha256']: raise ValueError('Original review sheet changed')
    rows=[]
    for source,decision in zip(sources,decisions):
        if source['split']!='train' or sha256(Path(source['image']))!=source['image_sha256'] or sha256(Path(source['label']))!=source['label_sha256']:
            raise ValueError('Train source identity changed')
        if source['size']!=[640,640] or decision['status'] not in ('held','proposed') or not decision['note'].strip():
            raise ValueError('Invalid decision or geometry')
        boxes=decision['boxes_xyxy']
        if decision['status']=='held' and boxes: raise ValueError('Held image cannot have supervised boxes')
        if decision['status']=='proposed' and (not boxes or len(boxes)!=len(source['original_boxes_xyxy'])):
            raise ValueError('Incomplete visible burner coverage')
        for box in boxes:
            if len(box)!=4 or not all(isinstance(v,(int,float)) and np.isfinite(v) for v in box): raise ValueError('Invalid visible flame box')
            x1,y1,x2,y2=box
            if not 0<=x1<x2<=640 or not 0<=y1<y2<=640: raise ValueError('Box outside source image')
        rows.append({**source,**decision})
    return inputs,proposal,rows


def render(review):
    target=review/'proposal_sheets'
    if target.exists(): raise ValueError('Existing proposal sheets')
    inputs,proposal,rows=validate(review); target.mkdir()
    tiles,sheets=[],[]
    for row in rows:
        im=cv2.imread(row['image'])
        if im is None or im.shape[:2]!=(640,640): raise ValueError('Image decoder geometry differs')
        for b in row['original_boxes_xyxy']: cv2.rectangle(im,tuple(map(round,b[:2])),tuple(map(round,b[2:])),(170,170,170),1)
        for b in row['boxes_xyxy']: cv2.rectangle(im,tuple(map(round,b[:2])),tuple(map(round,b[2:])),(0,255,0),2)
        tile=np.full((680,640,3),25,np.uint8); tile[40:680]=im
        cv2.putText(tile,f'Index {row["index"]}: {row["status"]} / green flame / gray old',(8,26),cv2.FONT_HERSHEY_SIMPLEX,.53,(240,240,240),1)
        tiles.append(tile)
    for start in range(0,32,4):
        page=tiles[start:start+4]
        pixels=np.concatenate([np.concatenate(page[:2],1),np.concatenate(page[2:],1)],0)
        path=target/f'sheet_{start//4+1:02}.jpg'
        if not cv2.imwrite(str(path),pixels): raise ValueError('Proposal encoding failed')
        sheets.append(dict(file=path.relative_to(review).as_posix(),sha256=sha256(path)))
    write(review/'proposal_render_manifest.json',dict(inputs_sha256=sha256(review/'review_inputs.json'),
          proposal_sha256=sha256(review/'manual_proposal.json'),script_sha256=sha256(Path(__file__)),sheets=sheets))


def finalize(review,seen):
    target=review/'review_complete.json'
    if target.exists(): raise ValueError('Existing completed train review')
    inputs,proposal,rows=validate(review)
    rendered=json.loads((review/'proposal_render_manifest.json').read_text(encoding='utf-8'))
    if rendered['inputs_sha256']!=sha256(review/'review_inputs.json') or rendered['proposal_sha256']!=sha256(review/'manual_proposal.json') or rendered['script_sha256']!=sha256(Path(__file__)):
        raise ValueError('Rendered proposal identity differs')
    seen=[Path(p).as_posix() for p in seen]
    if len(seen)!=len(set(seen)) or sorted(seen)!=sorted(s['file'] for s in rendered['sheets']):
        raise ValueError('All eight proposal sheets must be inspected')
    for sheet in rendered['sheets']:
        if sha256(review/sheet['file'])!=sheet['sha256']: raise ValueError('Seen sheet changed')
    for row in rows:
        if row['status']=='proposed': row['status']='approved_approximate_train_pilot'
    approved=[r for r in rows if r['status']=='approved_approximate_train_pilot']
    meta=dict(role=proposal['role'],policy=proposal['policy'],reviewer='Codex visual inspection of all original and proposal sheets',
              source_manifest_sha256=inputs['source_manifest_sha256'],input_sha256=sha256(review/'review_inputs.json'),
              proposal_sha256=sha256(review/'manual_proposal.json'),render_manifest_sha256=sha256(review/'proposal_render_manifest.json'),
              selection_script_sha256=inputs['script_sha256'],review_script_sha256=sha256(Path(__file__)),
              reviewed_images=32,approved_images=len(approved),approved_boxes=sum(len(r['boxes_xyxy']) for r in approved),
              held_images=32-len(approved),smoke_absence_unlabelled=True,training_experiment_started=False,
              original_labels_modified=False,independent_kitchen=False,annotations_are_approximate=True,
              visually_seen_sheets=rendered['sheets'],images=rows)
    write(target,meta)
    print(json.dumps({k:v for k,v in meta.items() if k not in ('images','visually_seen_sheets')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('render','finalize'))
    p.add_argument('--review',type=Path,required=True); p.add_argument('--seen-sheet',action='append',default=[])
    a=p.parse_args()
    if a.action=='render': render(a.review)
    else: finalize(a.review,a.seen_sheet)
