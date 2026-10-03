"""Audited train-only extension of visible-flame labels and reviewed no-fire sampling."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import yaml
from data_integrity import sha256, validate_yolo
from semantic_fire_dataset import verify as verify_parent, yolo_label
from fire_label_revision_dataset import validation_snapshot

PARENT_SHA = 'f3237c9ae34804918968a34945d51886266313aed83b7938e012586e11471069'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def quantiles(rows, count):
    rows = sorted(rows, key=lambda r: (r['group_id'], r['sha256']))
    if len(rows) < count:
        raise ValueError('Insufficient selection stratum')
    return [rows[((2*i+1)*len(rows))//(2*count)] for i in range(count)]


def prepare(parent, out):
    if out.exists():
        raise ValueError('Existing review; do not overwrite')
    checked = verify_parent(parent, PARENT_SHA)
    meta = read(parent/'build_manifest.json')
    base = Path(meta['base'])
    raw = list(csv.DictReader((base/'manifest.csv').open(encoding='utf-8-sig')))
    excluded = {r['image'] for r in meta['excluded_lineage']}
    earlier = {Path(r['image']).name for r in read(Path(meta['review'])/'review_inputs.json')['images']}
    eligible = [r for r in raw if r['split']=='train' and r['file'] in excluded and r['file'] not in earlier]
    selected = [(r, 'flame') for count in (1,2,3,4) for r in quantiles([r for r in eligible if int(r['fire_boxes'])==count],8)]
    for provenance in ('nofire_real_indoor','object_pan'):
        selected += [(r,'negative') for r in quantiles([r for r in raw if r['split']=='train' and r['provenance']==provenance and r['fire_boxes']=='0'],8)]
    forbidden = {r['sha256'] for r in raw if r['split']!='train'}
    if len(selected)!=48 or len({r['sha256'] for r,_ in selected})!=48 or any(r['sha256'] in forbidden for r,_ in selected):
        raise ValueError('Duplicate or non-train pixels selected')
    out.mkdir(parents=True)
    rows=[]
    for index,(r,kind) in enumerate(selected,1):
        image=base/'images/train'/r['file']; label=base/'labels/train'/(image.stem+'.txt')
        if sha256(image)!=r['sha256']:
            raise ValueError('Selected source changed')
        text=label.read_text(encoding='utf-8'); validate_yolo(text)
        pixels=cv2.imread(str(image))
        if pixels is None or (kind=='flame' and pixels.shape[:2]!=(640,640)):
            raise ValueError('Unexpected source geometry')
        boxes=[]
        for line in text.splitlines():
            _,x,y,w,h=map(float,line.split())
            boxes.append([(x-w/2)*640,(y-h/2)*640,(x+w/2)*640,(y+h/2)*640])
        if len(boxes)!=int(r['fire_boxes']):
            raise ValueError('Manifest label count differs')
        rows.append(dict(index=index,kind=kind,image=str(image.resolve()),image_sha256=sha256(image),
            label=str(label.resolve()),label_sha256=sha256(label),split='train',size=[int(pixels.shape[1]),int(pixels.shape[0])],
            source_group=r['group_id'],provenance=r['provenance'],license=r['license'],original_boxes_xyxy=boxes))
    sheets=render_rows(out,rows,'original',False)
    write(out/'review_inputs.json',dict(parent=str(parent.resolve()),parent_identity=checked,source_manifest_sha256=sha256(base/'manifest.csv'),
        role='48 train-only originals selected without predictions; no independent kitchen claim',
        selection='Eight midpoint quantiles per original flame-count stratum 1..4, plus eight per no-fire source; earlier pilot excluded',
        predictions_shown=False,no_exact_pixel_overlap_with_original_val_test=True,independent_kitchen=False,
        smoke_absence_unlabelled=True,script_sha256=sha256(Path(__file__)),images=rows,sheets=sheets))
    print('Prepared 32 flame + 16 negative train originals in 12 sheets',flush=True)


def render_rows(out, rows, prefix, proposal):
    tiles=[]; sheets=[]
    for row in rows:
        im=cv2.imread(row['image'])
        if im is None or [im.shape[1],im.shape[0]]!=row['size']:
            raise ValueError('Unreadable original')
        if proposal:
            for box in row['original_boxes_xyxy']:
                cv2.rectangle(im,tuple(map(round,box[:2])),tuple(map(round,box[2:])),(170,170,170),1)
            for box in row['boxes_xyxy']:
                cv2.rectangle(im,tuple(map(round,box[:2])),tuple(map(round,box[2:])),(0,255,0),2)
        tile=np.full((680,640,3),25,np.uint8)
        ratio=min(640/im.shape[1],640/im.shape[0])
        resized=cv2.resize(im,(round(im.shape[1]*ratio),round(im.shape[0]*ratio)))
        h,w=resized.shape[:2]; tile[40+(640-h)//2:40+(640-h)//2+h,(640-w)//2:(640-w)//2+w]=resized
        name=f"{row['index']:02}: {row['kind']} / {row.get('status','original')} / {len(row['original_boxes_xyxy'])} old"
        cv2.putText(tile,name,(8,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1)
        tiles.append(tile)
    for start in range(0,len(tiles),4):
        page=tiles[start:start+4]
        pixels=np.concatenate([np.concatenate(page[:2],1),np.concatenate(page[2:],1)],0)
        path=out/f'{prefix}_{start//4+1:02}.jpg'
        if path.exists() or not cv2.imwrite(str(path),pixels):
            raise ValueError('Existing or failed sheet')
        sheets.append(dict(file=path.name,sha256=sha256(path)))
    return sheets


def validate_decisions(inputs, decisions):
    sources=inputs['images']
    if len(sources)!=48 or [r['index'] for r in decisions]!=list(range(1,49)):
        raise ValueError('Incomplete ordered review')
    rows=[]
    for source,decision in zip(sources,decisions):
        if source['split']!='train' or sha256(Path(source['image']))!=source['image_sha256'] or sha256(Path(source['label']))!=source['label_sha256']:
            raise ValueError('Source identity or split changed')
        status=decision['status']; boxes=decision['boxes_xyxy']
        if not decision['note'].strip() or status not in ('held','flame','negative'):
            raise ValueError('Missing explicit decision')
        if (status=='flame' and (source['kind']!='flame' or not boxes)) or (status!='flame' and boxes) or (status=='negative' and (source['kind']!='negative' or source['original_boxes_xyxy'])):
            raise ValueError('Held/negative treated as positive or uncertain flame treated as negative')
        for b in boxes:
            if len(b)!=4 or not all(type(v) in (int,float) and np.isfinite(v) for v in b) or not (0<=b[0]<b[2]<=640 and 0<=b[1]<b[3]<=640):
                raise ValueError('Invalid flame envelope')
        rows.append({**source,**decision})
    return rows


def render(review):
    inputs=read(review/'review_inputs.json'); proposal=read(review/'manual_proposal.json')
    rows=validate_decisions(inputs,proposal['images'])
    sheets=render_rows(review,rows,'proposal',True)
    write(review/'proposal_render_manifest.json',dict(input_sha256=sha256(review/'review_inputs.json'),proposal_sha256=sha256(review/'manual_proposal.json'),sheets=sheets,script_sha256=sha256(Path(__file__))))


def finalize(review,seen):
    if (review/'review_complete.json').exists():
        raise ValueError('Existing approval')
    inputs=read(review/'review_inputs.json'); proposal=read(review/'manual_proposal.json'); rendered=read(review/'proposal_render_manifest.json')
    rows=validate_decisions(inputs,proposal['images'])
    expected=inputs['sheets']+rendered['sheets']
    if len(seen)!=len(set(seen)) or set(seen)!={s['file'] for s in expected}:
        raise ValueError('Every original and proposal sheet must be inspected')
    for s in expected:
        if sha256(review/s['file'])!=s['sha256']:
            raise ValueError('Seen sheet changed')
    if rendered['input_sha256']!=sha256(review/'review_inputs.json') or rendered['proposal_sha256']!=sha256(review/'manual_proposal.json') or inputs['script_sha256']!=sha256(Path(__file__)) or rendered['script_sha256']!=sha256(Path(__file__)):
        raise ValueError('Review lineage changed')
    write(review/'review_complete.json',dict(input_sha256=sha256(review/'review_inputs.json'),proposal_sha256=sha256(review/'manual_proposal.json'),render_sha256=sha256(review/'proposal_render_manifest.json'),
        script_sha256=sha256(Path(__file__)),scope='Approximate train-only visual review; smoke absence unknown; no independent kitchen',
        original_labels_modified=False,independent_kitchen=False,smoke_absence_unlabelled=True,
        approved_flame_images=sum(r['status']=='flame' for r in rows),approved_flame_boxes=sum(len(r['boxes_xyxy']) for r in rows),
        approved_negative_images=sum(r['status']=='negative' for r in rows),held_images=sum(r['status']=='held' for r in rows),seen_sheets=expected,images=rows))


def approved_review(review):
    complete=read(review/'review_complete.json'); inputs=read(review/'review_inputs.json')
    for key,name in [('input_sha256','review_inputs.json'),('proposal_sha256','manual_proposal.json'),('render_sha256','proposal_render_manifest.json')]:
        if sha256(review/name)!=complete[key]:
            raise ValueError('Completed review changed')
    if complete['script_sha256']!=sha256(Path(__file__)) or complete['independent_kitchen'] or not complete['smoke_absence_unlabelled']:
        raise ValueError('Approval scope changed')
    rows=validate_decisions(inputs,read(review/'manual_proposal.json')['images'])
    if rows!=complete['images']:
        raise ValueError('Approved decisions changed')
    for s in complete['seen_sheets']:
        if sha256(review/s['file'])!=s['sha256']:
            raise ValueError('Approved sheet changed')
    return inputs,complete,rows


def planned_rows(parent,review):
    verify_parent(parent,PARENT_SHA)
    source=read(parent/'build_manifest.json'); inputs,complete,decisions=approved_review(review)
    if Path(inputs['parent']).resolve()!=parent.resolve() or inputs['parent_identity']['manifest_sha256']!=PARENT_SHA or inputs['source_manifest_sha256']!=sha256(Path(source['base'])/'manifest.csv'):
        raise ValueError('Wrong reviewed parent')
    kept=deepcopy(source['lineage']); excluded={r['image']:deepcopy(r) for r in source['excluded_lineage']}; present={r['image']:r for r in kept}
    for r in decisions:
        name=Path(r['image']).name
        if r['status']=='held':
            continue
        if r['status']=='flame':
            if name not in excluded or name in present or excluded[name]['sha256']!=r['image_sha256'] or excluded[name]['label_sha256']!=r['label_sha256']:
                raise ValueError('Flame approval not in exact quarantined train inventory')
            item=excluded.pop(name)
            item.update(weight=8,teacher_preserve=False,visible_flame_train_revision=True,retained_parent_revision=False,
                parent_label_sha256=r['label_sha256'],boxes_xyxy=r['boxes_xyxy'],v17_new_flame_revision=True)
            kept.append(item); present[name]=item
        else:
            if name not in present or present[name]['origin']!='legacy_train' or present[name]['sha256']!=r['image_sha256'] or present[name]['label_sha256']!=r['label_sha256']:
                raise ValueError('Negative outside exact parent train inventory')
            label=parent/'labels/train'/(Path(name).stem+'.txt')
            if label.read_text(encoding='utf-8').strip():
                raise ValueError('Negative parent has labels')
            present[name].update(weight=8,v17_reviewed_negative=True)
    return source,complete,kept,list(excluded.values())


def build(parent,review,out):
    if out.exists():
        raise ValueError('Existing extension dataset')
    source,complete,rows,excluded=planned_rows(parent,review)
    (out/'images/train').mkdir(parents=True); (out/'labels/train').mkdir(parents=True)
    decisions={Path(r['image']).name:r for r in complete['images']}; entries=[]
    for r in rows:
        name=r['image']; new=r.get('v17_new_flame_revision',False)
        image=Path(decisions[name]['image']) if new else parent/'images/train'/name
        target=out/'images/train'/name; label=out/'labels/train'/(Path(name).stem+'.txt')
        shutil.copyfile(image,target)
        if new:
            label.write_text(yolo_label(r['boxes_xyxy']),encoding='utf-8'); r['label_sha256']=sha256(label)
        else:
            shutil.copyfile(parent/'labels/train'/label.name,label)
        entries.extend([str(target.resolve())]*r['weight'])
    (out/'train.txt').write_text('\n'.join(entries)+'\n',encoding='utf-8')
    base=Path(source['base'])
    config=dict(path=str(out.resolve()),train=str((out/'train.txt').resolve()),val=str(base/'images/val'),nc=1,names={0:'fire'})
    (out/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    write(out/'build_manifest.json',dict(role='v17_train_only_visible_flame_and_negative_extension',parent=str(parent.resolve()),parent_sha256=PARENT_SHA,
        review=str(review.resolve()),review_sha256=sha256(review/'review_complete.json'),builder_sha256=sha256(Path(__file__)),
        base=str(base),validation=validation_snapshot(base),original_validation_retained=True,independent_kitchen=False,smoke_absence_unlabelled=True,
        unique_paths=len(rows),train_entries=len(entries),new_flame_images=complete['approved_flame_images'],new_flame_boxes=complete['approved_flame_boxes'],
        reviewed_negative_images=complete['approved_negative_images'],held_images=complete['held_images'],lineage=rows,excluded_lineage=excluded))
    print(json.dumps(verify(out,sha256(out/'build_manifest.json')),indent=2),flush=True)


def verify(folder,expected):
    if sha256(folder/'build_manifest.json')!=expected:
        raise ValueError('Extension manifest changed')
    meta=read(folder/'build_manifest.json')
    if meta['builder_sha256']!=sha256(Path(__file__)) or meta['parent_sha256']!=PARENT_SHA or meta['independent_kitchen'] or not meta['smoke_absence_unlabelled']:
        raise ValueError('Extension identity/scope changed')
    parent=Path(meta['parent']); review=Path(meta['review'])
    if sha256(review/'review_complete.json')!=meta['review_sha256']:
        raise ValueError('Approval changed')
    source,complete,rows,excluded=planned_rows(parent,review)
    if excluded!=meta['excluded_lineage'] or meta['validation']!=validation_snapshot(Path(source['base'])):
        raise ValueError('Quarantine or validation changed')
    counter=Counter()
    for row,actual in zip(rows,meta['lineage']):
        image=folder/'images/train'/row['image']; label=folder/'labels/train'/(image.stem+'.txt')
        if row.get('v17_new_flame_revision'):
            if label.read_text(encoding='utf-8')!=yolo_label(row['boxes_xyxy']):
                raise ValueError('Unapproved new geometry')
            row['label_sha256']=sha256(label)
        if row!=actual or sha256(image)!=row['sha256'] or sha256(label)!=row['label_sha256']:
            raise ValueError('Pixels, labels or inherited weights changed')
        validate_yolo(label.read_text(encoding='utf-8')); counter[image.resolve()]=row['weight']
    actual_counter=Counter(Path(p).resolve() for p in (folder/'train.txt').read_text(encoding='utf-8').splitlines())
    base=Path(meta['base'])
    config=dict(path=str(folder.resolve()),train=str((folder/'train.txt').resolve()),val=str(base/'images/val'),nc=1,names={0:'fire'})
    if actual_counter!=counter or len(meta['lineage'])!=len(rows) or len(rows)!=meta['unique_paths'] or sum(counter.values())!=meta['train_entries'] or yaml.safe_load((folder/'data.yaml').read_text(encoding='utf-8'))!=config:
        raise ValueError('Membership, sampling or configuration changed')
    if {p.name for p in (folder/'images/train').iterdir()}!={r['image'] for r in rows} or {p.name for p in (folder/'labels/train').iterdir()}!={Path(r['image']).stem+'.txt' for r in rows}:
        raise ValueError('Unlisted image/label')
    for field,review_field in [('new_flame_images','approved_flame_images'),('new_flame_boxes','approved_flame_boxes'),('reviewed_negative_images','approved_negative_images'),('held_images','held_images')]:
        if meta[field]!=complete[review_field]:
            raise ValueError('Review count differs')
    return dict(manifest_sha256=expected,images_verified=len(rows),entries_verified=sum(counter.values()),
        new_flame_images=meta['new_flame_images'],new_flame_boxes=meta['new_flame_boxes'],reviewed_negative_images=meta['reviewed_negative_images'],
        excluded_industrial_images=len(excluded),validation_images=len(meta['validation']),independent_kitchen=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','render','finalize','build','verify'))
    p.add_argument('--parent',type=Path); p.add_argument('--review',type=Path); p.add_argument('--out',type=Path)
    p.add_argument('--seen-sheet',action='append',default=[]); p.add_argument('--expected-sha256')
    a=p.parse_args()
    if a.action=='prepare': prepare(a.parent.resolve(),a.out.resolve())
    elif a.action=='render': render(a.review.resolve())
    elif a.action=='finalize': finalize(a.review.resolve(),a.seen_sheet)
    elif a.action=='build': build(a.parent.resolve(),a.review.resolve(),a.out.resolve())
    else: print(json.dumps(verify(a.out.resolve(),a.expected_sha256),indent=2))

