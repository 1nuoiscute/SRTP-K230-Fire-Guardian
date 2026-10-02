"""Build a separate train-only semantics pilot, quarantining unreviewed industrial labels."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import yaml
from data_integrity import sha256,validate_yolo
from fire_label_revision_dataset import read,verify as verify_parent,validation_snapshot
from train_flame_semantic_review import validate as validate_review


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def yolo_label(boxes):
    if not boxes: raise ValueError('Visible-flame positive must have boxes')
    lines=[]
    for x1,y1,x2,y2 in boxes:
        lines.append(f'0 {(x1+x2)/1280:.8f} {(y1+y2)/1280:.8f} {(x2-x1)/640:.8f} {(y2-y1)/640:.8f}')
    text='\n'.join(lines)+'\n'; validate_yolo(text)
    return text


def review_approved(review):
    meta=read(review/'review_complete.json')
    inputs,proposal,rows=validate_review(review)
    for key,name in [('input_sha256','review_inputs.json'),('proposal_sha256','manual_proposal.json'),('render_manifest_sha256','proposal_render_manifest.json')]:
        if meta[key]!=sha256(review/name): raise ValueError('Completed train review changed')
    if meta['review_script_sha256']!=sha256(Path(__file__).with_name('train_flame_semantic_review.py')):
        raise ValueError('Train review code changed')
    for row in rows:
        if row['status']=='proposed': row['status']='approved_approximate_train_pilot'
    if rows!=meta['images'] or meta['independent_kitchen'] or not meta['smoke_absence_unlabelled'] or not meta['annotations_are_approximate']:
        raise ValueError('Approved semantics scope differs')
    for sheet in meta['visually_seen_sheets']:
        if sha256(review/sheet['file'])!=sheet['sha256']: raise ValueError('Approved proposal sheet changed')
    approved={Path(r['image']).name:r for r in rows if r['status']=='approved_approximate_train_pilot'}
    if len(approved)!=meta['approved_images'] or sum(len(r['boxes_xyxy']) for r in approved.values())!=meta['approved_boxes']:
        raise ValueError('Approval counts differ')
    return meta,approved


def membership(parent_rows,industrial,approved,pilot_weight):
    if type(pilot_weight) is not int or not 1<=pilot_weight<=16:
        raise ValueError('Pilot weight outside declared bounded range')
    source={r['image']:r for r in parent_rows}
    if len(source)!=len(parent_rows) or not approved or not set(approved)<=industrial.keys() or not industrial.keys()<=source.keys():
        raise ValueError('Duplicate parent or out-of-train approval')
    kept,excluded=[],[]
    for name,row in source.items():
        if name in industrial:
            raw=industrial[name]
            if row['origin']!='legacy_train' or row['sha256']!=raw['sha256']:
                raise ValueError('Industrial source differs from exact legacy train parent')
            revision=approved.get(name)
            if revision is None:
                if row.get('label_revised') is True:
                    if row['teacher_preserve']:
                        raise ValueError('Reviewed parent revision must be exempt from teacher geometry')
                    kept.append({**row,'visible_flame_train_revision':False,'retained_parent_revision':True})
                else:
                    excluded.append(row)
                continue
            if revision['split']!='train' or revision['image_sha256']!=row['sha256'] or revision['label_sha256']!=row['label_sha256']:
                raise ValueError('Approved image/label differs from parent')
            kept.append({**row,'weight':pilot_weight,'teacher_preserve':False,
                         'visible_flame_train_revision':True,'parent_label_sha256':row['label_sha256'],
                         'boxes_xyxy':revision['boxes_xyxy'],'retained_parent_revision':False})
        else:
            kept.append({**row,'visible_flame_train_revision':False,'retained_parent_revision':row.get('label_revised') is True})
    return kept,excluded


def parent_and_plan(parent,expected_parent,review,pilot_weight):
    checked=verify_parent(parent,expected_parent)
    source=read(parent/'build_manifest.json')
    reviewed,approved=review_approved(review)
    base=Path(source['base'])
    if sha256(base/'manifest.csv')!=reviewed['source_manifest_sha256']:
        raise ValueError('Review original manifest differs')
    raw=list(csv.DictReader((base/'manifest.csv').open(encoding='utf-8-sig')))
    industrial={r['file']:r for r in raw if r['split']=='train' and r['provenance']=='kitchen_stove_fire'}
    if len(industrial)!=1045: raise ValueError('Original industrial training inventory differs')
    forbidden={r['sha256'] for r in raw if r['split']!='train'}
    if any(r['image_sha256'] in forbidden for r in approved.values()):
        raise ValueError('Approved pilot contains original val/test pixels')
    rows,excluded=membership(source['lineage'],industrial,approved,pilot_weight)
    return checked,source,reviewed,approved,rows,excluded


def build(parent,expected_parent,review,pilot_weight,out):
    parent,review,out=parent.resolve(),review.resolve(),out.resolve()
    if out.exists(): raise ValueError('Existing semantic dataset; no overwrite')
    checked,source,reviewed,approved,rows,excluded=parent_and_plan(parent,expected_parent,review,pilot_weight)
    (out/'images/train').mkdir(parents=True); (out/'labels/train').mkdir(parents=True)
    entries=[]
    for row in rows:
        image=parent/'images/train'/row['image']; old=parent/'labels/train'/(image.stem+'.txt')
        target=out/'images/train'/image.name; label=out/'labels/train'/old.name
        shutil.copyfile(image,target)
        if row['visible_flame_train_revision']:
            label.write_text(yolo_label(row['boxes_xyxy']),encoding='utf-8')
            row['label_sha256']=sha256(label)
        else:
            shutil.copyfile(old,label)
        entries.extend([str(target)]*row['weight'])
    (out/'train.txt').write_text('\n'.join(entries)+'\n',encoding='utf-8')
    base=Path(source['base'])
    config=dict(path=str(out),train=str(out/'train.txt'),val=str(base/'images/val'),nc=1,names={0:'fire'})
    (out/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    meta=dict(role='development_visible_flame_source_quarantine_pilot',parent=str(parent),base=str(base),
              parent_identity=checked,review=str(review),review_complete_sha256=sha256(review/'review_complete.json'),
              pilot_weight=pilot_weight,unique_paths=len(rows),train_entries=len(entries),
              corrected_images=len(approved),corrected_boxes=reviewed['approved_boxes'],
              retained_prior_revised_images=sum(r['retained_parent_revision'] for r in rows),
              retained_prior_revised_boxes=sum(len((parent/'labels/train'/(Path(r['image']).stem+'.txt')).read_text(encoding='utf-8').splitlines()) for r in rows if r['retained_parent_revision']),
              excluded_industrial_images=len(excluded),source_industrial_images=1045,
              source_policy='Keep exact previously approved parent revisions and new reviewed approximate train pilot. Quarantine other industrial source labels; excluded images are not negatives.',
              smoke_absence_unlabelled=True,independent_kitchen=False,original_validation_retained=True,
              validation=validation_snapshot(base),lineage=rows,excluded_lineage=excluded,
              builder_sha256=sha256(Path(__file__)))
    write(out/'build_manifest.json',meta)
    return verify(out,sha256(out/'build_manifest.json'))


def verify(folder,expected):
    folder=folder.resolve(); manifest=folder/'build_manifest.json'
    if sha256(manifest)!=expected: raise ValueError('Semantic dataset manifest changed')
    meta=read(manifest)
    if meta['role']!='development_visible_flame_source_quarantine_pilot' or not meta['smoke_absence_unlabelled'] or meta['independent_kitchen']:
        raise ValueError('Dataset semantics scope differs')
    if meta['builder_sha256']!=sha256(Path(__file__)): raise ValueError('Dataset builder source changed')
    parent,review=Path(meta['parent']),Path(meta['review'])
    if sha256(review/'review_complete.json')!=meta['review_complete_sha256']: raise ValueError('Completed review changed')
    checked,source,reviewed,approved,expected_rows,excluded=parent_and_plan(parent,meta['parent_identity']['manifest_sha256'],review,meta['pilot_weight'])
    if checked!=meta['parent_identity'] or excluded!=meta['excluded_lineage']:
        raise ValueError('Parent or quarantine inventory changed')
    actual_rows=meta['lineage']
    if len(actual_rows)!=len(expected_rows): raise ValueError('Membership count changed')
    expected_entries=Counter()
    for actual,row in zip(actual_rows,expected_rows):
        image=folder/'images/train'/row['image']; label=folder/'labels/train'/(image.stem+'.txt')
        if row['visible_flame_train_revision']:
            expected_text=yolo_label(row['boxes_xyxy'])
            if label.read_text(encoding='utf-8')!=expected_text: raise ValueError('Unapproved visible-flame label')
            row['label_sha256']=sha256(label)
        if actual!=row or sha256(image)!=row['sha256'] or sha256(label)!=row['label_sha256']:
            raise ValueError('Derived lineage, image or label differs')
        validate_yolo(label.read_text(encoding='utf-8'))
        expected_entries[image]=row['weight']
    entries=Counter(Path(p).resolve() for p in (folder/'train.txt').read_text(encoding='utf-8').splitlines() if p)
    image_names={p.name for p in (folder/'images/train').iterdir() if p.is_file()}
    label_names={p.name for p in (folder/'labels/train').iterdir() if p.is_file()}
    if entries!=expected_entries or image_names!={r['image'] for r in actual_rows} or label_names!={Path(r['image']).stem+'.txt' for r in actual_rows}:
        raise ValueError('Extra, missing or reweighted training files')
    if meta['unique_paths']!=len(actual_rows) or meta['train_entries']!=sum(entries.values()) or meta['excluded_industrial_images']!=len(excluded) or meta['corrected_images']!=len(approved) or meta['corrected_boxes']!=reviewed['approved_boxes']:
        raise ValueError('Derived count claim differs')
    retained_images=sum(r['retained_parent_revision'] for r in actual_rows)
    retained_boxes=sum(len((folder/'labels/train'/(Path(r['image']).stem+'.txt')).read_text(encoding='utf-8').splitlines()) for r in actual_rows if r['retained_parent_revision'])
    if meta['retained_prior_revised_images']!=retained_images or meta['retained_prior_revised_boxes']!=retained_boxes:
        raise ValueError('Previously approved revision retention differs')
    base=Path(meta['base']); config=yaml.safe_load((folder/'data.yaml').read_text(encoding='utf-8'))
    if config!=dict(path=str(folder),train=str(folder/'train.txt'),val=str(base/'images/val'),nc=1,names={0:'fire'}) or validation_snapshot(base)!=meta['validation']:
        raise ValueError('Validation/config changed')
    return dict(manifest_sha256=sha256(manifest),config_sha256=sha256(folder/'data.yaml'),train_list_sha256=sha256(folder/'train.txt'),
                images_verified=len(actual_rows),entries_verified=sum(entries.values()),corrected_images=len(approved),
                corrected_boxes=reviewed['approved_boxes'],excluded_industrial_images=len(excluded),
                retained_prior_revised_images=retained_images,retained_prior_revised_boxes=retained_boxes,
                teacher_preserved_images=sum(r['teacher_preserve'] for r in actual_rows),
                scope='Train-only approximate visible flame pilot with quarantined unreviewed industrial source; not independent evaluation')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--expected-parent-sha256',required=True); p.add_argument('--review',type=Path,required=True)
    p.add_argument('--pilot-weight',type=int,default=8); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(build(a.parent,a.expected_parent_sha256,a.review,a.pilot_weight,a.out),indent=2))
