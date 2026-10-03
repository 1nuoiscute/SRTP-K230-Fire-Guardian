"""Freeze reviewed queue images/labels into isolated roles, never a combined train split."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import shutil
from data_integrity import sha256
from fetch_kitchen_context_queue import license_allowed


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def validate_rows(rows):
    seen=set();identities=defaultdict(set);groups=defaultdict(set);authors=defaultdict(set)
    for row in rows:
        if row['file'] in seen or Path(row['file']).name!=row['file']:
            raise ValueError('Duplicate or unsafe file name')
        seen.add(row['file'])
        if row['role'] not in {'development','reserved_holdout','held'}: raise ValueError('Unknown role')
        if not row['visually_reviewed'] or not row['source_group'] or not row['note']:
            raise ValueError('Incomplete visual review')
        groups[row['source_group']].add(row['role'])
        authors[row['author'].strip().casefold()].add(row['role'])
        identities[row['original_sha1']].add(row['role'])
        if row['visible_target']=='unknown':
            if row['role']!='held' or row['boxes']: raise ValueError('Unknown visible target must stay held without labels')
        elif row['visible_target']=='visible_no_flame':
            if row['boxes']: raise ValueError('Visible-no-flame row cannot have boxes')
        elif row['visible_target']=='flame':
            if not row['boxes']: raise ValueError('Visible flame requires reviewed boxes')
        else: raise ValueError('Unknown visible-target policy')
        for box in row['boxes']:
            if len(box)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in box): raise ValueError('Invalid box')
            x1,y1,x2,y2=box
            if not (0<=x1<x2<=row['width'] and 0<=y1<y2<=row['height']): raise ValueError('Invalid box bounds')
    for registry in [groups,authors,identities]:
        for key,roles in registry.items():
            if not key or len(roles)>1: raise ValueError('Source, author or original identity crosses frozen roles')


def freeze(queue,review_path,audit_path,out):
    if out.exists(): raise ValueError('Refusing overwrite')
    review=read(review_path);audit=read(audit_path);manifest=read(queue/'manifest.json')
    if not review['no_model_predictions_on_queue']: raise ValueError('This queue is no longer prediction-blind')
    if sha256(queue/'manifest.json')!=review['queue_manifest_sha256'] or sha256(audit_path)!=review['audit_summary_sha256']:
        raise ValueError('Review input identity changed')
    if audit['queue_manifest_sha256']!=review['queue_manifest_sha256']: raise ValueError('Wrong overlap audit')
    rows=review['rows'];validate_rows(rows)
    source={r['file']:r for r in manifest if 'file' in r}
    overlaps={r['file']:r for r in audit['candidates']}
    if set(source)!=set(r['file'] for r in rows) or set(source)!=set(overlaps): raise ValueError('Review must cover entire downloaded queue')
    from PIL import Image
    for row in rows:
        original=source[row['file']];image=queue/row['file']
        if sha256(image)!=row['sha256'] or row['sha256']!=original['sha256']: raise ValueError('Reviewed image changed')
        for key in ['canonical_title','original_sha1','source_page','author','license','license_url','width','height']:
            if row[key]!=original[key]: raise ValueError('Source metadata mismatch')
        metadata=queue/original['metadata_file']
        if sha256(metadata)!=original['metadata_sha256']: raise ValueError('API metadata changed')
        if not license_allowed(row['license'],row['license_url']): raise ValueError('License not admitted')
        if overlaps[row['file']]['exact_byte_matches'] or overlaps[row['file']]['exact_pixel_matches']:
            raise ValueError('Known exposure cannot enter new-source roles')
        with Image.open(image) as im:
            if list(im.size)!=[row['width'],row['height']]: raise ValueError('Dimensions changed')
    # Copy only after all review/source/role checks pass. Held images get no empty labels.
    out.mkdir(parents=True);inventory=[]
    for row in rows:
        if row['role']=='held': continue
        image_dir=out/row['role']/'images';label_dir=out/row['role']/'labels'
        image_dir.mkdir(parents=True,exist_ok=True);label_dir.mkdir(parents=True,exist_ok=True)
        image=image_dir/row['file'];shutil.copy2(queue/row['file'],image)
        label=label_dir/(Path(row['file']).stem+'.txt');lines=[]
        for x1,y1,x2,y2 in row['boxes']:
            w,h=row['width'],row['height']
            lines.append(f'0 {(x1+x2)/2/w:.9f} {(y1+y2)/2/h:.9f} {(x2-x1)/w:.9f} {(y2-y1)/h:.9f}')
        label.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8',newline='\n')
        inventory.append(dict(file=row['file'],role=row['role'],source_group=row['source_group'],image_sha256=sha256(image),label_sha256=sha256(label),boxes=len(lines)))
    result=dict(role='reviewed_role_separated_data_not_yet_used_for_training_or_prediction',
        review_sha256=sha256(review_path),audit_sha256=sha256(audit_path),builder_sha256=sha256(Path(__file__)),
        queue_manifest_sha256=review['queue_manifest_sha256'],inventory=inventory,
        counts=dict(Counter(row['role'] for row in rows)),
        group_counts={role:len({r['source_group'] for r in rows if r['role']==role}) for role in ['development','reserved_holdout','held']},
        limits=['No data.yaml or merged train list is created.','Reserved holdout is excluded from development copies.','Any future training builder must bind this review and manifest and reject reserved group/source identities.','This tiny targeted held-out candidate is not representative project acceptance.'])
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(counts=result['counts'],group_counts=result['group_counts'],labelled_burner_regions=sum(r['boxes'] for r in inventory)),indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['queue','review','audit','out']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();freeze(args.queue,args.review,args.audit,args.out)
