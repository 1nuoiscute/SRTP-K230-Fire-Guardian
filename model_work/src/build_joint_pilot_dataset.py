"""Build a new reviewed two-class dataset; preserve raw publisher and old data."""
import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import yaml
from data_integrity import sha256
from joint_pilot_dataset import converted_source_labels,verify_dataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--expected-review-sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); out=a.out.resolve(); review_path=a.review.resolve()
    if out.exists(): raise SystemExit('Refusing overwrite')
    if sha256(review_path)!=a.expected_review_sha256: raise SystemExit('Review identity changed')
    review=json.loads(review_path.read_text(encoding='utf-8'))
    if sha256(review_path.with_name('review_pending.json'))!=review['review_pending_sha256'] or sha256(review_path.with_name('decisions.json'))!=review['decisions_sha256']:
        raise SystemExit('Review chain changed')
    for sheet in review['sheets']:
        if sha256(review_path.parent/sheet['file'])!=sheet['sha256']: raise SystemExit('Reviewed sheets changed')
    selected=[r for r in review['images'] if r.get('approved_for_training') is True]
    groups=defaultdict(list)
    for row in selected:
        if row['manual_review']['decision']!='approved_source_labels' or row['errors']: raise SystemExit('Invalid approval')
        groups[row['reviewed_development_group']].append(row)
        if sha256(Path(row['image']))!=row['sha256'] or sha256(Path(row['label']))!=row['label_sha256']: raise SystemExit('Source changed')
    strata=defaultdict(list)
    for group,rows in groups.items():
        classes=tuple(sorted({int(b[0]) for r in rows for b in r['boxes']}))
        strata[classes].append(group)
    validation=set()
    for classes,keys in strata.items():
        if len(keys)<2: raise SystemExit('Too few approved groups for development stratification')
        ordered=sorted(keys,key=lambda k:hashlib.sha256(('20261002:val:'+k).encode()).hexdigest())
        validation.update(ordered[:min(len(keys)-1,max(1,round(len(keys)*.2)))])
    for split in ('train','val'):
        (out/'images'/split).mkdir(parents=True); (out/'labels'/split).mkdir(parents=True)
    manifest={'role':'fully reviewed visible two-class development pilot, no independent test',
              'script_sha256':sha256(Path(__file__)),'source_review':str(review_path),'source_review_sha256':sha256(review_path),
              'raw_names':{0:'smoke',1:'fire'},'names':{0:'fire',1:'smoke'},
              'split_rule':'Seed 20261002 hash-sorted reviewed groups, ~20% groups per raw class-set stratum; all validation visually exposed',
              'images':[]}
    for row in selected:
        split='val' if row['reviewed_development_group'] in validation else 'train'
        name='dfire_'+row['sha256'][:24]+Path(row['image']).suffix.lower()
        target=out/'images'/split/name; label=out/'labels'/split/(Path(name).stem+'.txt')
        if target.exists(): raise SystemExit('Filename collision')
        shutil.copyfile(row['image'],target)
        text=converted_source_labels(Path(row['label']).read_text(encoding='utf-8-sig'))
        label.write_text(text,encoding='utf-8')
        counts=Counter(int(line.split()[0]) for line in text.splitlines())
        manifest['images'].append({'review_index':row['index'],'file':name,'split':split,'group':row['reviewed_development_group'],
            'image_sha256':sha256(target),'label_sha256':sha256(label),'fire_boxes':counts[0],'smoke_boxes':counts[1]})
    config={'path':str(out),'train':'images/train','val':'images/val','nc':2,'names':{0:'fire',1:'smoke'}}
    (out/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    (out/'build_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    result=verify_dataset(out,config)
    result['split_counts']={s:{'images':sum(r['split']==s for r in manifest['images']),
                             'fire_boxes':sum(r['fire_boxes'] for r in manifest['images'] if r['split']==s),
                             'smoke_boxes':sum(r['smoke_boxes'] for r in manifest['images'] if r['split']==s)} for s in ('train','val')}
    (out/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
