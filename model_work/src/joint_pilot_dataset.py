"""Identity and partial-label guards for reviewed fire/smoke pilot datasets."""
import json
import math
from pathlib import Path
from data_integrity import sha256


def converted_source_labels(text):
    """Explicit raw D-Fire 0=smoke,1=fire -> project 0=fire,1=smoke."""
    lines = []
    for line in text.splitlines():
        if not line.strip(): continue
        values = list(map(float,line.split()))
        if len(values)!=5 or not all(math.isfinite(v) for v in values): raise ValueError('Malformed source label')
        cls,x,y,w,h = values
        if cls not in (0,1) or not 0<=x<=1 or not 0<=y<=1 or not 0<w<=1 or not 0<h<=1:
            raise ValueError('Invalid source class/coordinate')
        if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6:
            raise ValueError('Source box outside image')
        lines.append(f'{1-int(cls)} {x:.8f} {y:.8f} {w:.8f} {h:.8f}')
    return '\n'.join(lines)+('\n' if lines else '')


def verify_dataset(folder,config,expected_manifest_sha256=None):
    folder = Path(folder).resolve(); manifest_path = folder/'build_manifest.json'
    if expected_manifest_sha256 and sha256(manifest_path)!=expected_manifest_sha256:
        raise ValueError('Dataset manifest changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if config.get('names')!={0:'fire',1:'smoke'} or config.get('nc')!=2:
        raise ValueError('Project class mapping changed')
    if Path(config['path']).resolve()!=folder or config.get('train')!='images/train' or config.get('val')!='images/val' or 'test' in config:
        raise ValueError('Dataset config redirects splits or claims a test')
    review_path = Path(manifest['source_review'])
    if sha256(review_path)!=manifest['source_review_sha256']: raise ValueError('Source review changed')
    review = json.loads(review_path.read_text(encoding='utf-8'))
    if sha256(review_path.with_name('review_pending.json'))!=review['review_pending_sha256']:
        raise ValueError('Pending review changed')
    if sha256(review_path.with_name('decisions.json'))!=review['decisions_sha256']:
        raise ValueError('Decisions changed')
    for sheet in review['sheets']:
        if sha256(review_path.parent/sheet['file'])!=sheet['sha256']: raise ValueError('Review sheet changed')
    approved = {r['index']:r for r in review['images'] if r.get('approved_for_training') is True}
    rows = manifest['images']
    if len(rows)!=len(approved) or {r['review_index'] for r in rows}!=set(approved):
        raise ValueError('Dataset must contain every approved image exactly once, no held images')
    groups = {}; image_set = set(); label_set = set()
    for row in rows:
        source = approved[row['review_index']]
        if source['manual_review']['decision']!='approved_source_labels' or source['errors']:
            raise ValueError('Unapproved/invalid source')
        group = source['reviewed_development_group']
        if row['group']!=group or row['split'] not in ('train','val'):
            raise ValueError('Group/split identity changed')
        if group in groups and groups[group]!=row['split']: raise ValueError('Reviewed group crosses splits')
        groups[group] = row['split']
        image = folder/'images'/row['split']/row['file']
        label = folder/'labels'/row['split']/(Path(row['file']).stem+'.txt')
        if Path(row['file']).name!=row['file'] or image in image_set: raise ValueError('Unsafe/duplicate filename')
        image_set.add(image); label_set.add(label)
        if sha256(Path(source['image']))!=source['sha256'] or sha256(image)!=source['sha256'] or row['image_sha256']!=source['sha256']:
            raise ValueError('Image changed')
        if sha256(Path(source['label']))!=source['label_sha256']: raise ValueError('Original label changed')
        if sha256(label)!=row['label_sha256']: raise ValueError('Converted label changed')
        text = converted_source_labels(Path(source['label']).read_text(encoding='utf-8-sig'))
        if label.read_text(encoding='utf-8')!=text: raise ValueError('Class conversion changed')
    actual_images = {p for split in ('train','val') for p in (folder/'images'/split).iterdir() if p.is_file()}
    actual_labels = {p for split in ('train','val') for p in (folder/'labels'/split).iterdir() if p.is_file()}
    if actual_images!=image_set or actual_labels!=label_set: raise ValueError('Unexpected/missing files')
    if not all(any(r['split']==split for r in rows) for split in ('train','val')): raise ValueError('Empty split')
    return {'verified_images':len(rows),'reviewed_development_groups':len(groups),'source_review_sha256':sha256(review_path),
            'manifest_sha256':sha256(manifest_path),'role':'development-exposed validation, not independent kitchen test'}
