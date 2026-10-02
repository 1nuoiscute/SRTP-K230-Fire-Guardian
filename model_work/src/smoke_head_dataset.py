"""Derive smoke-only head targets from approved two-class truth, preserving splits."""
import argparse
import json
import shutil
from pathlib import Path
import yaml
from data_integrity import sha256
from joint_pilot_dataset import converted_source_labels, verify_dataset


def smoke_only_labels(joint_text):
    # The strict parser swaps class ids. Joint 1=smoke becomes head-only 0;
    # joint 0=fire becomes 1 and is excluded from this smoke-only task.
    converted = converted_source_labels(joint_text)
    rows = [line for line in converted.splitlines() if line.split()[0]=='0']
    return '\n'.join(rows)+('\n' if rows else '')


def build_dataset(parent, expected_parent_sha256, out):
    parent=Path(parent).resolve(); out=Path(out).resolve()
    if out.exists(): raise ValueError('Refusing overwrite')
    config=yaml.safe_load((parent/'data.yaml').read_text(encoding='utf-8'))
    verified=verify_dataset(parent,config,expected_parent_sha256)
    source=json.loads((parent/'build_manifest.json').read_text(encoding='utf-8'))
    rows=[]; out.mkdir(parents=True)
    for row in source['images']:
        split=row['split']; filename=row['file']
        (out/'images'/split).mkdir(parents=True,exist_ok=True)
        (out/'labels'/split).mkdir(parents=True,exist_ok=True)
        image=out/'images'/split/filename; label=out/'labels'/split/(Path(filename).stem+'.txt')
        shutil.copyfile(parent/'images'/split/filename,image)
        text=smoke_only_labels((parent/'labels'/split/label.name).read_text(encoding='utf-8'))
        label.write_text(text,encoding='utf-8')
        rows.append({'file':filename,'split':split,'group':row['group'],'image_sha256':sha256(image),
                     'source_label_sha256':row['label_sha256'],'label_sha256':sha256(label),'smoke_boxes':len(text.splitlines())})
    manifest={'role':'Smoke head only, same reviewed exposed development splits; fire truth excluded from this single-class task',
              'parent':str(parent),'parent_manifest_sha256':expected_parent_sha256,
              'source_review_sha256':verified['source_review_sha256'],'script_sha256':sha256(Path(__file__)),
              'names':{0:'smoke'},'images':rows}
    (out/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    target={'path':str(out),'train':'images/train','val':'images/val','nc':1,'names':{0:'smoke'}}
    (out/'data.yaml').write_text(yaml.safe_dump(target,sort_keys=False),encoding='utf-8')
    return verify_smoke_dataset(out,sha256(out/'build_manifest.json'))


def verify_smoke_dataset(folder, expected_manifest_sha256):
    folder=Path(folder).resolve(); manifest_path=folder/'build_manifest.json'
    if sha256(manifest_path)!=expected_manifest_sha256: raise ValueError('Smoke manifest changed')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    config=yaml.safe_load((folder/'data.yaml').read_text(encoding='utf-8'))
    if config!={'path':str(folder),'train':'images/train','val':'images/val','nc':1,'names':{0:'smoke'}}:
        raise ValueError('Smoke class/path config changed')
    parent=Path(manifest['parent'])
    parent_check=verify_dataset(parent,yaml.safe_load((parent/'data.yaml').read_text(encoding='utf-8')),manifest['parent_manifest_sha256'])
    if parent_check['source_review_sha256']!=manifest['source_review_sha256']: raise ValueError('Review chain changed')
    original=json.loads((parent/'build_manifest.json').read_text(encoding='utf-8'))
    source={(r['split'],r['file']):r for r in original['images']}; seen=set(); images=set(); labels=set()
    for row in manifest['images']:
        key=(row['split'],row['file'])
        if key not in source or key in seen: raise ValueError('Missing/duplicate/unapproved child image')
        seen.add(key); old=source[key]
        if row['group']!=old['group'] or row['source_label_sha256']!=old['label_sha256']:
            raise ValueError('Source group/label changed')
        image=folder/'images'/row['split']/row['file']; label=folder/'labels'/row['split']/(Path(row['file']).stem+'.txt')
        images.add(image); labels.add(label)
        if sha256(image)!=old['image_sha256'] or row['image_sha256']!=old['image_sha256']: raise ValueError('Child image changed')
        expected=smoke_only_labels((parent/'labels'/row['split']/label.name).read_text(encoding='utf-8'))
        if label.read_text(encoding='utf-8')!=expected or sha256(label)!=row['label_sha256']:
            raise ValueError('Smoke-only class conversion changed')
        if row['smoke_boxes']!=len(expected.splitlines()): raise ValueError('Smoke count changed')
    if seen!=source.keys(): raise ValueError('Not every parent image is present')
    if images!={p for s in ('train','val') for p in (folder/'images'/s).iterdir() if p.is_file()} or labels!={p for s in ('train','val') for p in (folder/'labels'/s).iterdir() if p.is_file()}:
        raise ValueError('Unexpected child files')
    return {'verified_images':len(seen),'manifest_sha256':expected_manifest_sha256,'parent_manifest_sha256':manifest['parent_manifest_sha256'],
            'source_review_sha256':manifest['source_review_sha256'],'role':'exposed smoke-head development data, no independent test'}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--expected-parent-sha256',required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(build_dataset(a.parent,a.expected_parent_sha256,a.out),indent=2))


if __name__=='__main__':main()
