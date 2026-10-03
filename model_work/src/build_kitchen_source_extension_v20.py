"""Extend v19 with reviewed queue development roles; keep parent data and holdouts intact."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import shutil
from data_integrity import sha256, validate_yolo
from freeze_kitchen_context_review import validate_rows


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def label_text(row):
    w,h=row['width'],row['height']
    return ''.join(f'0 {(a+c)/(2*w):.9f} {(b+d)/(2*h):.9f} {(c-a)/w:.9f} {(d-b)/h:.9f}\n' for a,b,c,d in row['boxes'])


def development_rows(rows, positive_budget, negative_budget):
    validate_rows(rows)
    groups=defaultdict(list)
    for row in rows:
        if row['role']=='development': groups[row['source_group']].append(row)
    result=[]
    for group, members in sorted(groups.items()):
        kinds={r['visible_target'] for r in members}
        if not kinds <= {'flame','visible_no_flame'} or len(kinds)!=1:
            raise ValueError('Mixed or unknown new development group')
        budget=positive_budget if kinds=={'flame'} else negative_budget
        if isinstance(budget,bool) or not isinstance(budget,int) or budget<=0 or budget%len(members):
            raise ValueError('Group sampling budget cannot be split evenly')
        for row in sorted(members,key=lambda r:r['file']):
            result.append(dict(row,weight=budget//len(members)))
    return result


def reject_forbidden(rows, lineage):
    forbidden=[r for r in rows if r['role']!='development']
    identities={r['sha256'] for r in forbidden}
    groups={r['source_group'] for r in forbidden}
    authors={r['author'].strip().casefold() for r in forbidden}
    originals={r['original_sha1'] for r in forbidden}
    for item in lineage:
        if item['sha256'] in identities or item.get('source_group') in groups or item.get('author','').strip().casefold() in authors or item.get('original_sha1') in originals:
            raise ValueError('Reserved or held identity in training membership')


def reject_repeated_source_images(chosen, parent):
    known={r['sha256'] for r in parent['lineage']}
    known.update(r['sha256'] for r in parent['validation'])
    if len({r['sha256'] for r in chosen})!=len(chosen) or any(r['sha256'] in known for r in chosen):
        raise ValueError('Repeated or parent/validation image in new source extension')


def planned(plan_path):
    plan=read(plan_path)
    if len(plan['queues'])!=2: raise ValueError('Both frozen queues are mandatory')
    parent=Path(plan['parent'])
    from curate_train_context_v19 import verify as verify_parent
    verify_parent(parent,plan['parent_manifest_sha256'])
    meta=read(parent/'build_manifest.json'); review_rows=[]; frozen_sources={}
    for queue in plan['queues']:
        review_path=Path(queue['review']); folder=Path(queue['frozen_data'])
        if sha256(review_path)!=queue['review_sha256'] or sha256(folder/'manifest.json')!=queue['frozen_manifest_sha256']:
            raise ValueError('Frozen queue review or data manifest changed')
        review=read(review_path); manifest=read(folder/'manifest.json')
        if manifest['review_sha256']!=queue['review_sha256'] or not review['no_model_predictions_on_queue']:
            raise ValueError('Wrong queue approval identity')
        review_rows.extend(review['rows'])
        inventory={r['file']:r for r in manifest['inventory']}
        expected={r['file'] for r in review['rows'] if r['role']!='held'}
        if set(inventory)!=expected: raise ValueError('Frozen inventory omits or adds an approved image')
        for row in review['rows']:
            if row['role']=='held': continue
            item=inventory[row['file']]
            if item['role']!=row['role'] or item['source_group']!=row['source_group'] or item['image_sha256']!=row['sha256'] or item['boxes']!=len(row['boxes']):
                raise ValueError('Frozen role or label metadata changed')
            image=folder/row['role']/'images'/row['file'];label=folder/row['role']/'labels'/(Path(row['file']).stem+'.txt')
            if sha256(image)!=row['sha256'] or sha256(label)!=item['label_sha256'] or label.read_text(encoding='utf-8')!=label_text(row):
                raise ValueError('Frozen image or reviewed physical label changed')
            if row['role']=='development': frozen_sources[row['file']]=(image,label)
    chosen=development_rows(review_rows,plan['positive_entries_per_group'],plan['negative_entries_per_group'])
    # Validation bytes plus every parent train image are forbidden new-copy identities.
    reject_repeated_source_images(chosen,meta)
    rows=deepcopy(meta['lineage']); names={r['image'] for r in rows}
    for row in chosen:
        if row['file'] in names:raise ValueError('New source filename collides with parent')
        names.add(row['file']); image,label=frozen_sources[row['file']]
        rows.append(dict(image=row['file'],sha256=row['sha256'],label_sha256=sha256(label),
            origin='reviewed_commons_queue',source_group=row['source_group'],author=row['author'],original_sha1=row['original_sha1'],
            source_page=row['source_page'],license=row['license'],license_url=row['license_url'],weight=row['weight'],
            visible_flame_train_revision=bool(row['boxes']),boxes_xyxy=row['boxes'],teacher_preserve=False,
            v20_new_source=True,visible_target=row['visible_target'],full_kitchen_context=row['full_kitchen_context']))
    reject_forbidden(review_rows,rows)
    actual=[len(chosen),sum(bool(r['boxes']) for r in chosen),sum(len(r['boxes']) for r in chosen),sum(r['weight'] for r in chosen),len(rows),sum(r['weight'] for r in rows)]
    expected=[plan[k] for k in ['expected_new_images','expected_new_flame_images','expected_new_flame_boxes','expected_new_entries','expected_total_images','expected_total_entries']]
    if actual!=expected:raise ValueError('Frozen source membership or sampling counts differ')
    return plan,meta,rows,frozen_sources


def config(folder,base):
    return dict(path=str(folder.resolve()),train=str((folder/'train.txt').resolve()),val=str(Path(base)/'images/val'),nc=1,names={0:'fire'})


def build(plan_path,out):
    if out.exists():raise ValueError('Refusing existing source-extension output')
    plan,meta,rows,sources=planned(plan_path);parent=Path(plan['parent'])
    import yaml
    (out/'images/train').mkdir(parents=True);(out/'labels/train').mkdir(parents=True)
    entries=[]
    for row in rows:
        image,label=sources.get(row['image'],(parent/'images/train'/row['image'],parent/'labels/train'/(Path(row['image']).stem+'.txt')))
        target=out/'images/train'/row['image'];shutil.copyfile(image,target)
        shutil.copyfile(label,out/'labels/train'/label.name)
        entries.extend([str(target.resolve())]*row['weight'])
    (out/'train.txt').write_text('\n'.join(entries)+'\n',encoding='utf-8',newline='\n')
    (out/'data.yaml').write_text(yaml.safe_dump(config(out,meta['base']),sort_keys=False),encoding='utf-8',newline='\n')
    write(out/'build_manifest.json',dict(role='v20_reviewed_source_development_extension',plan=str(plan_path.resolve()),plan_sha256=sha256(plan_path),
        parent=str(parent.resolve()),parent_sha256=plan['parent_manifest_sha256'],base=meta['base'],builder_sha256=sha256(Path(__file__)),
        validation=meta['validation'],lineage=rows,excluded_lineage=meta['excluded_lineage'],unique_paths=len(rows),train_entries=len(entries),
        independent_kitchen=False,smoke_absence_unlabelled=True,train_diagnostics_exposed=True))
    return verify(out,sha256(out/'build_manifest.json'))


def verify(folder,expected):
    if sha256(folder/'build_manifest.json')!=expected:raise ValueError('Source-extension manifest changed')
    meta=read(folder/'build_manifest.json');plan_path=Path(meta['plan'])
    if sha256(plan_path)!=meta['plan_sha256'] or sha256(Path(__file__))!=meta['builder_sha256']:
        raise ValueError('Frozen extension plan or builder changed')
    plan,parent,rows,_=planned(plan_path)
    if meta['lineage']!=rows or meta['validation']!=parent['validation'] or meta['excluded_lineage']!=parent['excluded_lineage'] or meta['base']!=parent['base'] or meta['parent']!=str(Path(plan['parent']).resolve()) or meta['parent_sha256']!=plan['parent_manifest_sha256'] or meta['independent_kitchen'] or not meta['smoke_absence_unlabelled']:
        raise ValueError('Inherited data membership/scope changed')
    expected_paths=Counter()
    for row in rows:
        image=folder/'images/train'/row['image'];label=folder/'labels/train'/(Path(row['image']).stem+'.txt')
        if sha256(image)!=row['sha256'] or sha256(label)!=row['label_sha256']:raise ValueError('Extension image or label changed')
        validate_yolo(label.read_text(encoding='utf-8'));expected_paths[image.resolve()]=row['weight']
    import yaml
    actual=Counter(Path(p).resolve() for p in (folder/'train.txt').read_text(encoding='utf-8').splitlines())
    if actual!=expected_paths or yaml.safe_load((folder/'data.yaml').read_text(encoding='utf-8'))!=config(folder,parent['base']) or meta['unique_paths']!=len(rows) or meta['train_entries']!=sum(expected_paths.values()):
        raise ValueError('Weighted train membership/config changed')
    if {p.name for p in (folder/'images/train').iterdir()}!={r['image'] for r in rows} or {p.name for p in (folder/'labels/train').iterdir()}!={Path(r['image']).stem+'.txt' for r in rows}:
        raise ValueError('Unlisted extension files')
    return dict(manifest_sha256=expected,images_verified=len(rows),entries_verified=sum(expected_paths.values()),new_source_images=14,new_flame_boxes=4,validation_images=len(meta['validation']),independent_kitchen=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','verify']);p.add_argument('--plan',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--expected-sha256')
    a=p.parse_args();result=build(a.plan.resolve(),a.out.resolve()) if a.action=='build' else verify(a.out.resolve(),a.expected_sha256)
    print(json.dumps(result,indent=2),flush=True)
