"""Freeze a new queue while preserving hash-bound roles from earlier queues."""
import argparse
import json
from pathlib import Path
from data_integrity import sha256
from freeze_kitchen_context_review import freeze, read, validate_rows


def validate_inheritance(review, plan_path):
    if sha256(plan_path) != review['inheritance_plan_sha256']:
        raise ValueError('Inheritance plan changed')
    plan = read(plan_path)
    parents = plan['inherited_reviews']
    if not parents:
        raise ValueError('Missing inherited reviews')
    seen = set()
    rows = []
    for item in parents:
        path = Path(item['path'])
        identity = str(path.resolve())
        if identity in seen:
            raise ValueError('Duplicate inherited review')
        seen.add(identity)
        if sha256(path) != item['sha256']:
            raise ValueError('Inherited review changed')
        parent = read(path)
        if not parent['rows']:
            raise ValueError('Empty inherited review')
        rows.extend(parent['rows'])
    validate_rows(rows + review['rows'])
    return parents


def freeze_inherited(queue, review_path, audit_path, inheritance_path, out):
    review = read(review_path)
    parents = validate_inheritance(review, inheritance_path)
    result = freeze(queue, review_path, audit_path, out)
    result.update(inheritance_plan_sha256=sha256(inheritance_path),
                  inherited_reviews=parents,
                  inheritance_guard_sha256=sha256(Path(__file__)))
    result['limits'].append('Author strings and manually assigned groups are conservative guards, not proof of scene independence.')
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['queue','review','audit','inheritance','out']:
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    freeze_inherited(a.queue,a.review,a.audit,a.inheritance,a.out)
