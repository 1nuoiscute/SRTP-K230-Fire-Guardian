"""Copy an approved fire dataset, applying only byte-bound reviewed label revisions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import yaml
from data_integrity import sha256, validate_yolo
from dataset_preflight import verify_reviewed_dataset


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reviewed_revisions(review):
    meta = read(review)
    for sheet in meta['sheets']:
        if Path(sheet['file']).name != sheet['file'] or sha256(review.parent / sheet['file']) != sheet['sha256']:
            raise ValueError('Review rendering changed')
    approved = {}
    for row in meta['images']:
        if row.get('approved_for_training') is not True:
            continue
        image, old, new = map(Path, (row['image'], row['original_label'], row['proposed_label']))
        for path, key in ((image, 'image_sha256'), (old, 'original_label_sha256'), (new, 'proposed_label_sha256')):
            if sha256(path) != row[key]:
                raise ValueError('Approved source/revision identity changed')
        old_text, new_text = old.read_text(), new.read_text()
        validate_yolo(new_text)
        if len(old_text.split()) != len(new_text.split()) or not new_text.strip():
            raise ValueError('Revision changed instance count')
        if image.name in approved:
            raise ValueError('Duplicate approved revision')
        approved[image.name] = row
    if not approved:
        raise ValueError('No approved revisions')
    return approved


def validation_snapshot(base):
    images = sorted((base / 'images/val').glob('*'))
    return [{'image': p.name, 'sha256': sha256(p),
             'label_sha256': sha256(base / 'labels/val' / (p.stem + '.txt'))} for p in images if p.is_file()]


def build(parent, review, out):
    parent, review, out = parent.resolve(), review.resolve(), out.resolve()
    if out.exists():
        raise ValueError('Output exists; never overwrite a dataset')
    source = read(parent / 'build_manifest.json')
    checked = verify_reviewed_dataset(parent, yaml.safe_load((parent / 'data.yaml').read_text()))
    approved = reviewed_revisions(review)
    if not set(approved).issubset({r['image'] for r in source['lineage']}):
        raise ValueError('Approved revision is outside the parent train split')
    base = Path(source['base'])
    lineage, entries = [], []
    (out / 'images/train').mkdir(parents=True)
    (out / 'labels/train').mkdir(parents=True)
    for row in source['lineage']:
        root = base if row['origin'] == 'legacy_train' else parent
        image = root / 'images/train' / row['image']
        label = root / 'labels/train' / (image.stem + '.txt')
        revision = approved.get(image.name)
        if revision and (row['origin'] != 'legacy_train' or row['sha256'] != revision['image_sha256'] or sha256(label) != revision['original_label_sha256']):
            raise ValueError('Revision does not match exact parent label')
        chosen = Path(revision['proposed_label']) if revision else label
        dst = out / 'images/train' / image.name
        shutil.copyfile(image, dst)
        shutil.copyfile(chosen, out / 'labels/train' / label.name)
        lineage.append({**row, 'label_sha256': sha256(chosen), 'parent_label_sha256': sha256(label),
                        'label_revised': revision is not None,
                        'teacher_preserve': row['origin'] == 'legacy_train' and revision is None})
        entries.extend([str(dst)] * row['weight'])
    (out / 'train.txt').write_text('\n'.join(entries) + '\n', encoding='utf-8')
    config = dict(path=str(out), train=str(out / 'train.txt'), val=str(base / 'images/val'), nc=1, names={0: 'fire'})
    (out / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    meta = dict(role='development_only_fire_label_revision', base=str(base), parent=str(parent),
                parent_identity=checked, review=str(review), review_sha256=sha256(review),
                unique_paths=len(lineage), train_entries=len(entries), label_revisions=len(approved),
                validation=validation_snapshot(base), lineage=lineage)
    (out / 'build_manifest.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return verify(out, sha256(out / 'build_manifest.json'))


def verify(folder, expected_sha):
    folder = Path(folder).resolve()
    manifest = folder / 'build_manifest.json'
    if sha256(manifest) != expected_sha:
        raise ValueError('Revision manifest changed')
    meta = read(manifest)
    if meta['role'] != 'development_only_fire_label_revision':
        raise ValueError('Wrong dataset role')
    parent = Path(meta['parent'])
    checked = verify_reviewed_dataset(parent, yaml.safe_load((parent / 'data.yaml').read_text()))
    if checked != meta['parent_identity'] or sha256(Path(meta['review'])) != meta['review_sha256']:
        raise ValueError('Parent or review changed')
    approved = reviewed_revisions(Path(meta['review']))
    original = {r['image']: r for r in read(parent / 'build_manifest.json')['lineage']}
    rows = {r['image']: r for r in meta['lineage']}
    if len(rows) != len(meta['lineage']) or rows.keys() != original.keys():
        raise ValueError('Parent membership changed')
    entries = Counter(Path(p).resolve() for p in (folder / 'train.txt').read_text().splitlines() if p)
    expected_entries = Counter()
    for name, row in rows.items():
        source = original[name]
        revision = approved.get(name)
        for key in ('image', 'sha256', 'origin', 'source_group', 'weight'):
            if row[key] != source[key]:
                raise ValueError('Source lineage/sampling changed')
        expected_label = revision['proposed_label_sha256'] if revision else source['label_sha256']
        if row['label_sha256'] != expected_label or row['parent_label_sha256'] != source['label_sha256'] or row['label_revised'] != (revision is not None):
            raise ValueError('Unapproved label revision')
        if row['teacher_preserve'] != (source['origin'] == 'legacy_train' and revision is None):
            raise ValueError('Teacher mask changed')
        image = folder / 'images/train' / name
        label = folder / 'labels/train' / (image.stem + '.txt')
        if sha256(image) != row['sha256'] or sha256(label) != expected_label:
            raise ValueError('Copied data bytes changed')
        validate_yolo(label.read_text())
        expected_entries[image] = row['weight']
    if entries != expected_entries or sum(entries.values()) != meta['train_entries'] or len(entries) != meta['unique_paths'] or len(approved) != meta['label_revisions']:
        raise ValueError('Train count or weight changed')
    base = Path(meta['base'])
    config = yaml.safe_load((folder / 'data.yaml').read_text())
    expected = dict(path=str(folder), train=str(folder / 'train.txt'), val=str(base / 'images/val'), nc=1, names={0: 'fire'})
    if config != expected or validation_snapshot(base) != meta['validation']:
        raise ValueError('Validation/config identity changed')
    return dict(manifest_sha256=sha256(manifest), config_sha256=sha256(folder / 'data.yaml'),
                train_list_sha256=sha256(folder / 'train.txt'), images_verified=len(rows),
                entries_verified=sum(entries.values()), label_revisions=len(approved),
                teacher_preserved_images=sum(r['teacher_preserve'] for r in rows.values()),
                scope='Approved development labels only; source bytes and sampling retained; not independent evaluation')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.parent, a.review, a.out), indent=2))
