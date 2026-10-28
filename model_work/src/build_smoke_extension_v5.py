"""Append approved smoke appearances to train; preserve every v4 val member."""
import argparse
import json
from pathlib import Path
import shutil
import yaml
from data_integrity import sha256


def build(parent, review_path, out):
    if out.exists():
        raise ValueError('Existing output')
    old = json.loads((parent / 'build_manifest.json').read_text(encoding='utf-8'))
    review = json.loads(review_path.read_text(encoding='utf-8'))
    if sha256(review_path.with_name('decisions.json')) != review['decisions_sha256']:
        raise ValueError('Decisions changed')
    rows = []
    for split in ('train', 'val'):
        (out / 'images' / split).mkdir(parents=True)
        (out / 'labels' / split).mkdir(parents=True)
    for r in old['images']:
        image = parent / 'images' / r['split'] / r['file']
        label = parent / 'labels' / r['split'] / (image.stem + '.txt')
        if sha256(image) != r['image_sha256'] or sha256(label) != r['label_sha256']:
            raise ValueError('Parent bytes changed')
        target = out / 'images' / r['split'] / image.name
        dest = out / 'labels' / r['split'] / label.name
        shutil.copyfile(image, target)
        shutil.copyfile(label, dest)
        rows.append(dict(file=str(target), label=str(dest), split=r['split'], group=r['group'],
                         image_sha256=sha256(target), label_sha256=sha256(dest),
                         review_status='approved', origin='v4_parent'))
    existing = {r['image_sha256'] for r in rows}
    for r in review['images']:
        if not r['approved_for_training']:
            continue
        if r['sha256'] in existing or r['errors'] or r['split'] != 'train':
            raise ValueError('Repeated or invalid extension')
        source = Path(r['image'])
        if sha256(source) != r['sha256'] or sha256(Path(r['label'])) != r['label_sha256']:
            raise ValueError('Extension source changed')
        name = 'extension_' + r['sha256'][:24] + source.suffix.lower()
        image = out / 'images/train' / name
        label = out / 'labels/train' / (image.stem + '.txt')
        shutil.copyfile(source, image)
        # Raw D-Fire 0=smoke; fire targets are outside this single-class task.
        lines = ['0 ' + line.partition(' ')[2] for line in Path(r['label']).read_text(encoding='utf-8-sig').splitlines()
                 if line.strip() and int(line.split()[0]) == 0]
        label.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8', newline='\n')
        rows.append(dict(file=str(image), label=str(label), split='train', group=r['reviewed_development_group'],
                         image_sha256=sha256(image), label_sha256=sha256(label), review_status='approved',
                         origin='extension', review_index=r['index'], source_sha256=r['sha256']))
        existing.add(r['sha256'])
    config = dict(path=str(out), train='images/train', val='images/val', nc=1, names={0: 'smoke'})
    (out / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    manifest = dict(role='Visible smoke appearance extension; exposed development only',
                    parent=str(parent), parent_manifest_sha256=sha256(parent / 'build_manifest.json'),
                    source_review=str(review_path), source_review_sha256=sha256(review_path), images=rows,
                    independent_test=False, event_truth=False,
                    comparison_limit='Same epoch schedule with more train images changes optimizer update budget')
    (out / 'build_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(images=len(rows), new_images=sum(r['origin']=='extension' for r in rows),
                          val=sum(r['split']=='val' for r in rows))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    build(a.parent.resolve(), a.review.resolve(), a.out.resolve())
