"""Fixed-control evaluation for smoke dataset extension v5.

Evaluates: (a) 9 exposed val dev images, (b) 16 extension training images,
and collects prediction-only diagnostics for 19 unknown cooking-plume images.
Same images, same inference parameters for every model. No causal or
generalization claims; extension data only increases training update volume.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_integrity import sha256, matches  # existing project module

torch.set_num_threads(2)

INFER_ARGS = dict(imgsz=640, conf=0.25, iou=0.6, device=0, rect=True, half=False, verbose=False)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='smoke extension dataset root (contains build_manifest.json)')
    ap.add_argument('--models', required=True, help='JSON list of {label,weights,expected_sha256}')
    ap.add_argument('--cooking-review', required=True, help='old review_complete.json')
    ap.add_argument('--out', required=True, help='output directory (must not exist)')
    return ap.parse_args()


def load_manifest(data_root: Path):
    manifest = json.loads((data_root / 'build_manifest.json').read_text(encoding='utf-8'))
    val, ext = [], []
    for row in manifest['images']:
        if row['split'] == 'val':
            val.append(row)
        elif row['origin'] == 'extension':
            ext.append(row)
    return val, ext


def load_cooking(review_path: Path):
    data = json.loads(review_path.read_text(encoding='utf-8'))
    rows = data['reviews'] if isinstance(data, dict) else data
    out = []
    for r in rows:
        img = Path(r['image'])
        digest_field = r.get('sha256') or r.get('image_sha256')
        out.append({'image': img, 'sha256': digest_field})
    return out


def yolo_to_xyxy(box, w, h):
    cx, cy, bw, bh = box
    return [ (cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h ]


def read_gt(label_path: Path, w, h):
    boxes = []
    if not label_path.is_file():
        raise ValueError('Missing label, cannot infer negative: '+str(label_path))
    if label_path.is_file():
        for line in label_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f'bad YOLO label line in {label_path}')
            cls, x, y, bw, bh = map(float, parts)
            if cls != 0:
                raise ValueError('Unexpected smoke class')
            boxes.append(yolo_to_xyxy((x, y, bw, bh), w, h))
    return boxes


def draw_boxes(img, boxes, color, thickness=2):
    for b in boxes:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, thickness)


def save_sheets(out_dir: Path, model_label: str, per_image, prefix):
    """Contact sheets, 4 images per page. GT green, predictions blue (no GT for unknowns)."""
    sheet_paths = []
    for page in range(math.ceil(len(per_image) / 4)):
        tiles = per_image[page*4:(page+1)*4]
        h, w = 400, 640
        canvas = np.full((h*2+30, w*2+30, 3), 255, np.uint8)
        for i, t in enumerate(tiles):
            im = t['im'].copy()
            if t.get('gt') is not None:
                draw_boxes(im, t['gt'], (0, 200, 0))
            draw_boxes(im, t['pred'], (255, 0, 0))
            ratio=min(w/im.shape[1],(h-30)/im.shape[0])
            im=cv2.resize(im,(round(im.shape[1]*ratio),round(im.shape[0]*ratio)))
            r, c = divmod(i, 2)
            y0, x0 = r*(h+10), c*(w+10)
            canvas[y0:y0+im.shape[0], x0:x0+im.shape[1]] = im
            gt_text=str(len(t['gt'])) if t['gt'] is not None else 'UNKNOWN'
            cv2.putText(canvas, f"{t['name'][:40]} GT={gt_text} P={len(t['pred'])}",
                        (x0+2, y0+im.shape[0]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        p = out_dir / f'contact_{prefix}_{model_label}_p{page:02d}.jpg'
        cv2.imwrite(str(p), canvas)
        sheet_paths.append(p.name)
    return sheet_paths


def evaluate_group(model, entries, out_dir, model_label, group_name, with_gt=True):
    results = []
    tp = fp = fn = 0
    images_with_fp = 0
    for e in entries:
        img_path = Path(e['file'])
        if sha256(img_path)!=e['image_sha256']:
            raise ValueError('Image changed: '+str(img_path))
        if with_gt and sha256(Path(e['label']))!=e['label_sha256']:
            raise ValueError('Label changed: '+str(e['label']))
        im = cv2.imread(str(img_path))
        if im is None:
            raise RuntimeError(f'cannot read {img_path}')
        h, w = im.shape[:2]
        pred = model(img_path, **INFER_ARGS)[0].boxes
        pred_xyxy = [[float(v) for v in box.xyxy[0].tolist()] for box in pred]
        gt = None
        m = None
        if with_gt:
            gt = read_gt(Path(e['label']), w, h)
            m = matches(gt, pred_xyxy)
            tp += m['tp']; fp += m['fp']; fn += m['fn']
            if m['fp'] > 0:
                images_with_fp += 1
        results.append({
            'image': str(img_path), 'image_sha256': e.get('image_sha256'),
            'gt_count': len(gt) if gt is not None else None,
            'pred_count': len(pred_xyxy),
            'pred_xyxy': pred_xyxy,
            'pred_confidence': [float(b.conf.item()) for b in pred],
            'gt_xyxy': gt,
            **({'tp': m['tp'], 'fp': m['fp'], 'fn': m['fn']} if m else {}),
        })
        per_img_vis = {'im': im, 'name': img_path.name, 'gt': gt, 'pred': pred_xyxy}
        results[-1]['_vis'] = per_img_vis  # stripped before writing
    sheets = save_sheets(out_dir, model_label, [r.pop('_vis') for r in results], group_name)
    summary = {'images': len(entries), 'images_with_fp': images_with_fp if with_gt else None,
               'tp': tp, 'fp': fp, 'fn': fn} if with_gt else {
               'images': len(entries),
               'images_with_predictions': sum(1 for r in results if r['pred_count'] > 0),
               'box_count': sum(r['pred_count'] for r in results)}
    return {'summary': summary, 'per_image': results, 'contact_sheets': sheets}


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'refusing to overwrite existing output directory: {out}')
    out.mkdir(parents=True)

    val_entries, ext_entries = load_manifest(Path(args.data))
    cooking = load_cooking(Path(args.cooking_review))
    if (len(val_entries),len(ext_entries),len(cooking))!=(9,16,19):
        raise ValueError('Fixed evaluation membership changed')
    models = json.loads(Path(args.models).read_text(encoding='utf-8'))

    summary = {'inference_args': {**INFER_ARGS, 'agnostic_nms': False},
               'script_sha256':sha256(Path(__file__)),
               'manifest_sha256':sha256(Path(args.data)/'build_manifest.json'),
               'models_sha256':sha256(Path(args.models)),
               'cooking_review_sha256':sha256(Path(args.cooking_review)),
               'note': 'Fixed-control evaluation on exposed development and extension training images only; '
                       'no causal or generalization claims.',
               'groups': {'val_dev': None, 'extension_train': None, 'unknown_cooking_plume': None},
               'models': {}}

    for m in models:
        weights = Path(m['weights'])
        digest = sha256(weights)
        if digest != m['expected_sha256']:
            raise SystemExit(f"sha256 mismatch for {m['label']}: {digest} != {m['expected_sha256']}")
        model = YOLO(str(weights))
        names = getattr(model.model, 'names', None) or model.names
        if dict(names) != {0: 'smoke'}:
            raise SystemExit(f"{m['label']}: names must be {{0:'smoke'}}, got {dict(names)}")

        mdir = out / m['label']
        mdir.mkdir()
        val_res = evaluate_group(model, val_entries, mdir, m['label'], 'val_dev', with_gt=True)
        ext_res = evaluate_group(model, ext_entries, mdir, m['label'], 'ext_train', with_gt=True)

        unk_entries = [{'file': str(c['image']), 'image_sha256': c['sha256']} for c in cooking]
        unk_res = evaluate_group(model, unk_entries, mdir, m['label'], 'unknown_cooking', with_gt=False)

        (mdir / 'val_dev.json').write_text(json.dumps(val_res, indent=2), encoding='utf-8')
        (mdir / 'extension_train.json').write_text(json.dumps(ext_res, indent=2), encoding='utf-8')
        (mdir / 'unknown_cooking_plume.json').write_text(json.dumps(unk_res, indent=2), encoding='utf-8')

        summary['models'][m['label']] = {
            'weights': str(weights), 'weights_sha256': digest,
            'val_dev': val_res['summary'],
            'extension_train': ext_res['summary'],
            'unknown_cooking_plume': unk_res['summary'],
            'per_image_files': ['val_dev.json', 'extension_train.json', 'unknown_cooking_plume.json'],
            'contact_sheets': val_res['contact_sheets'] + ext_res['contact_sheets'] + unk_res['contact_sheets'],
        }

    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'ok', 'out': str(out),
                      'models': list(summary['models'])}, indent=2))


if __name__ == '__main__':
    main()
