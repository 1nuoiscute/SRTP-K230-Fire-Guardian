"""Select a fixed train-only industrial-stove pilot for explicit visible-flame review."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from data_integrity import sha256, validate_yolo

QUOTAS = {1:8,2:8,3:8,4:7,5:1}


def select(rows):
    if len(rows)!=1045 or len({r['group_id'] for r in rows})!=1045:
        raise ValueError('Expected unchanged complete industrial-stove training source')
    selected = []
    for count, quota in QUOTAS.items():
        eligible = sorted((r for r in rows if int(r['fire_boxes'])==count),key=lambda r:(r['group_id'],r['sha256']))
        if len(eligible)<quota:
            raise ValueError('Training stratum too small')
        # Midpoint quantiles avoid choosing frames in response to a model prediction.
        positions = [((2*i+1)*len(eligible))//(2*quota) for i in range(quota)]
        selected.extend(eligible[i] for i in positions)
    selected.sort(key=lambda r:(int(r['fire_boxes']),r['group_id'],r['sha256']))
    if len(selected)!=32 or len({r['file'] for r in selected})!=32:
        raise ValueError('Selection count or uniqueness differs')
    return selected


def prepare(data, out):
    if out.exists():
        raise ValueError('Existing train review; no overwrite')
    manifest = data/'manifest.csv'
    all_rows = list(csv.DictReader(manifest.open(encoding='utf-8-sig')))
    rows = [r for r in all_rows if r['split']=='train' and r['provenance']=='kitchen_stove_fire']
    selected = select(rows)
    forbidden = {r['sha256'] for r in all_rows if r['split']!='train'}
    if any(r['sha256'] in forbidden for r in selected):
        raise ValueError('Pilot overlaps existing val/test pixels')
    out.mkdir(parents=True)
    records, tiles, sheets = [], [], []
    for index, row in enumerate(selected,1):
        image = data/'images/train'/row['file']
        label = data/'labels/train'/(image.stem+'.txt')
        if sha256(image)!=row['sha256']:
            raise ValueError('Selected training image changed')
        text = label.read_text(encoding='utf-8'); validate_yolo(text)
        pixels = cv2.imread(str(image))
        if pixels is None or pixels.shape[:2]!=(640,640):
            raise ValueError('Expected readable native 640-square training image')
        boxes = []
        for line in text.splitlines():
            if not line.strip(): continue
            _,x,y,w,h = map(float,line.split())
            boxes.append([(x-w/2)*640,(y-h/2)*640,(x+w/2)*640,(y+h/2)*640])
        if len(boxes)!=int(row['fire_boxes']):
            raise ValueError('Original training label count differs')
        tile = np.full((680,640,3),25,np.uint8); tile[40:680]=pixels
        cv2.putText(tile,f'Index {index}: train originals / {len(boxes)} old labels / no predictions',(8,27),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1)
        tiles.append(tile)
        records.append(dict(index=index,image=str(image.resolve()),image_sha256=row['sha256'],label=str(label.resolve()),
                            label_sha256=sha256(label),size=[640,640],split='train',source_group=row['group_id'],
                            original_boxes_xyxy=boxes,status='pending_visible_flame_review',license=row['license']))
    for start in range(0,32,4):
        page=tiles[start:start+4]
        sheet=np.concatenate([np.concatenate(page[:2],1),np.concatenate(page[2:],1)],0)
        path=out/f'sheet_{start//4+1:02}.jpg'
        if not cv2.imwrite(str(path),sheet): raise ValueError('Review encoding failed')
        sheets.append(dict(file=path.name,sha256=sha256(path)))
    meta=dict(role='Train-only visible-flame semantics pilot; unrelated to unchanged active scratch experiment',
              source_manifest_sha256=sha256(manifest),selected_images=32,source_train_images=1045,
              source_selection='Midpoint quantiles of group_id within original box-count strata',quotas=QUOTAS,
              predictions_shown=False,scenes_are_development_exposed=True,independent_kitchen=False,
              no_exact_pixel_overlap_with_original_val_test=True,smoke_absence_unlabelled=True,
              original_labels_modified=False,script_sha256=sha256(Path(__file__)),images=records,sheets=sheets)
    (out/'review_inputs.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k not in ('images','sheets')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); prepare(a.data,a.out)
