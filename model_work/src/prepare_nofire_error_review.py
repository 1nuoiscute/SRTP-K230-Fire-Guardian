"""Render every fixed-source no-visible-fire error without changing labels or thresholds."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from data_integrity import sha256


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def prepare(cache,model,out):
    if out.exists(): raise ValueError('Existing no-fire error review')
    inputs=json.loads((cache/'local_inputs.json').read_text(encoding='utf-8'))
    predictions=json.loads((cache/(model+'_local_predictions.json')).read_text(encoding='utf-8'))
    summary=json.loads((cache/'summary.json').read_text(encoding='utf-8'))
    if summary['configuration']!=dict(imgsz=640,conf=.25,nms_iou=.6,match_iou=.5):
        raise ValueError('Prediction policy changed')
    original={r['file']:r for r in inputs}; selected=[]
    for r in predictions:
        if r['provenance']!='nofire_real_indoor' or not r['predictions']: continue
        source=original[r['image']]; image=Path(source['image'])
        label=image.parents[2]/'labels/test'/(image.stem+'.txt')
        if source['gt'] or sha256(image)!=source['sha256'] or sha256(label)!=source['label_sha256'] or label.read_text(encoding='utf-8').strip():
            raise ValueError('Source/empty original label changed')
        selected.append(dict(index=len(selected)+1,image=str(image),image_sha256=sha256(image),label_sha256=sha256(label),
            original_empty_fire_label=True,provenance=r['provenance'],predictions=r['predictions']))
    if not selected: raise ValueError('No diagnostic errors')
    out.mkdir(parents=True); tiles=[]; sheets=[]
    for r in selected:
        im=cv2.imread(r['image']); h,w=im.shape[:2]
        ratio=min(640/w,320/h); resized=cv2.resize(im,(round(w*ratio),round(h*ratio)))
        rh,rw=resized.shape[:2]; top,left=40+(320-rh)//2,(640-rw)//2
        tile=np.full((360,640,3),25,np.uint8); tile[top:top+rh,left:left+rw]=resized
        for b in r['predictions']:
            x1,y1,x2,y2=b['xyxy']; cv2.rectangle(tile,(round(x1*ratio+left),round(y1*ratio+top)),(round(x2*ratio+left),round(y2*ratio+top)),(0,165,255),2)
        cv2.putText(tile,f"{r['index']:02}: empty fire label / {len(r['predictions'])} boxes",(8,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1)
        tiles.append(tile)
    for start in range(0,len(tiles),4):
        page=tiles[start:start+4]
        while len(page)<4: page.append(np.zeros((360,640,3),np.uint8))
        path=out/f'sheet_{start//4+1:02}.jpg'
        if not cv2.imwrite(str(path),np.concatenate([np.concatenate(page[:2],1),np.concatenate(page[2:],1)],0)):
            raise ValueError('Encoding failed')
        sheets.append(dict(file=path.name,sha256=sha256(path)))
    write(out/'review_inputs.json',dict(role='All exposed no-fire-source FP images; visible semantics remain to be reviewed',
        original_labels_modified=False,independent_kitchen=False,source_cache_summary_sha256=sha256(cache/'summary.json'),
        source_inputs_sha256=sha256(cache/'local_inputs.json'),prediction_sha256=sha256(cache/(model+'_local_predictions.json')),
        model_weights_sha256=summary['models'][model]['weights_sha256'],script_sha256=sha256(Path(__file__)),images=selected,sheets=sheets,
        images_with_predictions=len(selected),prediction_boxes=sum(len(r['predictions']) for r in selected)))
    print(f'{len(selected)} complete source-error images / {len(sheets)} sheets',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--model',required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); prepare(a.cache,a.model,a.out)
