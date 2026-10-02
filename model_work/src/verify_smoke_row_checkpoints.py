"""Check saved state and actual unfused CPU fire output; excludes NMS guarantees."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from data_integrity import sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--start',type=Path,required=True)
    p.add_argument('--expected-start-sha256',required=True)
    p.add_argument('--fire-baseline',type=Path,required=True)
    p.add_argument('--expected-fire-sha256',required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--expected-review-sha256',required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise SystemExit('Refusing overwrite')
    for path,expected in ((a.start,a.expected_start_sha256),(a.fire_baseline,a.expected_fire_sha256),(a.review,a.expected_review_sha256)):
        if sha256(path)!=expected: raise SystemExit('Reference changed')
    torch.set_num_threads(2)
    start=YOLO(str(a.start)).model.float().eval()
    baseline=YOLO(str(a.fire_baseline)).model.float().eval()
    if start.names!={0:'fire',1:'smoke'} or baseline.names!={0:'fire'}: raise SystemExit('Class mapping changed')
    entries=json.loads(a.models.read_text(encoding='utf-8')); candidates={}
    for entry in entries:
        if sha256(Path(entry['weights']))!=entry['expected_sha256']: raise SystemExit('Candidate changed')
        candidates[entry['label']]=YOLO(entry['weights']).model.float().eval()
    if len(candidates)!=len(entries): raise SystemExit('Duplicate candidate label')
    protected_rows=set()
    for i,branch in enumerate(start.model[-1].cv3):
        prefix=f'model.{len(start.model)-1}.cv3.{i}.{len(branch)-1}'
        protected_rows.update((prefix+'.weight',prefix+'.bias'))
    state=start.state_dict(); results={}
    for label,model in candidates.items():
        other=model.state_dict()
        if other.keys()!=state.keys() or model.names!=start.names: raise SystemExit('Candidate state/class shape changed')
        changed=[]; smoke_max=0
        for key,value in state.items():
            expected=value[0] if key in protected_rows else value
            actual=other[key][0] if key in protected_rows else other[key]
            if not torch.equal(expected,actual): raise SystemExit(f'Saved protected state changed: {key}')
            if key in protected_rows:
                delta=float((other[key][1]-value[1]).abs().max())
                if delta>0: changed.append(key)
                smoke_max=max(smoke_max,delta)
        results[label]={'weights_sha256':next(x['expected_sha256'] for x in entries if x['label']==label),
                        'protected_saved_state_equal':True,'changed_smoke_tensors':len(changed),
                        'smoke_parameter_max_abs_delta':smoke_max,'raw_comparison':[]}
    rows=sorted((r for r in json.loads(a.review.read_text(encoding='utf-8'))['images'] if r['approved_for_training']),key=lambda r:r['sha256'])
    with torch.inference_mode():
        for row in rows:
            if sha256(Path(row['image']))!=row['sha256']: raise SystemExit('Reviewed image changed')
            im=cv2.imread(row['image'])
            if im is None: raise SystemExit('Decode failed')
            square=LetterBox(new_shape=(640,640),auto=False,stride=32)(image=im)
            tensor=torch.from_numpy(np.ascontiguousarray(square[:,:,::-1].transpose(2,0,1))).float().unsqueeze(0)/255
            fire=baseline(tensor)[0]; untrained=start(tensor)[0]
            torch.testing.assert_close(fire,untrained[:,:5],atol=1e-5,rtol=1e-6)
            for label,model in candidates.items():
                output=model(tensor)[0]
                if output.shape!=untrained.shape: raise SystemExit('Output shape changed')
                torch.testing.assert_close(fire,output[:,:5],atol=1e-5,rtol=1e-6)
                results[label]['raw_comparison'].append({'image_sha256':row['sha256'],
                    'fire_geometry_max_abs_vs_v3':float((output[:,:5]-fire).abs().max()),
                    'smoke_max':float(output[:,5].max())})
    record={'role':'Saved protected state and raw unfused CPU output check; NMS/class competition needs separate regression',
            'script_sha256':sha256(Path(__file__)),'review_sha256':sha256(a.review),
            'start_sha256':sha256(a.start),'fire_baseline_sha256':sha256(a.fire_baseline),
            'models_list_sha256':sha256(a.models),'configuration':{'images':len(rows),'imgsz':640,'rect':False,'device':'cpu','atol':1e-5,'rtol':1e-6},
            'models':results}
    a.out.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'images':len(rows),'models':{k:{'raw_max':max(x['fire_geometry_max_abs_vs_v3'] for x in v['raw_comparison']),
                    'saved_equal':v['protected_saved_state_equal'],'smoke_tensors_changed':v['changed_smoke_tensors']} for k,v in results.items()}}))


if __name__=='__main__':main()
