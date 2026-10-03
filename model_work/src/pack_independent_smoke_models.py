"""Pack shared-feature dual-head inference candidates and check original fire path."""
import argparse
import copy
import json
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from independent_smoke_head import IndependentSmokeDetect
from data_integrity import sha256


def main():
    p=argparse.ArgumentParser(); p.add_argument('--fire',type=Path,required=True)
    p.add_argument('--expected-fire-sha256',required=True); p.add_argument('--smoke-models',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True); p.add_argument('--expected-review-sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise SystemExit('Refusing overwrite')
    if sha256(a.fire)!=a.expected_fire_sha256 or sha256(a.review)!=a.expected_review_sha256:
        raise SystemExit('Source identity changed')
    torch.set_num_threads(2); source=YOLO(str(a.fire)); original=source.model.float().eval()
    if source.names!={0:'fire'}: raise SystemExit('Expected original fire model')
    entries=json.loads(a.smoke_models.read_text(encoding='utf-8')); names=[r['label'] for r in entries]
    if len(names)!=len(set(names)) or any(not name.replace('_','').isalnum() for name in names):
        raise SystemExit('Duplicate/unsafe candidate name')
    prefix=f'model.{len(original.model)-1}.'; source_state=original.state_dict()
    a.out.mkdir(parents=True); models={}; listing=[]; result={}
    base_parameters=sum(p.numel() for p in original.parameters())
    for entry in entries:
        if sha256(Path(entry['weights']))!=entry['expected_sha256']: raise SystemExit('Smoke checkpoint changed')
        student=YOLO(entry['weights']).model.float().eval()
        if student.names!={0:'smoke'} or student.state_dict().keys()!=source_state.keys():
            raise SystemExit('Standalone smoke state/class mapping differs')
        for key,value in source_state.items():
            if not key.startswith(prefix) and not torch.equal(student.state_dict()[key],value):
                raise SystemExit(f'Shared features changed in saved student: {key}')
        combined=copy.deepcopy(original)
        combined.model[-1]=IndependentSmokeDetect(copy.deepcopy(original.model[-1]),copy.deepcopy(student.model[-1]))
        combined.names={0:'fire',1:'smoke'}; combined.yaml=copy.deepcopy(combined.yaml); combined.yaml['nc']=2
        # Prevent YOLO.train() silently rebuilding a standard head from the
        # inherited YAML and dropping both packed heads during weight loading.
        combined.yaml['inference_only']=True
        combined.yaml['head'][-1][2]='IndependentSmokeDetectInferenceOnly'
        combined.yaml['head'][-1][3]=[]
        combined.eval(); checkpoint=a.out/(entry['label']+'.pt')
        torch.save({'model':combined.half(),'epoch':-1,'optimizer':None,'train_args':copy.deepcopy(source.ckpt.get('train_args',{})),
                    'date':datetime.now(timezone.utc).isoformat(),'version':ultralytics.__version__},checkpoint)
        loaded=YOLO(str(checkpoint)).model.float().eval()
        for i in range(len(original.model)-1):
            if any(not torch.equal(v,loaded.model[i].state_dict()[k]) for k,v in original.model[i].state_dict().items()):
                raise SystemExit('Combined saved shared feature changed')
        fire_head=loaded.model[-1].fire_head.state_dict()
        if fire_head.keys()!=original.model[-1].state_dict().keys() or any(not torch.equal(v,fire_head[k]) for k,v in original.model[-1].state_dict().items()):
            raise SystemExit('Combined saved fire head changed')
        total=sum(p.numel() for p in loaded.parameters()); label=entry['label']; models[label]=loaded
        listing.append({'label':label,'weights':str(checkpoint.resolve()),'expected_sha256':sha256(checkpoint)})
        result[label]={'source_smoke_sha256':entry['expected_sha256'],'weights_sha256':sha256(checkpoint),
                       'protected_saved_state_equal':True,'parameters':total,'baseline_parameters':base_parameters,
                       'added_parameters':total-base_parameters,'raw_comparison':[]}
    rows=sorted((r for r in json.loads(a.review.read_text(encoding='utf-8'))['images'] if r['approved_for_training']),key=lambda r:r['sha256'])
    with torch.inference_mode():
        for row in rows:
            if sha256(Path(row['image']))!=row['sha256']: raise SystemExit('Reviewed image changed')
            im=cv2.imread(row['image'])
            if im is None: raise SystemExit('Decode failed')
            square=LetterBox(new_shape=(640,640),auto=False,stride=32)(image=im)
            tensor=torch.from_numpy(np.ascontiguousarray(square[:,:,::-1].transpose(2,0,1))).float().unsqueeze(0)/255
            reference=original(tensor)[0]; anchors=reference.shape[-1]
            for label,model in models.items():
                raw=model(tensor)[0]
                if raw.shape!=(1,6,anchors*2): raise SystemExit('Combined raw output layout changed')
                fire=raw[:,:5,:anchors]; torch.testing.assert_close(fire,reference,atol=1e-5,rtol=1e-6)
                if not bool(torch.all(raw[:,5,:anchors]==0)) or not bool(torch.all(raw[:,4,anchors:]==0)):
                    raise SystemExit('Independent class score isolation failed')
                result[label]['raw_comparison'].append({'image_sha256':row['sha256'],
                    'fire_geometry_max_abs_vs_v3':float((fire-reference).abs().max()),'smoke_max':float(raw[:,5,anchors:].max())})
    record={'role':'shared-feature independent-head inference, no training or board acceptance',
            'script_sha256':sha256(Path(__file__)),'head_module_sha256':sha256(Path(__file__).with_name('independent_smoke_head.py')),
            'fire_baseline_sha256':a.expected_fire_sha256,'review_sha256':a.expected_review_sha256,
            'input_smoke_models_sha256':sha256(a.smoke_models),'configuration':{'images':len(rows),'imgsz':640,'device':'cpu','rect':False},
            'models':result}
    (a.out/'raw_saved_check.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    (a.out/'models.json').write_text(json.dumps(listing,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'models':{k:{'sha256':v['weights_sha256'],'parameters':v['parameters'],'added_parameters':v['added_parameters'],
                      'raw_max':max(x['fire_geometry_max_abs_vs_v3'] for x in v['raw_comparison'])} for k,v in result.items()}}))


if __name__=='__main__':main()
