"""Initialize standalone smoke head using v2 smoke head and original v3 features."""
import argparse
import copy
import json
from datetime import datetime,timezone
from pathlib import Path
import torch
import ultralytics
from ultralytics import YOLO
from data_integrity import sha256


def main():
    p=argparse.ArgumentParser(); p.add_argument('--fire',type=Path,required=True)
    p.add_argument('--expected-fire-sha256',required=True); p.add_argument('--joint',type=Path,required=True)
    p.add_argument('--expected-joint-sha256',required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise SystemExit('Refusing overwrite')
    if sha256(a.fire)!=a.expected_fire_sha256 or sha256(a.joint)!=a.expected_joint_sha256:
        raise SystemExit('Source checkpoint changed')
    fire=YOLO(str(a.fire)); joint=YOLO(str(a.joint))
    if fire.names!={0:'fire'} or joint.names!={0:'fire',1:'smoke'}: raise SystemExit('Source class mapping changed')
    model=copy.deepcopy(fire.model).float().eval(); learned=joint.model.float().eval()
    prefix=f'model.{len(model.model)-1}.'
    rows=set()
    for i,branch in enumerate(model.model[-1].cv3):
        key=prefix+f'cv3.{i}.{len(branch)-1}'
        rows.update((key+'.weight',key+'.bias'))
    source=learned.state_dict(); target=model.state_dict(); original={k:v.clone() for k,v in target.items()}
    with torch.no_grad():
        for key,value in target.items():
            if key.startswith(prefix):
                selected=source[key][1:2] if key in rows else source[key]
                if selected.shape!=value.shape: raise SystemExit('Head state shape mismatch')
                value.copy_(selected)
            elif not torch.equal(value,original[key]): raise SystemExit('Shared features changed')
    model.names={0:'smoke'}
    a.out.mkdir(parents=True); checkpoint=a.out/'start.pt'
    torch.save({'model':model.half(),'epoch':-1,'optimizer':None,'train_args':copy.deepcopy(fire.ckpt.get('train_args',{})),
                'date':datetime.now(timezone.utc).isoformat(),'version':ultralytics.__version__},checkpoint)
    reloaded=YOLO(str(checkpoint)).model.float()
    for key,value in original.items():
        if not key.startswith(prefix) and not torch.equal(reloaded.state_dict()[key],value):
            raise SystemExit('Saved shared features changed')
    record={'role':'standalone smoke head start; fixed original v3 features, v2 smoke-head initialization',
            'fire_sha256':a.expected_fire_sha256,'joint_v2_sha256':a.expected_joint_sha256,
            'start_sha256':sha256(checkpoint),'script_sha256':sha256(Path(__file__)),
            'source_head_class_row':1,'output_names':{0:'smoke'},'shared_features_saved_equal':True,
            'shared_blocks':len(model.model)-1,'smoke_head_parameter_elements':sum(p.numel() for p in model.model[-1].parameters())}
    (a.out/'initialization.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
