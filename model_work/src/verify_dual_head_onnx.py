"""Export and validate static CPU dual-head raw/NMS parity on bound development inputs."""
import argparse
import json
import random
import shutil
import time
from pathlib import Path
from data_integrity import sha256
from pc_onnx_infer import preprocess
from pc_dual_head_onnx import make_dual_session, postprocess_dual
from verify_onnx_export import check_detections


def main():
    import cv2
    import numpy as np
    import onnx
    import onnxruntime
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.nms import non_max_suppression
    import yaml
    from joint_pilot_dataset import verify_dataset
    p=argparse.ArgumentParser(); p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True); p.add_argument('--pilot-data',type=Path,required=True)
    p.add_argument('--expected-manifest-sha256',required=True); p.add_argument('--evaluation-inputs',type=Path,required=True)
    p.add_argument('--expected-inputs-sha256',required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise SystemExit('Refusing overwrite')
    if sha256(a.weights)!=a.expected_sha256 or sha256(a.evaluation_inputs)!=a.expected_inputs_sha256:
        raise SystemExit('Checkpoint/evaluation inputs changed')
    verify_dataset(a.pilot_data,yaml.safe_load((a.pilot_data/'data.yaml').read_text(encoding='utf-8')),a.expected_manifest_sha256)
    manifest=json.loads((a.pilot_data/'build_manifest.json').read_text(encoding='utf-8')); inputs=[]
    for row in manifest['images']:
        inputs.append({'path':str(a.pilot_data/'images'/row['split']/row['file']),'sha256':row['image_sha256'],'group':'reviewed_'+row['split']})
    known=json.loads(a.evaluation_inputs.read_text(encoding='utf-8')); rng=random.Random(20261002)
    for source in sorted({row['provenance'] for row in known['legacy']}):
        rows=sorted((row for row in known['legacy'] if row['provenance']==source),key=lambda row:row['image_sha256'])
        for row in rng.sample(rows,4): inputs.append({'path':row['image'],'sha256':row['image_sha256'],'group':'legacy_'+source})
    for row in known['blue']: inputs.append({'path':row['image'],'sha256':row['image_sha256'],'group':'blue_primary'})
    for row in inputs:
        if sha256(Path(row['path']))!=row['sha256']: raise SystemExit('Diagnostic image changed')
    a.out.mkdir(parents=True); torch.set_num_threads(2)
    dump=lambda name,data:(a.out/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    config={'imgsz':640,'batch':1,'dynamic':False,'half':False,'opset':12,'simplify':False,'embedded_nms':False,
            'conf':.25,'nms_iou':.6,'max_det':300,'raw_geometry_tolerance':.1,'probability_tolerance':.0001,'nms_xyxy_tolerance':.1}
    dump('launch.json',{'weights_sha256':a.expected_sha256,'script_sha256':sha256(Path(__file__)),
                        'runtime_sha256':sha256(Path(__file__).with_name('pc_dual_head_onnx.py')),'configuration':config,
                        'inputs':len(inputs),'role':'Exposed FP32 CPU export consistency; no KModel/board/event acceptance'})
    dump('local_inputs.json',inputs); checkpoint=a.out/'model.pt'; shutil.copyfile(a.weights,checkpoint)
    exported=Path(YOLO(str(checkpoint)).export(format='onnx',imgsz=640,batch=1,dynamic=False,half=False,opset=12,simplify=False,nms=False,device='cpu'))
    onnx.checker.check_model(str(exported),full_check=True)
    session=make_dual_session(exported,sha256(exported)); name=session.get_inputs()[0].name
    reference=YOLO(str(checkpoint)).model.cpu().float().eval(); reference.fuse(verbose=False)
    for _ in range(3):session.run(None,{name:np.zeros((1,3,640,640),dtype=np.float32)})
    details=[]
    for index,row in enumerate(inputs,1):
        if sha256(Path(row['path']))!=row['sha256']: raise SystemExit('Input changed during verification')
        im=cv2.imdecode(np.fromfile(row['path'],dtype=np.uint8),cv2.IMREAD_COLOR)
        tensor,_=preprocess(im)
        expected=np.ascontiguousarray(LetterBox(new_shape=(640,640),auto=False)(image=im)[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
        if not np.array_equal(tensor,expected): raise RuntimeError('Preprocessing differs')
        with torch.inference_mode(): native=reference(torch.from_numpy(tensor))[0].numpy()
        before=time.perf_counter(); actual=session.run(None,{name:tensor})[0]; elapsed=(time.perf_counter()-before)*1000
        if native.shape!=actual.shape or not np.isfinite(actual).all(): raise RuntimeError('Raw output shape/finite mismatch')
        geometry=float(np.abs(native[:,:4]-actual[:,:4]).max()); probability=float(np.abs(native[:,4:]-actual[:,4:]).max())
        if geometry>.1 or probability>.0001: raise RuntimeError('Raw parity tolerance exceeded')
        baseline=non_max_suppression(torch.from_numpy(native.copy()),conf_thres=.25,iou_thres=.6,nc=2,max_det=300)[0].numpy()
        decoded=postprocess_dual(actual)
        if not check_detections(baseline,decoded,pixel_tolerance=.1,confidence_tolerance=.0001): raise RuntimeError('CPU NMS parity failed')
        details.append({'index':index,'image_sha256':row['sha256'],'group':row['group'],'raw_geometry_max_abs':geometry,
                        'raw_probability_max_abs':probability,'detections':len(decoded),'nms_equal_within_tolerance':True,'session_only_ms':elapsed})
        if index%10==0: print(f'PARITY {index}/{len(inputs)}',flush=True)
    dump('details.json',details)
    result={'role':'FP32 static CPU dual-head export parity; no quantization, board or event acceptance',
            'weights_sha256':a.expected_sha256,'onnx_sha256':sha256(exported),'script_sha256':sha256(Path(__file__)),
            'runtime_sha256':sha256(Path(__file__).with_name('pc_dual_head_onnx.py')),'inputs_sha256':sha256(a.out/'local_inputs.json'),
            'details_sha256':sha256(a.out/'details.json'),'configuration':config,'verified_images':len(details),
            'raw_geometry_max_abs':max(row['raw_geometry_max_abs'] for row in details),
            'raw_probability_max_abs':max(row['raw_probability_max_abs'] for row in details),'nms_matched_images':len(details),
            'session_only_latency_ms':{'median':float(np.median([row['session_only_ms'] for row in details])),
                                       'p95':float(np.percentile([row['session_only_ms'] for row in details],95))},
            'environment':{'torch':torch.__version__,'ultralytics':ultralytics.__version__,'onnx':onnx.__version__,'onnxruntime':onnxruntime.__version__}}
    dump('verification_complete.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
