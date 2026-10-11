"""Verify an already completed independent CPU video CLI against PyTorch CPU.

Keep strict CPU export tolerances; report GPU comparisons separately without
relaxing limits or claiming cross-platform equality from matching box counts.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from data_integrity import sha256


def verify(comparison, runtime, weights, out):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops
    from ultralytics.utils.nms import non_max_suppression
    from verify_onnx_export import check_detections
    from pc_onnx_infer import preprocess
    if out.exists(): raise ValueError('Refusing overwrite')
    plan=json.loads((comparison/'plan.json').read_text(encoding='utf-8'))
    summary=json.loads((runtime/'summary.json').read_text(encoding='utf-8'))
    rows=list(csv.DictReader((runtime/'per_frame.csv').open(encoding='utf-8',newline='')))
    if sha256(runtime/'per_frame.csv')!=summary['csv_sha256']: raise ValueError('CLI CSV changed')
    candidates=[m for m in plan['models'] if m['expected_sha256']==sha256(weights)]
    sources=[v for v in plan['videos'] if v['sha256']==summary['source_sha256']]
    if len(candidates)!=1 or len(sources)!=1: raise ValueError('Model/source ambiguity')
    label=candidates[0]['label']; video=Path(sources[0]['path']); vi=plan['videos'].index(sources[0])+1
    if sha256(video)!=summary['source_sha256']: raise ValueError('Source changed')
    samples=[r for r in map(json.loads,(comparison/'per_frame.jsonl').read_text(encoding='utf-8').splitlines()) if r['video_index']==vi]
    if len(samples)!=len(rows) or len(rows)!=summary['sampled_frames']: raise ValueError('Incomplete samples')
    cli={int(r['frame_index']):r for r in rows}; wanted={r['frame_index']:r for r in samples}
    if set(cli)!=set(wanted): raise ValueError('Sample schedule changed')
    torch.set_num_threads(2); model=YOLO(str(weights)).model.cpu().float().eval(); model.fuse(verbose=False)
    letterbox=LetterBox(new_shape=(640,640),auto=True); details=[]; gpu_details=[]
    cap=cv2.VideoCapture(str(video)); index=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if index in wanted:
                r=wanted[index]
                if hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()!=r['source_pixel_sha256']: raise ValueError('Source pixels changed')
                tensor,_=preprocess(frame,True)
                native_tensor=np.ascontiguousarray(letterbox(image=frame)[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.
                preprocessing_equal=np.array_equal(tensor,native_tensor)
                with torch.inference_mode():
                    result=model(torch.from_numpy(native_tensor)); raw=result[0] if isinstance(result,tuple) else result
                    boxes=non_max_suppression(raw.clone(),conf_thres=.25,iou_thres=.6,nc=1,max_time_img=1.)[0]
                    if len(boxes): boxes[:,:4]=ops.scale_boxes(native_tensor.shape[2:],boxes[:,:4],frame.shape)
                native=boxes.cpu().numpy()
                actual=[b['xyxy']+[b['confidence'],0.] for b in json.loads(cli[index]['predictions'])]
                actual=np.array(actual,dtype=np.float32).reshape(-1,6)
                gpu=np.array([b['xyxy']+[b['confidence'],0.] for b in r['predictions'][label]],dtype=np.float32).reshape(-1,6)
                passed=preprocessing_equal and check_detections(native,actual)
                details.append(dict(frame_index=index,source_pixel_sha256=r['source_pixel_sha256'],preprocessing_equal=bool(preprocessing_equal),native_boxes=len(native),cli_boxes=len(actual),passed=bool(passed)))
                equal_count=len(actual)==len(gpu)
                gpu_details.append(dict(frame_index=index,equal_count=equal_count,strict_numeric_match=check_detections(gpu,actual),
                    coordinate_max_abs=float(np.abs(gpu[:,:4]-actual[:,:4]).max()) if equal_count and len(actual) else 0.,
                    confidence_max_abs=float(np.abs(gpu[:,4]-actual[:,4]).max()) if equal_count and len(actual) else 0.))
            index+=1
    finally: cap.release()
    if index!=summary['decoded_frames'] or len(details)!=len(rows): raise ValueError('Incomplete decode')
    if sha256(video)!=summary['source_sha256'] or sha256(weights)!=candidates[0]['expected_sha256'] or sha256(runtime/'per_frame.csv')!=summary['csv_sha256']: raise ValueError('Inputs changed')
    result=dict(role='CPU PyTorch versus independent ONNX video CLI; GPU numeric agreement separate',
        weights_sha256=sha256(weights),onnx_sha256=summary['model_sha256'],source_sha256=summary['source_sha256'],
        comparison_per_frame_sha256=sha256(comparison/'per_frame.jsonl'),cli_csv_sha256=summary['csv_sha256'],script_sha256=sha256(Path(__file__)),
        tolerance=dict(coordinates=.1,confidence=.0001),samples=len(details),passed=sum(r['passed'] for r in details),all_passed=all(r['passed'] for r in details),
        cpu_details=details,gpu_comparison=dict(all_counts_equal=all(r['equal_count'] for r in gpu_details),strict_passed=sum(r['strict_numeric_match'] for r in gpu_details),
            strict_all_passed=all(r['strict_numeric_match'] for r in gpu_details),coordinate_max_abs=max(r['coordinate_max_abs'] for r in gpu_details),
            confidence_max_abs=max(r['confidence_max_abs'] for r in gpu_details),details=gpu_details,
            note='GPU/CPU comparison is distinct from the frozen CPU export test; tolerances unchanged and differences retained.'))
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cpu_details','gpu_comparison']},indent=2))
    if not result['all_passed']: raise ValueError('CPU video pipeline parity failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True); p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); verify(a.comparison,a.runtime,a.weights,a.out)
