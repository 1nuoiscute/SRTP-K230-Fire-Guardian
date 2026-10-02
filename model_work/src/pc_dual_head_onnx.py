"""CPU-only shared-feature independent fire/smoke ONNX runtime contract."""
import argparse
import ast
import json
from pathlib import Path
from data_integrity import sha256
from pc_onnx_infer import preprocess, postprocess, restore_boxes


def postprocess_dual(raw, conf=.25, nms_iou=.6, max_det=300, input_hw=(640,640)):
    import numpy as np
    if len(input_hw)!=2 or any(not isinstance(d,int) or d<=0 or d>640 or d%32 for d in input_hw):
        raise ValueError('Expected positive stride-32 input dimensions up to 640')
    anchors=sum((input_hw[0]//s)*(input_hw[1]//s) for s in (8,16,32))
    if raw.shape!=(1,6,anchors*2) or raw.dtype!=np.float32 or not np.isfinite(raw).all():
        raise ValueError('Expected finite FP32 independent dual-head output')
    if np.any(raw[:,5,:anchors]!=0) or np.any(raw[:,4,anchors:]!=0):
        raise ValueError('Independent class-score isolation violated')
    fire=postprocess(raw[:,:5,:anchors],conf,nms_iou,max_det,input_hw)
    smoke_raw=np.concatenate((raw[:,:4,anchors:],raw[:,5:6,anchors:]),axis=1)
    smoke=postprocess(smoke_raw,conf,nms_iou,max_det,input_hw); smoke[:,5]=1
    detections=np.concatenate((fire,smoke),axis=0)
    return detections[np.argsort(-detections[:,4],kind='stable')[:max_det]]


def make_dual_session(path, expected_sha256):
    import onnxruntime as ort
    if sha256(path)!=expected_sha256: raise ValueError('ONNX identity changed')
    settings=ort.SessionOptions(); settings.intra_op_num_threads=2; settings.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=settings,providers=['CPUExecutionProvider'])
    inputs,outputs=session.get_inputs(),session.get_outputs()
    if len(inputs)!=1 or inputs[0].type!='tensor(float)' or inputs[0].shape!=[1,3,640,640]:
        raise ValueError('Expected static FP32 square640 input')
    if len(outputs)!=1 or outputs[0].type!='tensor(float)' or outputs[0].shape!=[1,6,16800]:
        raise ValueError('Expected independent-head [1,6,16800] output')
    if ast.literal_eval(session.get_modelmeta().custom_metadata_map.get('names','{}'))!={0:'fire',1:'smoke'}:
        raise ValueError('Fire/smoke metadata differs')
    return session


def main():
    import cv2
    import numpy as np
    p=argparse.ArgumentParser(); p.add_argument('--model',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True); p.add_argument('--image',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if a.out.exists(): raise SystemExit('Refusing overwrite')
    session=make_dual_session(a.model,a.expected_sha256)
    im=cv2.imdecode(np.fromfile(a.image,dtype=np.uint8),cv2.IMREAD_COLOR)
    tensor,geometry=preprocess(im); raw=session.run(None,{session.get_inputs()[0].name:tensor})[0]
    detections=restore_boxes(postprocess_dual(raw),geometry)
    result={'role':'visual fire/smoke development evidence; no alarm/event classification',
            'model_sha256':a.expected_sha256,'image_sha256':sha256(a.image),'input_policy':'square640',
            'configuration':{'conf':.25,'nms_iou':.6,'max_det':300},'detections':detections.tolist(),
            'names':{0:'fire',1:'smoke'}}
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'detections':len(detections),'output':str(a.out)}))


if __name__=='__main__':main()
