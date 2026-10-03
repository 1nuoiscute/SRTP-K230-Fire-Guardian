"""Paired train-positive learnability diagnostic, not candidate acceptance.

Four arms: full/oracle-crop inputs x head-only/late-feature adaptation.
Same v16 initialization, 4 positives, 80 actual FP32 AdamW updates, no
augmentation/EMA/scheduler. All BN running statistics stay fixed. Reserved
sources are never predicted or optimized. No trained weights are exported.
"""
import argparse
import json
import os
from pathlib import Path

from data_integrity import iou, matches, sha256

START_SHA = 'e770cf68d12cd43b4494a1439dc9c65f14d1d5fc570b2e2e379cc084384418fd'
DATA_SHA = '15a509075be0d2ff062894cd7de1160699bbdc7b657d8199bfba77cb80bf36fa'
ARMS = [('full_head', 'full', 23), ('full_late', 'full', 10),
        ('crop_head', 'crop', 23), ('crop_late', 'crop', 10)]
CHECKPOINTS = [0, 20, 40, 80]
DEPENDENCIES = ['data_integrity.py', 'pc_onnx_infer.py',
                'build_kitchen_source_extension_v20.py', 'backbone_freeze_audit.py',
                'optimizer_cadence_audit.py']


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                          encoding='utf-8', newline='\n')


def crop_bounds(width, height, box):
    """Fixed, pre-prediction GT oracle context; never a deployment crop policy."""
    import math
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    cw, ch = max(128, 3*(x2-x1)), max(128, 3*(y2-y1))
    return [max(0, math.floor(cx-cw/2)), max(0, math.floor(cy-ch/2)),
            min(width, math.ceil(cx+cw/2)), min(height, math.ceil(cy+ch/2))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['start', 'data', 'reference', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise ValueError('Refusing existing diagnostic output')
    if sha256(args.start) != START_SHA or sha256(args.data/'build_manifest.json') != DATA_SHA:
        raise ValueError('Wrong source or dataset')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from backbone_freeze_audit import is_backbone_key, state_digest
    from optimizer_cadence_audit import OptimizerCadenceAudit
    from build_kitchen_source_extension_v20 import verify
    from pc_onnx_infer import preprocess, postprocess, restore_boxes
    from types import SimpleNamespace

    dataset = verify(args.data, DATA_SHA)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for fixed diagnostic')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    meta = json.loads((args.data/'build_manifest.json').read_text(encoding='utf-8'))
    selected = sorted([r for r in meta['lineage'] if r.get('v20_new_source')], key=lambda r:r['image'])
    positive = [r for r in selected if r['boxes_xyxy']]
    if len(selected) != 14 or len(positive) != 4 or any(len(r['boxes_xyxy']) != 1 for r in positive):
        raise ValueError('Wrong train-only diagnostic membership')
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    if reference['weights_sha256'] != START_SHA or reference['config'] != dict(imgsz=640,conf=.25,nms_iou=.6,match_iou=.5):
        raise ValueError('Wrong existing baseline comparison')
    previous = {r['image']:r for r in reference['frames']}
    if set(previous) != {r['image'] for r in selected}:
        raise ValueError('Baseline scope differs')
    images, crop_images, crop_gt, crop_coordinates = {}, {}, {}, {}
    for row in selected:
        path = args.data/'images/train'/row['image']
        label = args.data/'labels/train'/(Path(row['image']).stem+'.txt')
        if sha256(path) != row['sha256'] or sha256(label) != row['label_sha256']:
            raise ValueError('Diagnostic source bytes changed')
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('Cannot decode source')
        images[row['image']] = image
        if row['boxes_xyxy']:
            h,w = image.shape[:2]
            a,b,c,d = crop_bounds(w,h,row['boxes_xyxy'][0])
            crop_coordinates[row['image']] = [a,b,c,d]
            crop_images[row['image']] = image[b:d,a:c].copy()
            x1,y1,x2,y2 = row['boxes_xyxy'][0]
            crop_gt[row['image']] = [[x1-a,y1-b,x2-a,y2-b]]
    out.mkdir(parents=True)
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    sources = {}
    for name in [Path(__file__).name]+DEPENDENCIES:
        path = Path(__file__).with_name(name)
        payload = path.read_bytes()
        (snapshot/name).write_bytes(payload)
        sources[name] = sha256(path)
    plan = dict(role='Training-exposed learnability experiment; no model promotion or independent test',
        source_sha256=START_SHA, dataset=dataset, reference_sha256=sha256(args.reference),
        arms=[dict(name=n,input=v,frozen_modules=f) for n,v,f in ARMS], steps=80,
        checkpoints=CHECKPOINTS, seed=20261007, optimizer='AdamW',lr=.00005,weight_decay=.0005,
        batch_size=4,imgsz=640,training_padding='square640',conf=.25,nms_iou=.6,match_iou=.5,
        gradient_clip_norm=10,amp=False,ema=False,augment=False,scheduler=False,
        all_bn_running_statistics_fixed=True,oracle_crop_multiplier=3,oracle_crop_min_side=128,
        oracle_crop_requires_ground_truth=True,trained_weights_saved=False,
        source_files=sources, inputs=[dict(image=r['image'],sha256=r['sha256'],label_sha256=r['label_sha256'],
             source_group=r['source_group'],boxes=r['boxes_xyxy'],crop_xyxy=crop_coordinates.get(r['image'])) for r in selected],
        environment=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,gpu=torch.cuda.get_device_name(0)))
    write(out/'plan.json', plan)

    def train_mode(core):
        core.train()
        for module in core.modules():
            if isinstance(module,torch.nn.BatchNorm2d):
                module.eval()

    def predictions(core, image, gt, rectangular):
        tensor,geometry = preprocess(image, rectangular=rectangular)
        core.eval()
        with torch.no_grad():
            raw = core(torch.from_numpy(tensor).cuda())[0].detach().cpu().numpy()
        boxes = restore_boxes(postprocess(raw,.25,.6,input_hw=tuple(tensor.shape[2:])),geometry)
        items = [dict(xyxy=[float(v) for v in b[:4]],confidence=float(b[4])) for b in boxes]
        xyxy = [r['xyxy'] for r in items]
        return dict(predictions=items,**matches(gt,xyxy),
            best_iou=[max((iou(g,b) for b in xyxy),default=0) for g in gt],
            input_hw=list(tensor.shape[2:]), gt_input_wh=[[(g[2]-g[0])*geometry['gain'],(g[3]-g[1])*geometry['gain']] for g in gt])

    def evaluate(core):
        result = {}
        for view,rows,bank,gts,rect in [('full_rect',selected,images,None,True),
             ('full_square',positive,images,None,False),('oracle_crop_square',positive,crop_images,crop_gt,False)]:
            frames = [dict(image=r['image'],**predictions(core,bank[r['image']],
                gts[r['image']] if gts else r['boxes_xyxy'],rect)) for r in rows]
            totals = {k:sum(r[k] for r in frames) for k in ['tp','fp','fn']}
            result[view] = dict(frames=frames,**totals,
                negative_images_with_boxes=sum(bool(r['predictions']) for r in frames if not r['best_iou']))
        return result

    def batch_for(view):
        bank = images if view=='full' else crop_images
        tensors, normalized = [], []
        for row in positive:
            tensor,g = preprocess(bank[row['image']], rectangular=False)
            box = (row['boxes_xyxy'] if view=='full' else crop_gt[row['image']])[0]
            x1,y1,x2,y2 = box
            normalized.append([((x1+x2)/2*g['gain']+g['left'])/640,
                ((y1+y2)/2*g['gain']+g['top'])/640,(x2-x1)*g['gain']/640,(y2-y1)*g['gain']/640])
            tensors.append(tensor)
        return dict(img=torch.from_numpy(np.concatenate(tensors)).cuda(),
            cls=torch.zeros((4,1),device='cuda'),batch_idx=torch.arange(4,device='cuda'),
            bboxes=torch.tensor(normalized,dtype=torch.float32,device='cuda'))

    completed, baseline = {}, None
    for name,view,blocks in ARMS:
        torch.manual_seed(20261007)
        torch.cuda.manual_seed_all(20261007)
        arm = out/name
        arm.mkdir()
        wrapper = YOLO(str(args.start))
        core = wrapper.model.cuda().float()
        if len(core.model)!=24 or core.names!={0:'fire'}:
            raise ValueError('Unexpected architecture')
        initial = {k:v.detach().cpu().clone() for k,v in core.state_dict().items()}
        if state_digest(initial) != 'aa54d71b4f23ab60923a8be6539784628876891e299ed908386549fae88fe611':
            raise ValueError('Actual initialization differs from audited v16')
        core.args = get_cfg(overrides={'box':7.5,'cls':.5,'dfl':1.5})
        for n,p in core.named_parameters():
            p.requires_grad_(not is_backbone_key(n,blocks) and '.dfl.' not in n)
        optimizer = torch.optim.AdamW([p for p in core.parameters() if p.requires_grad],lr=.00005,weight_decay=.0005)
        cadence = OptimizerCadenceAudit(arm/'optimizer_cadence_audit.json')
        cadence.attach(optimizer)
        trainer = SimpleNamespace(optimizer=optimizer,epoch=0)
        batch = batch_for(view)
        records, losses, audits = [], [], []

        def audit_state(step):
            state = {k:v.detach().cpu() for k,v in core.state_dict().items()}
            changed = [k for k,v in state.items() if not torch.equal(v,initial[k])]
            if any(is_backbone_key(k,blocks) or '.dfl.' in k or k.endswith(('running_mean','running_var','num_batches_tracked')) for k in changed):
                raise RuntimeError('Fixed features, DFL or BN statistics changed')
            if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):
                raise RuntimeError('Nonfinite model state')
            head = [k for k in changed if k.startswith('model.23.') and k.endswith('.weight')]
            late = [k for k in changed if is_backbone_key(k,23) and k.endswith('.weight')]
            if step and (not head or (blocks==10 and not late) or (blocks==23 and late)):
                raise RuntimeError('Wrong actual learned scope')
            if len(cadence.steps)!=step or cadence.batches!=step:
                raise RuntimeError('Wrong actual optimizer updates')
            audits.append(dict(step=step,frozen_state_unchanged=True,bn_statistics_unchanged=True,
                changed_head_weight_tensors=len(head),changed_feature_weight_tensors=len(late),
                model_state_sha256=state_digest(state),trainable_parameters=sum(p.numel() for p in core.parameters() if p.requires_grad)))
            write(arm/'state_audit.json',audits)

        try:
            for step in range(81):
                if step in CHECKPOINTS:
                    audit_state(step)
                    measured = evaluate(core)
                    if step==0:
                        if baseline is None:
                            comparisons=[]
                            for frame in measured['full_rect']['frames']:
                                before = previous[frame['image']]
                                counts_equal=all(frame[k]==before[k] for k in ['tp','fp','fn']) and len(frame['predictions'])==len(before['predictions'])
                                if not counts_equal:
                                    raise RuntimeError('Baseline counts differ from fixed comparison')
                                deltas=[dict(coordinate_max_abs=max(abs(x-y) for x,y in zip(a['xyxy'],b['xyxy'])),
                                    confidence_abs=abs(a['confidence']-b['confidence']))
                                    for a,b in zip(frame['predictions'],before['predictions'])]
                                comparisons.append(dict(image=frame['image'],counts_equal=counts_equal,deltas=deltas,
                                    strict_passed=all(d['coordinate_max_abs']<=.1 and d['confidence_abs']<=.0001 for d in deltas)))
                            write(out/'baseline_previous_entry_comparison.json',dict(
                                scope='Unfused FP32 model with TF32 disabled versus historical fused GPU entry; not interchangeable numeric paths',
                                coordinate_tolerance=.1,probability_tolerance=.0001,frames=comparisons,
                                all_counts_equal=True,strict_all_passed=all(r['strict_passed'] for r in comparisons)))
                            baseline=measured
                        elif measured!=baseline:
                            raise RuntimeError('Arms did not start with identical predictions')
                    train_mode(core)
                    with torch.no_grad():
                        loss,parts = core.loss(batch)
                    if not torch.isfinite(loss).all():
                        raise RuntimeError('Nonfinite diagnostic evaluation loss')
                    records.append(dict(step=step,training_loss_components=[float(x) for x in parts],evaluations=measured))
                    write(arm/'measurements.json',records)
                    print(name,step,{k:(v['tp'],v['fp'],v['fn']) for k,v in measured.items()},flush=True)
                if step==80:
                    break
                train_mode(core)
                optimizer.zero_grad(set_to_none=True)
                loss,parts = core.loss(batch)
                if not torch.isfinite(loss).all():
                    raise RuntimeError('Nonfinite training loss')
                loss.sum().backward()
                gradients=[p.grad for p in core.parameters() if p.requires_grad]
                if not gradients or not all(g is not None and torch.isfinite(g).all() for g in gradients):
                    raise RuntimeError('Missing/nonfinite learned gradients')
                if any(p.grad is not None for p in core.parameters() if not p.requires_grad):
                    raise RuntimeError('Frozen parameter has gradient')
                norm = torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad],10,error_if_nonfinite=True)
                optimizer.step()
                cadence.on_batch_end(trainer)
                losses.append(dict(step=step+1,loss_components=[float(x) for x in parts.detach()],gradient_norm_before_clip=float(norm)))
            write(arm/'loss_trace.json',losses)
        finally:
            cadence.close()
        completed[name] = dict(actual_optimizer_steps=len(cadence.steps),final=records[-1],audits=audits,
            artifacts={p.name:sha256(p) for p in arm.glob('*.json')})
        write(out/'progress.json',completed)
        del core,wrapper,optimizer,batch
        torch.cuda.empty_cache()
    pages=[]
    for row in positive:
        for view,bank,gt_map,metric in [('full',images,None,'full_rect'),('crop',crop_images,crop_gt,'oracle_crop_square')]:
            tiles=[]
            for name,_,_ in ARMS:
                image=bank[row['image']].copy()
                gt=gt_map[row['image']] if gt_map else row['boxes_xyxy']
                frame=next(r for r in completed[name]['final']['evaluations'][metric]['frames'] if r['image']==row['image'])
                for box in gt:
                    a,b,c,d=map(round,box);cv2.rectangle(image,(a,b),(c,d),(0,255,0),2)
                for pred in frame['predictions']:
                    a,b,c,d=map(round,pred['xyxy']);cv2.rectangle(image,(a,b),(c,d),(255,0,0),2)
                h,w=image.shape[:2];scale=min(510/w,360/h)
                image=cv2.resize(image,(round(w*scale),round(h*scale)))
                ih,iw=image.shape[:2];tile=np.full((420,520,3),24,np.uint8)
                tile[50:50+ih,(520-iw)//2:(520-iw)//2+iw]=image
                cv2.putText(tile,name+' IoU %.3f'%frame['best_iou'][0],(8,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1)
                tiles.append(tile)
            path=out/(Path(row['image']).stem+'_'+view+'.jpg')
            if not cv2.imwrite(str(path),np.concatenate(tiles,axis=1)):
                raise RuntimeError('Cannot write review page')
            pages.append(dict(file=path.name,sha256=sha256(path),image=row['image'],view=view))
    if any(sha256(Path(__file__).with_name(n))!=h for n,h in sources.items()):
        raise RuntimeError('Diagnostic source changed during execution')
    result=dict(plan_sha256=sha256(out/'plan.json'),baseline_same_state_and_predictions_across_arms=True,
        baseline_previous_entry_comparison_sha256=sha256(out/'baseline_previous_entry_comparison.json'),
        completed_arms=completed,pages=pages,actual_visual_review=False,
        reserved_predicted_or_trained=False,trained_weights_saved=False)
    write(out/'complete.json',result)
    print('All four diagnostics completed; no candidate weights; visual inspection pending',flush=True)


if __name__=='__main__':
    main()
