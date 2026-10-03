"""Check a saved shared-feature composition against each learned source, on real CPU inputs."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from data_integrity import sha256
from fire_label_revision_dataset import verify, read
from independent_smoke_head import IndependentSmokeDetect
from pc_onnx_infer import preprocess


def state_equal(left, right):
    return left.keys() == right.keys() and all(torch.equal(v, right[k]) for k, v in left.items())


def verify_heads(fire, smoke_dual, combined, inputs, hardcase_data, expected_data, out, expected):
    if out.exists():
        raise ValueError('Existing composition audit; no overwrite')
    paths = dict(fire=fire, smoke_dual=smoke_dual, combined=combined, inputs=inputs)
    if any(sha256(p) != expected[k] for k, p in paths.items()):
        raise ValueError('Source, composition or input identity differs')
    dataset = verify(hardcase_data, expected_data)
    rows = read(inputs)
    rows += [dict(path=str(hardcase_data / 'images/train' / r['image']), sha256=r['sha256'], group=r['origin'])
             for r in read(hardcase_data / 'build_manifest.json')['lineage'] if r['origin'] != 'legacy_train']
    if not rows or any(sha256(Path(r['path'])) != r['sha256'] for r in rows):
        raise ValueError('Real composition input changed')
    torch.set_num_threads(2)
    source_fire = YOLO(str(fire)).model.cpu().float().eval()
    source_smoke = YOLO(str(smoke_dual)).model.cpu().float().eval()
    model = YOLO(str(combined)).model.cpu().float().eval()
    if source_fire.names != {0: 'fire'} or source_smoke.names != {0: 'fire', 1: 'smoke'} or model.names != source_smoke.names:
        raise ValueError('Wrong class mapping')
    if not isinstance(source_smoke.model[-1], IndependentSmokeDetect) or not isinstance(model.model[-1], IndependentSmokeDetect):
        raise ValueError('Independent learned boxes are required')
    blocks = len(source_fire.model) - 1
    if len(model.model) != blocks + 1 or len(source_smoke.model) != blocks + 1:
        raise ValueError('Feature architecture differs')
    for i in range(blocks):
        reference = source_fire.model[i].state_dict()
        if not state_equal(reference, model.model[i].state_dict()) or not state_equal(reference, source_smoke.model[i].state_dict()):
            raise ValueError('Saved shared features differ between learned sources')
    if not state_equal(source_fire.model[-1].state_dict(), model.model[-1].fire_head.state_dict()):
        raise ValueError('Saved learned fire head changed')
    if not state_equal(source_smoke.model[-1].smoke_head.state_dict(), model.model[-1].smoke_head.state_dict()):
        raise ValueError('Saved learned smoke head changed')
    out.mkdir(parents=True)
    (out / 'local_inputs.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    details = []
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            if sha256(Path(row['path'])) != row['sha256']:
                raise ValueError('Input changed during composition audit')
            im = cv2.imdecode(np.fromfile(row['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError('Image decode failed')
            tensor, _ = preprocess(im)
            tensor = torch.from_numpy(tensor)
            reference_fire, reference_dual, actual = source_fire(tensor)[0], source_smoke(tensor)[0], model(tensor)[0]
            anchors = reference_fire.shape[-1]
            if actual.shape != (1, 6, anchors * 2) or reference_dual.shape != actual.shape:
                raise ValueError('Composition output layout differs')
            raw_fire = actual[:, :5, :anchors]
            raw_smoke = actual[:, [0, 1, 2, 3, 5], anchors:]
            reference_smoke = reference_dual[:, [0, 1, 2, 3, 5], anchors:]
            if not torch.equal(raw_fire, reference_fire) or not torch.equal(raw_smoke, reference_smoke):
                raise ValueError('Learned source raw output changed')
            if not torch.all(actual[:, 5, :anchors] == 0) or not torch.all(actual[:, 4, anchors:] == 0):
                raise ValueError('Independent class isolation changed')
            f = non_max_suppression(reference_fire.clone(), conf_thres=.25, iou_thres=.6, nc=1, max_det=300)[0]
            s = non_max_suppression(reference_dual.clone(), conf_thres=.25, iou_thres=.6, nc=2, max_det=300)[0]
            c = non_max_suppression(actual.clone(), conf_thres=.25, iou_thres=.6, nc=2, max_det=300)[0]
            if not torch.equal(f, c[c[:, 5] == 0]) or not torch.equal(s[s[:, 5] == 1], c[c[:, 5] == 1]):
                raise ValueError('Composition changed final class-specific NMS detections')
            details.append(dict(index=index, image_sha256=row['sha256'], group=row['group'],
                                raw_fire_max_abs=float((raw_fire - reference_fire).abs().max()),
                                raw_smoke_max_abs=float((raw_smoke - reference_smoke).abs().max()),
                                fire_detections=len(f), smoke_detections=int((c[:, 5] == 1).sum()),
                                final_class_detections_exact=True))
            if index % 20 == 0:
                print(f'COMPOSITION {index}/{len(rows)}', flush=True)
    if any(sha256(p) != expected[k] for k, p in paths.items()):
        raise ValueError('Source identity changed during audit')
    detail_path = out / 'details.json'
    detail_path.write_text(json.dumps(details, indent=2) + '\n', encoding='utf-8')
    result = dict(role='Learned source composition consistency on exposed development inputs; no new training or generalization claim',
                  sources=expected, hardcase_dataset=dataset, script_sha256=sha256(Path(__file__)),
                  dependencies={n: sha256(Path(__file__).with_name(n)) for n in
                                ('independent_smoke_head.py', 'pc_onnx_infer.py', 'fire_label_revision_dataset.py', 'data_integrity.py')},
                  device='cpu', cpu_threads=2, imgsz=640, rect=False, conf=.25, nms_iou=.6, max_det=300,
                  saved_shared_features_equal=True, saved_fire_head_equal=True, saved_smoke_head_equal=True,
                  verified_images=len(details), unique_image_hashes=len({r['sha256'] for r in rows}),
                  raw_fire_max_abs=max(r['raw_fire_max_abs'] for r in details),
                  raw_smoke_max_abs=max(r['raw_smoke_max_abs'] for r in details),
                  final_class_detections_exact_images=sum(r['final_class_detections_exact'] for r in details),
                  local_inputs_sha256=sha256(out / 'local_inputs.json'), details_sha256=sha256(detail_path))
    (out / 'verification_complete.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('fire', 'smoke-dual', 'combined', 'inputs'):
        p.add_argument('--' + name, type=Path, required=True)
        p.add_argument('--expected-' + name + '-sha256', required=True)
    p.add_argument('--hardcase-data', type=Path, required=True)
    p.add_argument('--expected-data-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    expected = dict(fire=a.expected_fire_sha256, smoke_dual=a.expected_smoke_dual_sha256,
                    combined=a.expected_combined_sha256, inputs=a.expected_inputs_sha256)
    verify_heads(a.fire.resolve(), a.smoke_dual.resolve(), a.combined.resolve(), a.inputs.resolve(),
                 a.hardcase_data.resolve(), a.expected_data_sha256, a.out.resolve(), expected)
