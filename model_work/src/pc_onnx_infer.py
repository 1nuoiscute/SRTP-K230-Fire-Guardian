"""CPU fire-only YOLO11 ONNX inference without PyTorch/Ultralytics.

Contract: FP32 [1,3,H,W] RGB /255 -> [1,5,sum(stride-grid sizes)]
Static 640x640 or explicitly selected dynamic stride-32 rectangular padding.
xywh pixel coordinates + one fire probability, no objectness, no embedded NMS.
This is visual evidence only; a detection is not an abnormal-fire alarm.
"""
import argparse
import ast
import json
from pathlib import Path
import time

from data_integrity import sha256


def preprocess(image, rectangular=False):
    import cv2
    import numpy as np
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR image")
    height, width = image.shape[:2]
    if min(height, width) <= 0:
        raise ValueError("Empty image")
    gain = min(640 / height, 640 / width)
    new_width, new_height = round(width * gain), round(height * gain)
    padding_width, padding_height = 640 - new_width, 640 - new_height
    if rectangular:
        padding_width, padding_height = padding_width % 32, padding_height % 32
    dw, dh = padding_width / 2, padding_height / 2
    left, right = round(dw - .1), round(dw + .1)
    top, bottom = round(dh - .1), round(dh + .1)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR) if (width, height) != (new_width, new_height) else image
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.
    return tensor, {"width": width, "height": height, "gain": gain, "left": left, "top": top,
                    "input_height": tensor.shape[2], "input_width": tensor.shape[3],
                    "input_policy": "rectangular_stride32" if rectangular else "square640"}


def postprocess(raw, conf=.25, nms_iou=.6, max_det=300, input_hw=(640, 640)):
    import numpy as np
    if not 0 < conf < 1 or not 0 < nms_iou < 1 or not isinstance(max_det, int) or not 1 <= max_det <= 300:
        raise ValueError("Invalid postprocess configuration")
    if len(input_hw) != 2 or any(not isinstance(d, int) or d <= 0 or d > 640 or d % 32 for d in input_hw):
        raise ValueError("Input dimensions must be positive stride-32 multiples up to 640")
    height, width = input_hw
    anchors = sum((height // s) * (width // s) for s in (8, 16, 32))
    if raw.shape != (1, 5, anchors) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError("Expected finite FP32 fire-only output matching input stride grids")
    if (raw[:, 4, :] < -1e-6).any() or (raw[:, 4, :] > 1 + 1e-6).any():
        raise ValueError("Fire channel must already contain probabilities")
    candidates = raw[0].T[raw[0, 4] > conf]
    if not len(candidates):
        return np.empty((0, 6), dtype=np.float32)
    xywh, scores = candidates[:, :4], candidates[:, 4]
    if (xywh[:, 2:] < 0).any():
        raise ValueError("Negative box size")
    xyxy = np.column_stack((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2))
    order = np.argsort(-scores, kind="stable")[:30000]
    keep = []
    while len(order) and len(keep) < max_det:
        index = order[0]
        keep.append(index)
        rest = order[1:]
        intersection = np.maximum(0, np.minimum(xyxy[index, 2:], xyxy[rest, 2:]) - np.maximum(xyxy[index, :2], xyxy[rest, :2])).prod(axis=1)
        area = np.maximum(0, xyxy[index, 2:] - xyxy[index, :2]).prod()
        others = np.maximum(0, xyxy[rest, 2:] - xyxy[rest, :2]).prod(axis=1)
        union = area + others - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[overlap <= nms_iou]
    return np.column_stack((xyxy[keep], scores[keep], np.zeros(len(keep), dtype=np.float32))).astype(np.float32)


def restore_boxes(boxes, geometry):
    result = boxes.copy()
    result[:, [0, 2]] = ((result[:, [0, 2]] - geometry["left"]) / geometry["gain"]).clip(0, geometry["width"])
    result[:, [1, 3]] = ((result[:, [1, 3]] - geometry["top"]) / geometry["gain"]).clip(0, geometry["height"])
    return result


def make_session(model, digest):
    import onnxruntime as ort
    if sha256(model) != digest.lower():
        raise ValueError("ONNX identity mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].type != "tensor(float)" or len(inputs[0].shape) != 4:
        raise ValueError("Unsupported ONNX input contract")
    dynamic = all(isinstance(inputs[0].shape[i], str) for i in (0, 2, 3)) and inputs[0].shape[1] == 3
    static = inputs[0].shape == [1, 3, 640, 640]
    if not static and not dynamic:
        raise ValueError("Expected static 640 or dynamic rectangular ONNX input")
    if len(outputs) != 1 or outputs[0].type != "tensor(float)" or len(outputs[0].shape) != 3:
        raise ValueError("Unsupported ONNX output contract")
    output_shape = outputs[0].shape
    # Dynamic ONNX shape inference may leave the channel dimension symbolic.
    # Class metadata is checked below; every actual output must still have 5
    # channels and the exact stride-grid count in postprocess().
    if (static and output_shape != [1, 5, 8400]) or (dynamic and not (isinstance(output_shape[0], str) and (output_shape[1] == 5 or isinstance(output_shape[1], str)) and isinstance(output_shape[2], str))):
        raise ValueError("Unsupported ONNX fire-only output dimensions")
    names = ast.literal_eval(session.get_modelmeta().custom_metadata_map.get("names", "{}"))
    if names != {0: "fire"}:
        raise ValueError("Expected exactly one fire class")
    return session


def validate_input_policy(session, rectangular):
    if rectangular and session.get_inputs()[0].shape == [1, 3, 640, 640]:
        raise ValueError("Rectangular inference requires the verified dynamic ONNX export")


def main():
    import cv2
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--conf", type=float, default=.25)
    p.add_argument("--nms-iou", type=float, default=.6)
    p.add_argument("--rectangular", action="store_true", help="Use default PyTorch stride-32 rectangular padding; requires dynamic export")
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Refusing overwrite")
    session = make_session(a.model, a.expected_sha256)
    validate_input_policy(session, a.rectangular)
    image = cv2.imdecode(np.fromfile(a.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    tensor, geometry = preprocess(image, a.rectangular)
    begin = time.perf_counter()
    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    elapsed = (time.perf_counter() - begin) * 1000
    boxes = restore_boxes(postprocess(raw, a.conf, a.nms_iou, input_hw=tuple(tensor.shape[2:])), geometry)
    result = {"role": "PC visual detections; not an abnormal-fire alarm or board validation",
              "model_sha256": sha256(a.model), "image_sha256": sha256(a.image),
              "configuration": {"conf": a.conf, "nms_iou": a.nms_iou, "provider": "CPUExecutionProvider"},
              "geometry": geometry, "onnx_forward_ms_single_call": elapsed,
              "predictions": [{"class": "fire", "confidence": float(b[4]), "xyxy": b[:4].tolist()} for b in boxes]}
    a.out.mkdir(parents=True)
    for b in boxes:
        corners = [round(float(v)) for v in b[:4]]
        cv2.rectangle(image, tuple(corners[:2]), tuple(corners[2:]), (0, 128, 255), 2)
        cv2.putText(image, f"fire {b[4]:.3f}", (corners[0], max(20, corners[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 128, 255), 2)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Cannot encode overlay")
    encoded.tofile(a.out / "overlay.jpg")
    (a.out / "predictions.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
