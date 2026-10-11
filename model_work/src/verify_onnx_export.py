"""Export an isolated checkpoint copy and verify FP32 CPU pipeline parity.

Compares independent letterboxing against pinned Ultralytics, raw outputs, and
NumPy NMS against framework NMS. Identity-bound diagnostic images stay local.
No quantization, KModel, board validation, or alarm validation is implied.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import shutil

from data_integrity import sha256
from pc_onnx_infer import make_session, postprocess, preprocess

ROOT = Path(__file__).resolve().parents[2]


def check_detections(reference, actual, pixel_tolerance=.1, confidence_tolerance=.0001):
    """Maximum-cardinality pairing within all declared numeric tolerances."""
    if len(reference) != len(actual):
        return False
    edges = []
    for left in reference:
        edges.append([i for i, right in enumerate(actual)
                      if left[5] == right[5] and abs(float(left[4]) - float(right[4])) <= confidence_tolerance
                      and max(abs(float(a) - float(b)) for a, b in zip(left[:4], right[:4])) <= pixel_tolerance])
    owners = {}

    def augment(index, seen):
        for target in edges[index]:
            if target in seen:
                continue
            seen.add(target)
            if target not in owners or augment(owners[target], seen):
                owners[target] = index
                return True
        return False

    for index in range(len(reference)):
        augment(index, set())
    return len(owners) == len(reference)


def diagnostic_inputs(data):
    meta = json.loads((data / "build_manifest.json").read_text(encoding="utf-8"))
    inputs = []
    for row in meta["lineage"]:
        if row["origin"] != "legacy_train":
            inputs.append({"path": str(data / "images/train" / row["image"]), "sha256": row["sha256"], "group": row["origin"]})
    for row in json.loads((ROOT / "docs/blue_diagnostic_primary_boxes_20260930.json").read_text(encoding="utf-8")):
        inputs.append({"path": str(ROOT / "model_work/data/commons_blue_review_20260927" / (row["slug"] + ".jpg")),
                       "sha256": row["sha256"], "group": "blue_primary"})
    base = Path(meta["base"])
    rows = [r for r in csv.DictReader((base / "manifest.csv").open(encoding="utf-8-sig")) if r["split"] == "test"]
    rng = random.Random(20261002)
    for provenance in sorted({r["provenance"] for r in rows}):
        subset = sorted((r for r in rows if r["provenance"] == provenance), key=lambda r: r["file"])
        for row in rng.sample(subset, min(4, len(subset))):
            inputs.append({"path": str(base / "images/test" / row["file"]), "sha256": row["sha256"], "group": "legacy_" + provenance})
    for row in inputs:
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError("Diagnostic input identity mismatch")
    return inputs


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

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dynamic-rectangular", action="store_true")
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Refusing overwrite")
    if sha256(a.weights) != a.expected_sha256.lower():
        raise SystemExit("Checkpoint identity mismatch")
    inputs = diagnostic_inputs(a.data.resolve())
    a.out.mkdir(parents=True)
    dump = lambda name, value: (a.out / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    plan = {"weights_sha256": a.expected_sha256, "script_sha256": sha256(Path(__file__)),
            "runtime_script_sha256": sha256(Path(__file__).with_name("pc_onnx_infer.py")),
            "tolerance": {"raw_xywh_pixels": .1, "raw_probability": .0001,
                          "nms_xyxy_input_pixels": .1, "nms_probability": .0001},
            "config": {"imgsz": 640, "batch": 1, "dynamic": a.dynamic_rectangular, "half": False,
                       "opset": 12, "simplify": False, "embedded_nms": False, "conf": .25, "nms_iou": .6},
            "environment": {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                            "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__},
            "scope": "FP32 CPU export consistency on exposed diagnostics; no KModel, quantization, independent accuracy or board test"}
    dump("launch.json", plan)
    dump("local_inputs.json", inputs)
    try:
        checkpoint = a.out / "model.pt"
        shutil.copy2(a.weights, checkpoint)
        if sha256(checkpoint) != a.expected_sha256.lower():
            raise ValueError("Isolated copy identity mismatch")
        torch.set_num_threads(2)
        exported = Path(YOLO(str(checkpoint)).export(format="onnx", imgsz=640, batch=1, dynamic=a.dynamic_rectangular,
                                                 half=False, opset=12, simplify=False, nms=False, device="cpu"))
        onnx.checker.check_model(str(exported), full_check=True)
        session = make_session(exported, sha256(exported))
        reference = YOLO(str(checkpoint)).model.cpu().float().eval()
        reference.fuse(verbose=False)
        letterbox = LetterBox(new_shape=(640, 640), auto=a.dynamic_rectangular)
        details = []
        for index, row in enumerate(inputs):
            if sha256(Path(row["path"])) != row["sha256"]:
                raise ValueError("Input changed during verification")
            image = cv2.imdecode(np.fromfile(row["path"], dtype=np.uint8), cv2.IMREAD_COLOR)
            tensor, geometry = preprocess(image, a.dynamic_rectangular)
            framework_tensor = np.ascontiguousarray(letterbox(image=image)[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.
            preprocessing_equal = np.array_equal(tensor, framework_tensor)
            with torch.inference_mode():
                predicted = reference(torch.from_numpy(framework_tensor))
                raw = predicted[0] if isinstance(predicted, tuple) else predicted
            native = raw.cpu().numpy()
            actual = session.run(None, {session.get_inputs()[0].name: tensor})[0]
            if native.shape != actual.shape or not np.isfinite(actual).all():
                raise ValueError("Bad raw output contract")
            difference = np.abs(native - actual)
            coordinate_error = float(difference[:, :4].max())
            probability_error = float(difference[:, 4:].max())
            native_boxes = non_max_suppression(raw.clone(), conf_thres=.25, iou_thres=.6, nc=1, max_time_img=1.)[0].cpu().numpy()
            numpy_native_boxes = postprocess(native, input_hw=tuple(tensor.shape[2:]))
            onnx_boxes = postprocess(actual, input_hw=tuple(tensor.shape[2:]))
            native_nms_equal = check_detections(native_boxes, numpy_native_boxes)
            onnx_nms_equal = check_detections(native_boxes, onnx_boxes)
            passed = preprocessing_equal and coordinate_error <= .1 and probability_error <= .0001 and native_nms_equal and onnx_nms_equal
            details.append({**row, "size": [geometry["width"], geometry["height"]],
                            "input_shape": list(tensor.shape), "output_shape": list(actual.shape),
                            "preprocessing_equal": preprocessing_equal, "raw_coordinate_max_abs": coordinate_error,
                            "raw_probability_max_abs": probability_error, "native_boxes": len(native_boxes),
                            "onnx_boxes": len(onnx_boxes), "numpy_nms_reference_match": native_nms_equal,
                            "onnx_nms_reference_match": onnx_nms_equal, "passed": passed})
            print(f"Parity {index + 1}/{len(inputs)}: {row['group']} passed={passed}", flush=True)
            dump("local_parity_partial.json", details)
        if sha256(a.weights) != a.expected_sha256.lower() or sha256(checkpoint) != a.expected_sha256.lower():
            raise ValueError("Original or copied checkpoint changed")
        groups = {}
        for name in sorted({r["group"] for r in details}):
            subset = [r for r in details if r["group"] == name]
            groups[name] = {"images": len(subset), "passed": sum(r["passed"] for r in subset),
                            "native_boxes": sum(r["native_boxes"] for r in subset), "onnx_boxes": sum(r["onnx_boxes"] for r in subset)}
        summary = {**plan, "onnx_sha256": sha256(exported), "input_shape": session.get_inputs()[0].shape,
                   "output_shape": session.get_outputs()[0].shape,
                   "actual_input_shapes": sorted({tuple(r["input_shape"]) for r in details}),
                   "images": len(details), "passed": sum(r["passed"] for r in details), "all_passed": all(r["passed"] for r in details),
                   "raw_coordinate_max_abs": max(r["raw_coordinate_max_abs"] for r in details),
                   "raw_probability_max_abs": max(r["raw_probability_max_abs"] for r in details), "by_group": groups}
        dump("summary.json", summary)
        if not summary["all_passed"]:
            raise ValueError("Export consistency check failed; do not adopt")
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        dump("failed.json", {"error_type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
