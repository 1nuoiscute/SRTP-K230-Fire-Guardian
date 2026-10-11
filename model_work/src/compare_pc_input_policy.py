"""Expose static-square vs default rectangular PyTorch inference differences.

Export parity alone does not preserve the default wrapper's input dimensions.
All images are already development-exposed. This is not an accuracy measure.
"""
import argparse
import json
from pathlib import Path

from data_integrity import sha256
from pc_onnx_infer import make_session, postprocess, preprocess, restore_boxes
from verify_onnx_export import check_detections


def main():
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parity-run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Refusing overwrite")
    meta = json.loads((a.parity_run / "summary.json").read_text(encoding="utf-8"))
    if meta["all_passed"] is not True:
        raise SystemExit("Prior raw parity must pass")
    weights = a.parity_run / "model.pt"
    onnx = a.parity_run / "model.onnx"
    if sha256(weights) != meta["weights_sha256"]:
        raise SystemExit("Checkpoint changed")
    session = make_session(onnx, meta["onnx_sha256"])
    inputs = json.loads((a.parity_run / "local_inputs.json").read_text(encoding="utf-8"))
    torch.set_num_threads(2)
    model = YOLO(str(weights))
    rectangular_mode = bool(meta["config"]["dynamic"])
    a.out.mkdir(parents=True)
    details = []
    for index, row in enumerate(inputs):
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError("Image changed")
        image = cv2.imdecode(np.fromfile(row["path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        tensor, geometry = preprocess(image, rectangular_mode)
        actual = restore_boxes(postprocess(session.run(None, {session.get_inputs()[0].name: tensor})[0], input_hw=tuple(tensor.shape[2:])), geometry)
        options = {"imgsz": 640, "conf": .25, "iou": .6, "device": "cpu", "verbose": False}
        square = model.predict(image, rect=False, **options)[0].boxes.data.cpu().numpy()
        rectangular = model.predict(image, rect=True, **options)[0].boxes.data.cpu().numpy()
        details.append({**row, "square_wrapper_match": check_detections(square, actual, pixel_tolerance=1.),
                        "default_rect_wrapper_match": check_detections(rectangular, actual, pixel_tolerance=1.),
                        "square_boxes": len(square), "default_rect_boxes": len(rectangular), "onnx_boxes": len(actual)})
        (a.out / "local_partial.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        print(index + 1, "square_match", details[-1]["square_wrapper_match"], "default_rect_match", details[-1]["default_rect_wrapper_match"], flush=True)
    groups = {}
    for group in sorted({r["group"] for r in details}):
        subset = [r for r in details if r["group"] == group]
        groups[group] = {"images": len(subset), "square_wrapper_matches": sum(r["square_wrapper_match"] for r in subset),
                         "default_rect_wrapper_matches": sum(r["default_rect_wrapper_match"] for r in subset),
                         "square_boxes": sum(r["square_boxes"] for r in subset),
                         "default_rect_boxes": sum(r["default_rect_boxes"] for r in subset)}
    summary = {"role": "inference input-policy compatibility, not accuracy", "images": len(details),
               "weights_sha256": meta["weights_sha256"], "onnx_sha256": meta["onnx_sha256"],
               "script_sha256": sha256(Path(__file__)), "original_coordinate_tolerance_pixels": 1., "confidence_tolerance": .0001,
               "square_wrapper_matches": sum(r["square_wrapper_match"] for r in details),
               "default_rect_wrapper_matches": sum(r["default_rect_wrapper_match"] for r in details), "by_group": groups}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    key = "default_rect_wrapper_matches" if rectangular_mode else "square_wrapper_matches"
    if summary[key] != len(details):
        raise ValueError("Declared input-policy end-to-end wrapper mismatch; inspect evidence")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
