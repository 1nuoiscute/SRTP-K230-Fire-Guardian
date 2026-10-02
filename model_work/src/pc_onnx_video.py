"""Offline file-video inference using the validated static fire-only ONNX.

Writes original-coordinate boxes and media timestamps, compatible with the
existing visual-to-fusion CSV importer. No sensor values or alarm state invented.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import time

from data_integrity import sha256
from pc_onnx_infer import make_session, postprocess, preprocess, restore_boxes, validate_input_policy


def infer_video(session, model_digest, video, out, sample_fps=1., conf=.25, nms_iou=.6, boxed=False, rectangular=False):
    import cv2
    if out.exists():
        raise ValueError("Refusing overwrite")
    if not math.isfinite(sample_fps) or not 0 < sample_fps <= 30:
        raise ValueError("Invalid requested sample FPS")
    validate_input_policy(session, rectangular)
    source_digest = sha256(video)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nominal_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Cannot open video with valid nominal FPS")
    step = max(1, round(fps / sample_fps))
    out.mkdir(parents=True)
    writer = None
    count = sampled = with_boxes = box_count = 0
    forward_ms = []
    size = None
    fields = ["video", "frame_index", "time_seconds", "width", "height", "boxes", "predictions", "visual_ok", "model_sha256", "source_sha256"]
    summary = {"role": "unlabelled offline development predictions; sampled any-box counts are not accuracy or alarms",
               "source_sha256": source_digest, "model_sha256": model_digest,
               "script_sha256": sha256(Path(__file__)),
               "runtime_sha256": sha256(Path(__file__).with_name("pc_onnx_infer.py")),
               "configuration": {"requested_sample_fps": sample_fps, "step": step, "actual_sample_fps": fps / step,
                                 "conf": conf, "nms_iou": nms_iou,
                                 "input_policy": "rectangular_stride32" if rectangular else "square640"},
               "nominal_fps": fps, "nominal_frames": nominal_frames,
               "timestamp_basis": "frame_index/nominal_fps; no wall-clock or VFR timing validation"}
    dump = lambda name, value: (out / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    dump("launch.json", summary)
    try:
        with (out / "per_frame.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=fields)
            csv_writer.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                if size is None:
                    size = [w, h]
                    if boxed:
                        writer = cv2.VideoWriter(str(out / "boxed.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (w, h))
                        if not writer.isOpened():
                            raise ValueError("Cannot create sampled preview video")
                if size != [w, h]:
                    raise ValueError("Video dimensions changed; no partial success")
                if count % step == 0:
                    tensor, geometry = preprocess(frame, rectangular)
                    summary["actual_input_shape"] = list(tensor.shape)
                    begin = time.perf_counter()
                    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
                    forward_ms.append((time.perf_counter() - begin) * 1000)
                    detections = restore_boxes(postprocess(raw, conf, nms_iou, input_hw=tuple(tensor.shape[2:])), geometry)
                    boxes = [{"confidence": float(b[4]), "xyxy": b[:4].tolist()} for b in detections]
                    csv_writer.writerow({"video": video.name, "frame_index": count, "time_seconds": count / fps,
                                         "width": w, "height": h, "boxes": len(boxes), "predictions": json.dumps(boxes),
                                         "visual_ok": 1, "model_sha256": model_digest, "source_sha256": source_digest})
                    sampled += 1
                    with_boxes += bool(boxes)
                    box_count += len(boxes)
                    if writer is not None:
                        for b in boxes:
                            c = [round(v) for v in b["xyxy"]]
                            cv2.rectangle(frame, tuple(c[:2]), tuple(c[2:]), (0, 128, 255), 2)
                            cv2.putText(frame, f"fire {b['confidence']:.3f}", (c[0], max(20, c[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 128, 255), 2)
                        writer.write(frame)
                count += 1
        if not count or not sampled:
            raise ValueError("No readable frames")
        if sha256(video) != source_digest:
            raise ValueError("Video changed during inference")
        summary.update({"decoded_to_read_end": True, "decoded_frames": count,
                        "nominal_minus_decoded_frames": nominal_frames - count,
                        "sampled_frames": sampled, "frames_with_any_box": with_boxes, "total_boxes": box_count,
                        "size": size, "nominal_decoded_duration_s": count / fps,
                        "mean_onnx_forward_ms": sum(forward_ms) / len(forward_ms),
                        "csv_sha256": sha256(out / "per_frame.csv"),
                        "limitations": "Decoder read-end may also reflect truncation; count discrepancy retained. Sampling may miss short events. No VFR, camera, event or real-time verification."})
        dump("summary.json", summary)
        return summary
    except Exception as error:
        dump("failed.json", {"error_type": type(error).__name__, "message": str(error), "decoded_frames": count, "sampled_frames": sampled})
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sample-fps", type=float, default=1.)
    p.add_argument("--conf", type=float, default=.25)
    p.add_argument("--nms-iou", type=float, default=.6)
    p.add_argument("--boxed", action="store_true")
    p.add_argument("--rectangular", action="store_true")
    a = p.parse_args()
    session = make_session(a.model, a.expected_sha256)
    result = infer_video(session, a.expected_sha256.lower(), a.video, a.out, a.sample_fps, a.conf, a.nms_iou, a.boxed, a.rectangular)
    if sha256(a.model) != a.expected_sha256.lower():
        raise ValueError("Model changed during video inference")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
