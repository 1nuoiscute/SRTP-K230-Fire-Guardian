#!/usr/bin/env python3
"""Adapt the 4-class YOLOv8 ONNX output to vendor ob_det.elf's 80 classes.

The vendor executable expects [1, 84, N]: four box channels followed by the
80 COCO class channels.  This script preserves the original network and adds
only a deterministic output adapter.  The four project classes are placed in
COCO slots whose long label strings can be patched safely in a copy of the ELF:

    fire  -> class 3  (motorcycle)
    smoke -> class 9  (traffic light)
    stove -> class 10 (fire hydrant)
    pan   -> class 12 (parking meter)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def shape_of(value_info: onnx.ValueInfoProto) -> list[int]:
    return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]


def add_slice_initializers(model: onnx.ModelProto, stem: str, start: int, end: int) -> list[str]:
    specs = {
        f"{stem}_starts": np.asarray([start], dtype=np.int64),
        f"{stem}_ends": np.asarray([end], dtype=np.int64),
        f"{stem}_axes": np.asarray([1], dtype=np.int64),
        f"{stem}_steps": np.asarray([1], dtype=np.int64),
    }
    model.graph.initializer.extend(
        numpy_helper.from_array(value, name=name) for name, value in specs.items()
    )
    return list(specs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.input)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise SystemExit("Expected exactly one model input and one model output")

    input_shape = shape_of(model.graph.input[0])
    output_shape = shape_of(model.graph.output[0])
    if (len(input_shape) != 4 or input_shape[:2] != [1, 3]
            or input_shape[2] != input_shape[3]
            or input_shape[2] <= 0 or input_shape[2] % 32 != 0):
        raise SystemExit(f"Unsupported input shape: {input_shape}")
    side = input_shape[2]
    rows = sum((side // stride) ** 2 for stride in (8, 16, 32))
    if output_shape != [1, 8, rows]:
        raise SystemExit(
            f"Unexpected shapes: input={input_shape}, output={output_shape}; "
            f"expected [1, 8, {rows}] output"
        )

    source = model.graph.output[0].name
    pieces: list[str] = []

    # Original output channel slices: boxes, fire, smoke, stove, pan.
    for name, start, end in (
        ("boxes", 0, 4),
        ("fire", 4, 5),
        ("smoke", 5, 6),
        ("stove", 6, 7),
        ("pan", 7, 8),
    ):
        initializer_names = add_slice_initializers(model, name, start, end)
        output_name = f"vendor_{name}"
        model.graph.node.append(
            helper.make_node(
                "Slice",
                [source, *initializer_names],
                [output_name],
                name=f"VendorSlice_{name}",
            )
        )

    # 80 class channels: 0..2 zero, fire at 3, 4..8 zero, smoke at 9,
    # stove at 10, 11 zero, pan at 12, and 13..79 zero.
    for name, channels in (("zeros_0_2", 3), ("zeros_4_8", 5), ("zero_11", 1), ("zeros_13_79", 67)):
        tensor_name = f"vendor_{name}"
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.zeros((1, channels, rows), dtype=np.float32),
                name=tensor_name,
            )
        )

    pieces.extend(
        [
            "vendor_boxes",
            "vendor_zeros_0_2",
            "vendor_fire",
            "vendor_zeros_4_8",
            "vendor_smoke",
            "vendor_stove",
            "vendor_zero_11",
            "vendor_pan",
            "vendor_zeros_13_79",
        ]
    )
    output_name = "vendor_output0"
    model.graph.node.append(
        helper.make_node("Concat", pieces, [output_name], axis=1, name="Vendor80Concat")
    )
    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 84, rows])
    )

    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"saved {args.output}")
    print(f"input={input_shape}, output=[1, 84, {rows}]")


if __name__ == "__main__":
    main()
