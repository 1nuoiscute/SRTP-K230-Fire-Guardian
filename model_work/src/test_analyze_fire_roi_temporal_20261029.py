"""Tests for saved-cache temporal rescue analysis; no model or video access."""
from __future__ import annotations

import contextlib
import io
import hashlib
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_fire_roi_temporal_20261029 as analysis


def _box():
    return {"confidence": 0.75, "xyxy": [10.0, 20.0, 30.0, 40.0]}


def _record(frame_index: int, empty_models: set[str], rescued_models: set[str]):
    predictions = {}
    for name in analysis.MODEL_NAMES:
        full_empty = name in empty_models
        rescued = name in rescued_models
        predictions[name] = {
            "full_frame": [] if full_empty else [_box()],
            "crop_attempted": full_empty,
            "crop_predictions": [_box()] if rescued else [],
        }
    return {
        "video_index": 7,
        "frame_index": frame_index,
        "time_seconds": frame_index / 28.02,
        "width": 720,
        "height": 1280,
        "source_pixel_sha256": f"{frame_index:064x}",
        "roi_xyxy": [72, 320, 648, 1152],
        "predictions": predictions,
    }


def _fixture_records():
    rows = []
    for frame in range(analysis.VIDEO7_FRAMES):
        empty = set()
        rescued = set()
        if 708 <= frame <= 719:
            empty.update(("v18_last", "v24_last"))
            rescued.update(("v18_last", "v24_last"))
        if 710 <= frame <= 713 or 715 <= frame <= 720:
            empty.add("v16_best")
            rescued.add("v16_best")
        if frame == 252:
            empty.add("v24_last")
        if frame == 560:
            empty.update(analysis.MODEL_NAMES)
            rescued.update(analysis.MODEL_NAMES)
        rows.append(_record(frame, empty, rescued))
    return rows


class FireRoiTemporalAnalysisTests(unittest.TestCase):
    def test_known_window_and_point_frames_are_recomputed(self):
        result = analysis.analyze_video7_records(_fixture_records())
        v18 = result["models"]["v18_last"]
        late_run = next(run for run in v18["empty_runs_before"] if run["start_frame"] <= 708 <= run["end_frame"])
        self.assertEqual((late_run["start_frame"], late_run["end_frame"]), (708, 719))
        self.assertEqual(v18["empty_runs_after"], [])
        self.assertEqual(
            result["window_708_719"]["models"]["v18_last"]["rescued_frame_indices"],
            list(range(708, 720)),
        )
        self.assertTrue(
            result["window_708_719"]["models"]["v24_last"]["all_frames_have_box_after"]
        )
        self.assertEqual(
            result["selected_frames_252_560"]["252"]["models"]["v24_last"]["post_rescue_empty"],
            True,
        )
        for name in analysis.MODEL_NAMES:
            state = result["selected_frames_252_560"]["560"]["models"][name]
            self.assertTrue(state["rescued"])
            self.assertFalse(state["post_rescue_empty"])
        v16 = result["models"]["v16_best"]
        self.assertEqual(len(result["window_708_719"]["models"]["v16_best"]["rescued_frame_indices"]), 9)
        self.assertEqual(v16["rescued_frames"], 11)
        self.assertEqual(v16["empty_after_frames"], 0)

    def test_missing_video7_frame_is_rejected(self):
        rows = _fixture_records()
        del rows[200]
        with self.assertRaisesRegex(ValueError, "continuous"):
            analysis.analyze_video7_records(rows)

    def test_missing_model_and_inconsistent_crop_state_are_rejected(self):
        rows = _fixture_records()
        del rows[100]["predictions"]["v24_last"]
        with self.assertRaisesRegex(ValueError, "exactly"):
            analysis.analyze_video7_records(rows)

        rows = _fixture_records()
        rows[708]["predictions"]["v18_last"]["crop_attempted"] = False
        with self.assertRaisesRegex(ValueError, "conflicts"):
            analysis.analyze_video7_records(rows)

    def test_nonfinite_box_data_is_rejected(self):
        rows = _fixture_records()
        rows[560]["predictions"]["v16_best"]["crop_predictions"][0]["confidence"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            analysis.analyze_video7_records(rows)

    def test_extreme_integer_is_a_finite_input_error(self):
        rows = _fixture_records()
        rows[0]["time_seconds"] = 10**400
        with self.assertRaisesRegex(ValueError, "line 1.time_seconds must be finite"):
            analysis.analyze_video7_records(rows)

    def test_cli_formats_extreme_integer_error_without_traceback(self):
        stderr = io.StringIO()
        with patch.object(
            analysis,
            "analyze_comparison_dir",
            side_effect=lambda _: analysis._finite_number(10**400, "time_seconds"),
        ):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    analysis.main(["--comparison-dir", "unused-cache"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("time_seconds must be finite", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_summary_hash_mismatch_is_rejected(self):
        plan_bytes = b'{"videos":[]}'
        records_bytes = b'{}\n'
        summary = {
            "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
        }
        actual_plan, actual_records = analysis.verify_completion_hashes(
            summary, plan_bytes, records_bytes
        )
        self.assertEqual(actual_plan, summary["plan_sha256"])
        self.assertEqual(actual_records, summary["records_sha256"])
        with self.assertRaisesRegex(ValueError, "plan.json SHA-256"):
            analysis.verify_completion_hashes(summary, b'{"videos":[1]}', records_bytes)
        with self.assertRaisesRegex(ValueError, "per_frame.jsonl SHA-256"):
            analysis.verify_completion_hashes(summary, plan_bytes, b'{"changed":true}\n')


if __name__ == "__main__":
    unittest.main()
