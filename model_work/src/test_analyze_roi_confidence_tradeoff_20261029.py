"""Small standard-library tests for confidence filtering and cache states."""
from __future__ import annotations

import sys
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_roi_confidence_tradeoff_20261029 as analysis


def box(score: float) -> dict:
    return {"confidence": score, "xyxy": [1, 2, 3, 4]}


class ConfidenceTradeoffTests(unittest.TestCase):
    def test_threshold_keeps_equality_and_filters_below(self):
        boxes = [box(0.249999), box(0.25), box(0.30), box(0.50)]
        self.assertEqual([b["confidence"] for b in analysis.filter_saved_boxes(boxes, 0.25)], [0.25, 0.30, 0.50])
        self.assertEqual([b["confidence"] for b in analysis.filter_saved_boxes(boxes, 0.30)], [0.30, 0.50])
        self.assertEqual([b["confidence"] for b in analysis.filter_saved_boxes(boxes, 0.50)], [0.50])

    def test_empty_crop_list_is_valid_attempt_with_no_saved_boxes(self):
        pred = {"full_frame": [], "crop_attempted": True, "crop_predictions": []}
        self.assertEqual(analysis.validate_conditional(pred, "fixture"), "crop_attempted_no_boxes")
        self.assertEqual(analysis.filter_saved_boxes(pred["crop_predictions"], 0.25), [])

    def test_full_frame_box_means_crop_not_triggered(self):
        pred = {"full_frame": [box(0.8)], "crop_attempted": False, "crop_predictions": []}
        self.assertEqual(analysis.validate_conditional(pred, "fixture"), "full_frame_existing_crop_not_triggered")

    def test_nonempty_crop_predictions_means_saved_roi_box(self):
        pred = {"full_frame": [], "crop_attempted": True, "crop_predictions": [box(0.4)]}
        self.assertEqual(analysis.validate_conditional(pred, "fixture"), "crop_attempted_with_boxes")

    def test_inconsistent_trigger_state_is_rejected(self):
        pred = {"full_frame": [], "crop_attempted": False, "crop_predictions": []}
        with self.assertRaisesRegex(ValueError, "conflicts"):
            analysis.validate_conditional(pred, "fixture")

    def test_completion_plan_hash_chain_mismatch_is_rejected(self):
        plan_bytes = b'{"fixture":"plan"}'
        rows_bytes = b'{"fixture":"rows"}\n'
        done_bytes = json.dumps({
            "plan_sha256": "0" * 64,
            "per_image_sha256": hashlib.sha256(rows_bytes).hexdigest(),
        }).encode("utf-8")
        fixture_bytes = iter((plan_bytes, done_bytes, rows_bytes))
        with patch.multiple(
            analysis,
            IMAGE_PLAN_SHA=hashlib.sha256(plan_bytes).hexdigest(),
            IMAGE_DONE_SHA=hashlib.sha256(done_bytes).hexdigest(),
            IMAGE_ROWS_SHA=hashlib.sha256(rows_bytes).hexdigest(),
        ), patch.object(Path, "read_bytes", side_effect=lambda: next(fixture_bytes)):
            with self.assertRaisesRegex(ValueError, "does not bind plan.json"):
                analysis.verify_bound_cache(Path("memory-fixture"), "image")

    def test_selected_frame_pixel_identity_mismatch_is_rejected(self):
        rows = [
            {"video_index": 7, "frame_index": frame,
             "source_pixel_sha256": ("0" * 64 if frame == 560 else analysis.VIDEO_PIXEL_SHA[frame])}
            for frame in analysis.VIDEO_FRAMES
        ]
        with self.assertRaisesRegex(ValueError, "frame 560 pixel identity mismatch"):
            analysis.selected_records(rows, [])


if __name__ == "__main__":
    unittest.main()
