"""Unit tests for the saved v18/v24 equal-count ranking helpers."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_v18_v24_equal_counts_20261029 as analysis


class MatchingTests(unittest.TestCase):
    def test_exact_assignment_beats_greedy_on_multibox_case(self):
        boxes_a = [[0, 0, 1, 1], [2, 2, 3, 3]]
        boxes_b = [[10, 10, 11, 11], [12, 12, 13, 13]]
        # Greedy takes 0.90 first and is left with 0.10 (sum 1.00).
        # The optimal one-to-one permutation is 0.80 + 0.85 (sum 1.65).
        matrix = {
            (tuple(boxes_a[0]), tuple(boxes_b[0])): 0.90,
            (tuple(boxes_a[0]), tuple(boxes_b[1])): 0.80,
            (tuple(boxes_a[1]), tuple(boxes_b[0])): 0.85,
            (tuple(boxes_a[1]), tuple(boxes_b[1])): 0.10,
        }

        def fixture_iou(a, b):
            return matrix[(tuple(a), tuple(b))]

        with patch.object(analysis, "box_iou", side_effect=fixture_iou):
            selected = analysis.exact_assignment_ious(boxes_a, boxes_b)

        self.assertEqual(selected, (0.80, 0.85))
        self.assertAlmostEqual(sum(selected), 1.65)

    def test_exact_assignment_rejects_empty_or_unequal_inputs(self):
        with self.assertRaisesRegex(ValueError, "equal, nonzero"):
            analysis.exact_assignment_ious([], [])
        with self.assertRaisesRegex(ValueError, "equal, nonzero"):
            analysis.exact_assignment_ious([[0, 0, 1, 1]], [])


class CountClassificationTests(unittest.TestCase):
    def test_zero_equal_positive_and_unequal_counts(self):
        self.assertEqual(analysis.classify_box_counts(0, 0), "both_zero")
        self.assertEqual(analysis.classify_box_counts(2, 2), "equal_positive")
        self.assertEqual(analysis.classify_box_counts(1, 2), "count_mismatch")
        self.assertEqual(analysis.classify_box_counts(0, 1), "count_mismatch")


class CacheIdentityTests(unittest.TestCase):
    def synthetic_cache(self, temp_root: Path):
        plan_videos = []
        clips = []
        rows = []
        review_videos = []
        for video_index in range(1, 10):
            source = temp_root / f"video_{video_index}.bin"
            source.write_bytes(f"synthetic-video-{video_index}".encode())
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            plan_videos.append({"path": str(source), "sha256": source_sha})
            clips.append({
                "video_index": video_index,
                "source_sha256": source_sha,
                "step": 1,
                "decoded_frames": 3,
                "sampled_frames": 3,
                "model_counts": {},
            })
            review_frames = []
            for sample_index in range(3):
                pixel_sha = hashlib.sha256(f"{video_index}:{sample_index}".encode()).hexdigest()
                row = {
                    "video_index": video_index,
                    "video_sha256": source_sha,
                    "sample_index": sample_index,
                    "frame_index": sample_index,
                    "width": 720,
                    "height": 1280,
                    "source_pixel_sha256": pixel_sha,
                }
                rows.append(row)
                review_frames.append({
                    "video_index": video_index,
                    "video_sha256": source_sha,
                    "sample_index": sample_index,
                    "frame_index": sample_index,
                    "source_pixel_sha256": pixel_sha,
                })
            review_videos.append({"frames": review_frames})
        plan = {"videos": plan_videos}
        comparison = {"clips": clips, "total_samples": 27}
        review_manifest = {"videos": review_videos}
        return plan, comparison, rows, review_manifest

    def test_complete_synthetic_cache_identity_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.synthetic_cache(Path(directory))
            clips = analysis.validate_cache_identity(*fixture)
        self.assertEqual(len(clips), 9)

    def test_duplicate_sample_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = list(self.synthetic_cache(Path(directory)))
            fixture[2].append(dict(fixture[2][0]))
            with self.assertRaisesRegex(ValueError, "Duplicate per-frame identity"):
                analysis.validate_cache_identity(*fixture)

    def test_missing_sample_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = list(self.synthetic_cache(Path(directory)))
            fixture[2] = fixture[2][:-1]
            with self.assertRaisesRegex(ValueError, "Sampled frame sequence mismatch"):
                analysis.validate_cache_identity(*fixture)


if __name__ == "__main__":
    unittest.main()
