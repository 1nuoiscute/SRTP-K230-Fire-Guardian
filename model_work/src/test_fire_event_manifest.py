import contextlib
import copy
import io
import json
import unittest
from unittest import mock

from fire_event_manifest import ManifestValidationError, main, validate_manifest


def valid_manifest():
    return {
        "schema_version": 1,
        "events": [{
            "event_id": "event-001", "source_group_id": "kitchen-a-stove-1-date-camera-op",
            "usage": "candidate_independent_test",
            "source": {"path": "capture/event-001.mp4", "sha256": "a" * 64, "relation": "screened"},
            "duration_s": 12.0, "image": {"width": 640, "height": 480},
            "roi": {"version": "roi-v1", "xyxy": [100, 80, 500, 420]},
            "intervals": [
                {"start_s": 0, "end_s": 3, "physical_state": "off", "visibility": "clear_no_flame", "negative_example": True, "physical_evidence": "operator-log#12", "visual_evidence": "blind-review#interval-0"},
                {"start_s": 3, "end_s": 8, "physical_state": "on", "visibility": "flame_visible"},
                {"start_s": 8, "end_s": 12, "physical_state": "off", "visibility": "occluded"},
            ],
            "samples": [
                {"frame_number": 0, "timestamp_s": 0.0, "physical_state": "off", "visibility": "clear_no_flame", "negative_example": True, "physical_evidence": "operator-log#12", "visual_evidence": "blind-review#frame-0", "boxes": []},
                {"frame_number": 120, "timestamp_s": 4.0, "physical_state": "on", "visibility": "flame_visible", "boxes": [{"xyxy": [150, 180, 260, 310], "roi_relation": "inside"}]},
            ],
        }],
    }


class FireEventManifestTests(unittest.TestCase):
    def assert_invalid(self, data, message):
        with self.assertRaisesRegex(ManifestValidationError, message):
            validate_manifest(data)

    def test_valid_manifest_has_no_warnings(self):
        self.assertEqual(validate_manifest(valid_manifest()), [])

    def test_same_sha_cannot_cross_source_group_or_usage(self):
        for group, usage in (("other-group", "candidate_independent_test"),
                             ("kitchen-a-stove-1-date-camera-op", "development")):
            with self.subTest(group=group, usage=usage):
                data = valid_manifest()
                second = copy.deepcopy(data["events"][0])
                second.update(event_id="event-002", source_group_id=group, usage=usage)
                data["events"].append(second)
                self.assert_invalid(data, "same original SHA-256")

    def test_repeated_sha_within_same_group_and_use_warns_possible_double_count(self):
        data = valid_manifest()
        second = copy.deepcopy(data["events"][0])
        second.update(event_id="event-002")
        data["events"].append(second)
        warnings = validate_manifest(data)
        self.assertTrue(any("possible duplicate event counting" in warning for warning in warnings))

    def test_empty_samples_warn(self):
        data = valid_manifest()
        data["events"][0]["samples"] = []
        warnings = validate_manifest(data)
        self.assertTrue(any("samples is empty" in warning for warning in warnings))

    def test_off_clear_without_explicit_negative_warns(self):
        data = valid_manifest()
        interval = data["events"][0]["intervals"][0]
        interval.pop("negative_example")
        interval.pop("physical_evidence")
        interval.pop("visual_evidence")
        sample = data["events"][0]["samples"][0]
        sample.pop("negative_example")
        sample.pop("physical_evidence")
        sample.pop("visual_evidence")
        warnings = validate_manifest(data)
        self.assertTrue(any("interval 0 is off/clear" in warning for warning in warnings))
        self.assertTrue(any("sampled frame 0 is off/clear" in warning for warning in warnings))

    def test_unknown_source_is_explicit_warning(self):
        data = valid_manifest()
        data["events"][0]["source"]["relation"] = "unknown"
        warnings = validate_manifest(data)
        self.assertTrue(any("independence is unestablished" in warning for warning in warnings))

    def test_uncovered_time_is_warned(self):
        data = valid_manifest()
        data["events"][0]["intervals"][1]["start_s"] = 4
        warnings = validate_manifest(data)
        self.assertTrue(any("uncovered interval [3, 4)" in warning for warning in warnings))

    def test_frame_boxes_must_be_in_sample_annotations_not_intervals(self):
        data = valid_manifest()
        data["events"][0]["intervals"][1]["boxes"] = [{"xyxy": [150, 180, 260, 310]}]
        self.assert_invalid(data, "timestamped samples")

    def test_sample_requires_explicit_frame_number_and_timestamp(self):
        for key in ("frame_number", "timestamp_s"):
            with self.subTest(key=key):
                data = valid_manifest()
                del data["events"][0]["samples"][1][key]
                self.assert_invalid(data, key)

    def test_unknown_state_warns_and_visibility_remains_independent(self):
        data = valid_manifest()
        data["events"][0]["samples"][1].update(physical_state="on", visibility="occluded", boxes=[])
        self.assertEqual(validate_manifest(data), [])
        data["events"][0]["samples"][1].update(physical_state="unknown", visibility="occluded")
        self.assertTrue(any("unknown physical state" in warning for warning in validate_manifest(data)))

    def test_known_sample_interval_conflicts_are_warned(self):
        data = valid_manifest()
        data["events"][0]["samples"][1].update(physical_state="off", visibility="clear_no_flame", boxes=[])
        warnings = validate_manifest(data)
        self.assertTrue(any("physical_state=off conflicts" in warning for warning in warnings))
        self.assertTrue(any("visibility=clear_no_flame conflicts" in warning for warning in warnings))

    def test_physical_off_with_visible_flame_is_warned_for_transition_review(self):
        data = valid_manifest()
        sample = data["events"][0]["samples"][1]
        sample.update(physical_state="off", visibility="flame_visible")
        warnings = validate_manifest(data)
        self.assertTrue(any("physical_state=off with visible flame" in warning for warning in warnings))

    def test_occluded_interval_can_contain_visible_flame_sample(self):
        data = valid_manifest()
        data["events"][0]["intervals"][1]["visibility"] = "occluded"
        warnings = validate_manifest(data)
        self.assertFalse(any("visibility=flame_visible conflicts" in warning for warning in warnings))

    def test_occluded_sample_can_have_locatable_partial_flame(self):
        data = valid_manifest()
        sample = data["events"][0]["samples"][1]
        sample.update(visibility="occluded", partially_visible_flame=True)
        self.assertFalse(any("visibility=occluded conflicts" in warning for warning in validate_manifest(data)))

    def test_occluded_sample_boxes_require_explicit_partial_flame_marker(self):
        data = valid_manifest()
        sample = data["events"][0]["samples"][1]
        sample.update(visibility="occluded")
        self.assert_invalid(data, "partially_visible_flame=true")

    def test_negative_requires_physical_off_and_clear_view(self):
        for physical, visibility in (("on", "clear_no_flame"), ("off", "occluded"), ("unknown", "clear_no_flame")):
            with self.subTest(physical=physical, visibility=visibility):
                data = valid_manifest()
                data["events"][0]["samples"][0].update(physical_state=physical, visibility=visibility)
                self.assert_invalid(data, "negative examples require physical_state=off")

    def test_negative_requires_both_physical_and_visual_evidence_references(self):
        for key in ("physical_evidence", "visual_evidence"):
            with self.subTest(scope="sample", key=key):
                data = valid_manifest()
                del data["events"][0]["samples"][0][key]
                self.assert_invalid(data, key)
            with self.subTest(scope="interval", key=key):
                data = valid_manifest()
                del data["events"][0]["intervals"][0][key]
                self.assert_invalid(data, key)

    def test_roi_geometry_bounds_and_positive_area_are_checked(self):
        for coords, relation, message in (([150, 180, 260, 310], "outside", "expected inside"),
                                          ([-1, 180, 20, 210], "outside", "original image bounds"),
                                          ([150, 180, 150, 210], "inside", "positive area")):
            with self.subTest(coords=coords):
                data = valid_manifest()
                data["events"][0]["samples"][1]["boxes"][0].update(xyxy=coords, roi_relation=relation)
                self.assert_invalid(data, message)

    def test_overlapping_intervals_rejected(self):
        data = valid_manifest()
        data["events"][0]["intervals"][1]["start_s"] = 2
        self.assert_invalid(data, "must not overlap")

    def test_source_group_usage_must_be_frozen(self):
        data = valid_manifest()
        second = copy.deepcopy(data["events"][0])
        second.update(event_id="event-002", source={**second["source"], "sha256": "b" * 64}, usage="development")
        data["events"].append(second)
        self.assert_invalid(data, "share frozen usage")

    def test_cli_accepts_json_from_stdin_and_states_limits(self):
        stdin = io.StringIO(json.dumps(valid_manifest()))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()), mock.patch("sys.stdin", stdin):
            code = main(["-"])
        report = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "structurally_valid")
        self.assertTrue(any("does not prove source independence" in item for item in report["limitations"]))
        self.assertTrue(any("authentic, sufficient, or complete" in item for item in report["limitations"]))
        self.assertIn("Sampled-frame labels do not establish truth for unsampled frames.", report["limitations"])


if __name__ == "__main__":
    unittest.main()




