"""Focused checks of the offline rule prototype, not alarm validation."""
import csv
import unittest
from pathlib import Path

from fusion_replay import Config, FusionEngine, Observation, outside_fraction


ROOT = Path(__file__).resolve().parents[1]
CONFIG = Config.read(ROOT / "data" / "fusion_demo_config_20260927.json")


def observation(t_s, **changes):
    values = dict(clip_id="test", t_s=t_s, fire_conf=0.8,
                  flame_outside_frac=0.05, flame_area_frac=0.02,
                  smoke_index=0.1, temp_delta_c=0.5, humidity_delta_pct=2.0,
                  certified_smoke_alarm=False, visual_ok=True, smoke_ok=True,
                  temp_ok=True, reset_requested=False)
    values.update(changes)
    return Observation(**values)


class FusionReplayTests(unittest.TestCase):
    def test_normal_burner_flame(self):
        result = FusionEngine(CONFIG).step(observation(0))
        self.assertEqual(result["state"], "NORMAL")
        self.assertIn("visible_flame_within_burner_roi", result["reason"])

    def test_missing_channels_are_unknown_without_risk(self):
        result = FusionEngine(CONFIG).step(observation(
            0, smoke_index=None, temp_delta_c=None, smoke_ok=False, temp_ok=False))
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertIn("missing:smoke,temperature", result["reason"])

    def test_missing_channels_do_not_hide_visible_risk(self):
        result = FusionEngine(CONFIG).step(observation(
            0, flame_outside_frac=0.8, smoke_index=None, temp_delta_c=None,
            smoke_ok=False, temp_ok=False))
        self.assertEqual(result["state"], "WATCH")
        self.assertIn("missing:smoke,temperature", result["reason"])

    def test_steam_hint_does_not_clear_smoke(self):
        result = FusionEngine(CONFIG).step(observation(
            0, smoke_index=0.4, humidity_delta_pct=15))
        self.assertEqual(result["state"], "WATCH")
        self.assertIn("steam_possible_not_a_safety_clear", result["reason"])

    def test_persistent_outside_flame_warning(self):
        engine = FusionEngine(CONFIG)
        states = [engine.step(observation(t, flame_outside_frac=0.8))["state"]
                  for t in (0, 1, 2)]
        self.assertEqual(states, ["WATCH", "WATCH", "WARNING"])

    def test_persistent_combined_signal_alarm_and_reset(self):
        engine = FusionEngine(CONFIG)
        states = [engine.step(observation(t, flame_outside_frac=0.8,
                                          smoke_index=0.8, temp_delta_c=9))["state"]
                  for t in range(5)]
        self.assertEqual(states[-1], "ALARM")
        self.assertEqual(engine.step(observation(5))["state"], "ALARM")
        self.assertEqual(engine.step(observation(6, reset_requested=True))["state"], "NORMAL")

    def test_independent_alarm_latches_and_active_reset_rejected(self):
        engine = FusionEngine(CONFIG)
        self.assertEqual(engine.step(observation(0, certified_smoke_alarm=True))["state"], "ALARM")
        rejected = engine.step(observation(1, certified_smoke_alarm=True,
                                           reset_requested=True))
        self.assertEqual(rejected["state"], "ALARM")
        self.assertIn("manual_reset_rejected", rejected["reason"])
        self.assertEqual(engine.step(observation(2, reset_requested=True))["state"], "NORMAL")

    def test_time_gap_breaks_persistence(self):
        engine = FusionEngine(CONFIG)
        states = [engine.step(observation(t, flame_outside_frac=0.8))["state"]
                  for t in (0, 1, 10, 11)]
        self.assertEqual(states, ["WATCH"] * 4)

    def test_non_increasing_time_rejected(self):
        engine = FusionEngine(CONFIG)
        engine.step(observation(1))
        with self.assertRaisesRegex(ValueError, "Non-increasing"):
            engine.step(observation(1))

    def test_bad_visual_geometry_rejected(self):
        with (ROOT / "data" / "fusion_demo_scenarios_20260927.csv").open(
                encoding="utf-8-sig", newline="") as stream:
            row = dict(next(csv.DictReader(stream)))
        row["flame_outside_frac"] = ""
        with self.assertRaisesRegex(ValueError, "flame geometry"):
            Observation.from_row(row)

    def test_box_outside_fraction(self):
        self.assertEqual(outside_fraction([0, 0, 10, 10], [0, 0, 10, 10]), 0)
        self.assertEqual(outside_fraction([0, 0, 10, 10], [10, 10, 20, 20]), 1)


if __name__ == "__main__":
    unittest.main()
