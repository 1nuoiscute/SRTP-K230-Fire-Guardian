"""Regression tests for confidence/geometry coupling at the visual replay boundary."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from fusion_replay import Config, FusionEngine, Observation, replay, select_visual_evidence, visual_to_fusion

CONFIG_PATH = Path(__file__).resolve().parents[1] / "data/fusion_demo_config_20260927.json"


def box(confidence, xyxy):
    return {"confidence": confidence, "xyxy": xyxy}


class VisualConversionTests(unittest.TestCase):
    def test_weak_outside_box_cannot_borrow_inside_confidence(self):
        v = select_visual_evidence([box(.9, [10,10,20,20]), box(.26, [70,70,80,80])], [0,0,50,50], 100,100,.55)
        self.assertEqual((v["fire_conf"], v["flame_outside_frac"], v["visual_prediction_index"]), (.9,0,0))

    def test_eligible_outside_box_is_not_hidden_by_stronger_inside_box(self):
        v = select_visual_evidence([box(.9, [10,10,20,20]), box(.7, [70,70,90,90])], [0,0,50,50],100,100,.55)
        self.assertEqual((v["fire_conf"],v["flame_outside_frac"],v["flame_area_frac"]),(.7,1,.04))

    def test_landscape_dimensions_used_for_area(self):
        v = select_visual_evidence([box(.8,[50,25,150,75])], [0,0,200,100],200,100,.55)
        self.assertEqual(v["flame_area_frac"],.25)

    def test_below_threshold_still_keeps_one_consistent_box(self):
        v = select_visual_evidence([box(.4,[70,70,80,80])], [0,0,50,50],100,100,.55)
        self.assertEqual((v["fire_conf"],v["flame_outside_frac"],v["visual_qualified_boxes"]),(.4,1,0))

    def test_empty_detections_have_zero_geometry(self):
        v = select_visual_evidence([], [0,0,50,50],100,100,.55)
        self.assertEqual((v["fire_conf"],v["flame_outside_frac"],v["flame_area_frac"]),(0,0,0))

    def test_invalid_box_or_roi_rejected(self):
        for boxes, roi in [([box(.8,[-1,0,20,20])],[0,0,50,50]),
                           ([box(float("nan"),[0,0,20,20])],[0,0,50,50]),
                           ([],[0,0,101,50])]:
            with self.subTest(boxes=boxes,roi=roi), self.assertRaises(ValueError):
                select_visual_evidence(boxes,roi,100,100,.55)

    def test_conversion_replay_keeps_missing_channels_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            input_path, roi_path, out = folder/"in.csv",folder/"roi.json",folder/"fusion.csv"
            config_path = folder/"config.json"
            config_data=json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            config_data["fire_conf_min"] = .55
            config_path.write_text(json.dumps(config_data),encoding="utf-8")
            with input_path.open("w",newline="",encoding="utf-8") as stream:
                w=csv.DictWriter(stream,fieldnames=["video","time_seconds","width","height","predictions"])
                w.writeheader();w.writerow({"video":"demo","time_seconds":0,"width":100,"height":100,
                    "predictions":json.dumps([box(.9,[10,10,20,20]),box(.26,[70,70,80,80])])})
            roi_path.write_text(json.dumps({"demo":{"width":100,"height":100,"roi_xyxy":[0,0,50,50]}}),encoding="utf-8")
            visual_to_fusion(input_path,roi_path,out,config_path)
            with out.open(encoding="utf-8-sig",newline="") as stream:
                obs=Observation.from_row(next(csv.DictReader(stream)))
            result=FusionEngine(Config.read(config_path)).step(obs)
            self.assertEqual((result["outside_flame"],result["state"]),(0,"UNKNOWN"))
            self.assertFalse(obs.smoke_ok);self.assertIsNone(obs.smoke_index)
            meta=json.loads(out.with_suffix(".csv.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["missing_channels"],["smoke","temperature"])
            replay(out,config_path,folder/"replay")
            with self.assertRaisesRegex(ValueError,"Conversion/replay config differs"):
                replay(out,CONFIG_PATH,folder/"wrong_config")
            out.write_text(out.read_text(encoding="utf-8-sig")+"\n",encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError,"Converted input identity changed"):
                replay(out,config_path,folder/"changed_input")
            roi_path.write_text(json.dumps({"demo":{"width":200,"height":100,"roi_xyxy":[0,0,50,50]}}),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"dimensions differ"):
                visual_to_fusion(input_path,roi_path,folder/"bad.csv",config_path)


if __name__ == "__main__": unittest.main()
