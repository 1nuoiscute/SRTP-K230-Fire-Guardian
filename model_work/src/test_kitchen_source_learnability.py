"""Regression checks for GT-oracle context geometry used only in diagnostics."""
import unittest
from diagnose_kitchen_source_learnability_v1 import crop_bounds


class OracleCropGeometryTests(unittest.TestCase):
    def test_full_kitchen_small_target_retains_context_and_label(self):
        self.assertEqual(crop_bounds(1920,1280,[1018,912,1146,947]),[890,865,1274,994])

    def test_boundary_target_is_not_cut_by_oracle_context(self):
        self.assertEqual(crop_bounds(200,200,[0,0,10,10]),[0,0,69,69])

    def test_large_target_keeps_whole_image_when_context_exceeds_it(self):
        self.assertEqual(crop_bounds(1600,1200,[633,142,1405,924]),[0,0,1600,1200])


if __name__=='__main__':
    unittest.main()
