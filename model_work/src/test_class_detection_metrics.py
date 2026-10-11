import unittest
from class_detection_metrics import scores_by_known_class


class ClassMetricsTests(unittest.TestCase):
    def test_cross_class_box_cannot_match_target(self):
        result=scores_by_known_class({0:[[0,0,10,10]],1:[]},[{'class':1,'xyxy':[0,0,10,10]}])
        self.assertEqual((result[0]['tp'],result[0]['fn'],result[1]['fp']),(0,1,1))

    def test_unknown_smoke_not_scored_as_fire_false_positive(self):
        result=scores_by_known_class({0:[]},[{'class':1,'xyxy':[0,0,10,10]}])
        self.assertEqual(result[0]['fp'],0)
        self.assertNotIn(1,result)
        self.assertIsNone(result[0]['f1'])


if __name__=='__main__':unittest.main()
