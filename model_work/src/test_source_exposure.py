import unittest
from source_exposure import commons_source_key, training_source_keys, source_exposure_summary


class SourceExposureTests(unittest.TestCase):
    def test_resized_bytes_do_not_create_a_new_work(self):
        training={"lineage":[{"source_page":"https://commons.wikimedia.org/wiki/File:Gas_cooker_blue_flame.jpg", "sha256":"thumbnail"}]}
        old_page="https://commons.wikimedia.org/wiki/File%3AGas%20cooker%20blue%20flame.jpg"
        self.assertIn(commons_source_key(old_page),training_source_keys(training))

    def test_different_works_stay_distinct(self):
        a="https://commons.wikimedia.org/wiki/File:Gas_stove_blue_flames.jpg"
        b="https://commons.wikimedia.org/wiki/File:Gas_stove_flame.jpg"
        self.assertNotEqual(commons_source_key(a),commons_source_key(b))

    def test_nonfile_or_noncommons_url_rejected(self):
        for url in ["https://commons.wikimedia.org/wiki/Category:Gas_flames", "https://example.com/wiki/File:test.jpg"]:
            with self.assertRaises(ValueError): commons_source_key(url)

    def test_strata_do_not_mix_training_sources_with_untrained_sources(self):
        common={"intended":"blue_flame","gt_boxes":1,"det_025":1,"localized_025":1,"det_050":1,"localized_050":1}
        rows=[{**common,"source_key":"File:train.jpg","source_exposure":"training_source_in_comparison"},
              {**common,"source_key":"File:other.jpg","source_exposure":"untrained_source_development_diagnostic","localized_025":0}]
        groups=source_exposure_summary(rows)
        self.assertEqual(groups["training_source_in_comparison"]["summary"]["0.25"]["blue_flame"]["localized"],1)
        self.assertEqual(groups["untrained_source_development_diagnostic"]["summary"]["0.25"]["blue_flame"]["localized"],0)
        self.assertEqual(sum(v["images"] for v in groups.values()),2)


if __name__ == "__main__": unittest.main()
