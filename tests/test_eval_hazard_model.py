"""Metric code of tools/eval_hazard_model.py (no model, no images)."""

import unittest

from tools.eval_hazard_model import (KNIFE, PERSON, SCISSORS, PerImage, average_precision,
                                     best_f2_threshold, bootstrap, class_ap, image_rates,
                                     match_image, paired_bootstrap, pr_at)

BOX = (10, 10, 50, 50)
FAR = (200, 200, 240, 240)


class MatchingTests(unittest.TestCase):
    def test_one_gt_matched_once(self):
        self.assertEqual(match_image([BOX], [(0.9, BOX), (0.8, BOX)]), [(0.9, True), (0.8, False)])

    def test_miss_by_location(self):
        self.assertEqual(match_image([BOX], [(0.9, FAR)]), [(0.9, False)])

    def test_perfect_ap(self):
        self.assertAlmostEqual(average_precision([(0.9, True), (0.8, True)], 2), 1.0)

    def test_half_recall_ap(self):
        self.assertAlmostEqual(average_precision([(0.9, True)], 2), 51 / 101)

    def test_fp_ranked_first_lowers_ap(self):
        self.assertLess(average_precision([(0.9, False), (0.8, True)], 1), 1.0)

    def test_no_gt_is_nan(self):
        self.assertNotEqual(average_precision([], 0), average_precision([], 0))


def img(gt_knife=(), pred=()):
    return PerImage({KNIFE: list(gt_knife)}, {KNIFE: list(pred)})


class ImageLevelTests(unittest.TestCase):
    def test_rates(self):
        ims = [img([BOX], [(0.6, BOX)]), img([BOX], [(0.1, BOX)]), img([], [(0.5, FAR)]), img([], [])]
        rec, far = image_rates(ims, 0.25)
        self.assertAlmostEqual(rec, 0.5)
        self.assertAlmostEqual(far, 0.5)

    def test_wrong_label_still_counts_as_hazard_alert(self):
        im = PerImage({KNIFE: [BOX]}, {SCISSORS: [(0.7, BOX)]})
        self.assertEqual(image_rates([im], 0.5)[0], 1.0)

    def test_pr_and_threshold(self):
        ims = [img([BOX], [(0.9, BOX)]), img([], [(0.3, FAR)])]
        self.assertEqual(pr_at(ims, KNIFE, 0.5), (1.0, 1.0))
        self.assertGreaterEqual(best_f2_threshold(ims, KNIFE), 0.35)

    def test_person_class_independent(self):
        im = PerImage({PERSON: [BOX], KNIFE: []}, {PERSON: [(0.9, BOX)]})
        self.assertAlmostEqual(class_ap([im], PERSON), 1.0)
        self.assertFalse(im.has_hazard)


class BootstrapTests(unittest.TestCase):
    def test_interval_contains_point(self):
        ims = [img([BOX], [(0.9, BOX)])] * 5 + [img([BOX], [])] * 5
        p, lo, hi = bootstrap(lambda s: image_rates(s, 0.5)[0], ims, reps=200)
        self.assertAlmostEqual(p, 0.5)
        self.assertLessEqual(lo, p)
        self.assertGreaterEqual(hi, p)

    def test_paired_identical_models_zero(self):
        ims = [img([BOX], [(0.9, BOX)]), img([BOX], [])]
        self.assertEqual(paired_bootstrap(lambda s: image_rates(s, 0.5)[0], ims, ims, reps=50), (0.0, 0.0, 0.0))


class SizeRecallTests(unittest.TestCase):
    def test_small_and_large_buckets(self):
        from tools.eval_hazard_model import recall_by_size
        small, large = (10, 10, 20, 20), (0, 0, 500, 500)   # in a 1000x1000 image: 1% and 50%
        im = PerImage({KNIFE: [small], SCISSORS: [large]}, {KNIFE: [(0.9, large)]}, (1000, 1000))
        self.assertEqual(recall_by_size([im], 0.25, 0.0, 0.03), 0.0)
        self.assertEqual(recall_by_size([im], 0.25, 0.08, 9.0), 1.0)  # label-agnostic match
        self.assertNotEqual(recall_by_size([im], 0.25, 0.03, 0.08), recall_by_size([im], 0.25, 0.03, 0.08))


if __name__ == "__main__":
    unittest.main()
