"""Person-guided zoom (src/detectors/hazard_zoom.py) without a model."""

import unittest

import numpy as np

from src.detectors.hazard_zoom import ZoomConfig, crop_boxes, merge, zoom_detect

W, H = 1280, 720
CFG = ZoomConfig(enabled=True)


def person(conf, box):
    return ("person", conf, box)


class CropTests(unittest.TestCase):
    def test_far_person_gets_a_crop_with_arm_margin(self):
        (x1, y1, x2, y2), = crop_boxes([person(0.8, (600, 300, 640, 420))], W, H, CFG)
        self.assertLess(x1, 600 - 0.5 * 40)
        self.assertGreater(x2, 640 + 0.5 * 40)
        self.assertGreaterEqual(x2 - x1, CFG.min_crop)

    def test_near_person_is_skipped(self):
        self.assertEqual(crop_boxes([person(0.9, (100, 0, 500, 700))], W, H, CFG), [])

    def test_low_conf_person_is_skipped(self):
        self.assertEqual(crop_boxes([person(0.1, (600, 300, 640, 420))], W, H, CFG), [])

    def test_crops_clipped_to_frame(self):
        (x1, y1, x2, y2), = crop_boxes([person(0.8, (0, 0, 30, 90))], W, H, CFG)
        self.assertEqual((x1, y1), (0, 0))
        self.assertLessEqual(x2, W)

    def test_max_crops_and_highest_conf_first(self):
        ps = [person(0.4 + 0.1 * k, (100 + 200 * k, 300, 140 + 200 * k, 420)) for k in range(5)]
        crops = crop_boxes(ps, W, H, ZoomConfig(enabled=True, max_crops=2))
        self.assertEqual(len(crops), 2)
        self.assertGreater(crops[0][0], 800)  # the 0.8 person at x=900 first

    def test_overlapping_people_share_one_crop(self):
        ps = [person(0.9, (600, 300, 640, 420)), person(0.8, (602, 302, 642, 422))]
        self.assertEqual(len(crop_boxes(ps, W, H, CFG)), 1)

    def test_unknown_option_rejected(self):
        with self.assertRaises(ValueError):
            ZoomConfig.from_dict({"enable": True})


class MergeTests(unittest.TestCase):
    def test_new_object_added(self):
        out = merge([("knife", 0.3, (0, 0, 10, 10))], [("knife", 0.6, (50, 50, 60, 60))], 0.5)
        self.assertEqual(len(out), 2)

    def test_duplicate_keeps_higher_conf(self):
        out = merge([("knife", 0.3, (0, 0, 10, 10))], [("knife", 0.6, (0, 0, 10, 11))], 0.5)
        self.assertEqual(out, [("knife", 0.6, (0, 0, 10, 11))])

    def test_duplicate_lower_conf_dropped(self):
        out = merge([("knife", 0.9, (0, 0, 10, 10))], [("knife", 0.6, (0, 0, 10, 11))], 0.5)
        self.assertEqual(out, [("knife", 0.9, (0, 0, 10, 10))])

    def test_different_label_not_merged(self):
        out = merge([("knife", 0.9, (0, 0, 10, 10))], [("scissors", 0.6, (0, 0, 10, 10))], 0.5)
        self.assertEqual(len(out), 2)


class ZoomDetectTests(unittest.TestCase):
    def test_crop_coordinates_mapped_back(self):
        img = np.zeros((H, W, 3), np.uint8)
        full = [person(0.8, (600, 300, 640, 420))]
        seen = []

        def predict(crops):
            seen.extend(c.shape for c in crops)
            return [[("scissors", 0.7, (10.0, 20.0, 30.0, 40.0))]]

        merged, n = zoom_detect(img, full, "person", CFG, predict)
        self.assertEqual(n, 1)
        sc = [d for d in merged if d[0] == "scissors"][0]
        (cx1, cy1, _, _), = crop_boxes(full, W, H, CFG)
        self.assertEqual(sc[2], (10.0 + cx1, 20.0 + cy1, 30.0 + cx1, 40.0 + cy1))
        self.assertEqual(len(seen), 1)

    def test_no_people_no_model_call(self):
        img = np.zeros((H, W, 3), np.uint8)

        def predict(crops):
            raise AssertionError("must not be called")

        merged, n = zoom_detect(img, [("knife", 0.5, (0, 0, 10, 10))], "person", CFG, predict)
        self.assertEqual(n, 0)
        self.assertEqual(len(merged), 1)


if __name__ == "__main__":
    unittest.main()
