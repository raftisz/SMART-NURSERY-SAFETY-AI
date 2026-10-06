"""LabelMatchPolicy: per-label thresholds and the persistence (flicker) filter."""

import unittest

from src.cctv_core.events.policies import LabelMatchPolicy, center_shift
from src.cctv_core.schemas import BBox, Detection, DetectorOutput


def d(label, conf, box):
    return Detection(label, conf, BBox(*box))


def out(dets, t, cam="cam0"):
    return DetectorOutput("hazard_object", cam, 0, t, dets)


class PerLabelThresholdTests(unittest.TestCase):
    def test_override_applies_only_to_its_label(self):
        p = LabelMatchPolicy("hazard_object", ["scissors", "knife"], 0.25,
                             per_label_confidence={"knife": 0.5})
        self.assertEqual(p.evaluate(out([d("knife", 0.4, (0, 0, 10, 10))], 1.0)), [])
        self.assertEqual(len(p.evaluate(out([d("scissors", 0.3, (0, 0, 10, 10))], 1.0))), 1)

    def test_override_case_insensitive(self):
        p = LabelMatchPolicy("hazard_object", ["scissors"], 0.9, per_label_confidence={"Scissors": 0.3})
        self.assertEqual(len(p.evaluate(out([d("SCISSORS", 0.35, (0, 0, 10, 10))], 1.0))), 1)

    def test_unknown_label_is_a_config_error(self):
        with self.assertRaises(ValueError):
            LabelMatchPolicy("hazard_object", ["scissors"], per_label_confidence={"knfe": 0.3})

    def test_default_behaviour_unchanged(self):
        p = LabelMatchPolicy("hazard_object", ["scissors", "knife"], 0.4)
        self.assertEqual(p.evaluate(out([d("knife", 0.39, (0, 0, 10, 10))], 1.0)), [])
        self.assertEqual(len(p.evaluate(out([d("knife", 0.40, (0, 0, 10, 10))], 1.0))), 1)


class PersistenceTests(unittest.TestCase):
    def policy(self, **kw):
        opts = {"enabled": True, "min_frames": 2, "window_s": 1.0, "max_shift": 1.5}
        opts.update(kw)
        return LabelMatchPolicy("hazard_object", ["scissors", "knife"], 0.25, persistence=opts)

    def test_first_sighting_does_not_count(self):
        p = self.policy()
        self.assertEqual(p.evaluate(out([d("scissors", 0.9, (100, 100, 140, 160))], 1.0)), [])

    def test_same_place_twice_counts(self):
        p = self.policy()
        p.evaluate(out([d("scissors", 0.9, (100, 100, 140, 160))], 1.0))
        self.assertEqual(len(p.evaluate(out([d("scissors", 0.9, (105, 102, 145, 162))], 1.1))), 1)

    def test_flicker_in_different_places_never_counts(self):
        p = self.policy()
        for k, x in enumerate((0, 400, 800, 1200)):
            self.assertEqual(p.evaluate(out([d("knife", 0.5, (x, 0, x + 30, 30))], 1.0 + 0.1 * k)), [])

    def test_other_label_does_not_chain(self):
        p = self.policy()
        p.evaluate(out([d("knife", 0.9, (100, 100, 140, 160))], 1.0))
        self.assertEqual(p.evaluate(out([d("scissors", 0.9, (100, 100, 140, 160))], 1.1)), [])

    def test_old_sighting_outside_window_is_forgotten(self):
        p = self.policy(window_s=0.5)
        p.evaluate(out([d("knife", 0.9, (100, 100, 140, 160))], 1.0))
        self.assertEqual(p.evaluate(out([d("knife", 0.9, (100, 100, 140, 160))], 2.0)), [])

    def test_missed_frame_inside_window_is_allowed(self):
        p = self.policy(min_frames=3)
        p.evaluate(out([d("knife", 0.9, (100, 100, 140, 160))], 1.0))
        p.evaluate(out([], 1.1))
        p.evaluate(out([d("knife", 0.9, (102, 100, 142, 160))], 1.2))
        p.evaluate(out([], 1.3))
        self.assertEqual(len(p.evaluate(out([d("knife", 0.9, (104, 100, 144, 160))], 1.4))), 1)

    def test_cameras_are_independent(self):
        p = self.policy()
        p.evaluate(out([d("knife", 0.9, (100, 100, 140, 160))], 1.0, cam="a"))
        self.assertEqual(p.evaluate(out([d("knife", 0.9, (100, 100, 140, 160))], 1.1, cam="b")), [])

    def test_slow_moving_object_still_chains(self):
        p = self.policy(min_frames=4)
        res = []
        for k in range(4):  # moves half a box per frame, e.g. carried by a child
            x = 100 + 20 * k
            res = p.evaluate(out([d("scissors", 0.8, (x, 100, x + 40, 160))], 1.0 + 0.07 * k))
        self.assertEqual(len(res), 1)

    def test_disabled_by_default(self):
        p = LabelMatchPolicy("hazard_object", ["knife"], 0.25)
        self.assertEqual(len(p.evaluate(out([d("knife", 0.9, (0, 0, 10, 10))], 1.0))), 1)

    def test_center_shift_units(self):
        a, b = BBox(0, 0, 30, 40), BBox(50, 0, 80, 40)  # diagonal 50, centres 50 apart
        self.assertAlmostEqual(center_shift(a, b), 1.0)


if __name__ == "__main__":
    unittest.main()
