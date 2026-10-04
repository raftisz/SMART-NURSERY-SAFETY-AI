import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.cctv_core.events.manager import EventTypeConfig
from src.cctv_core.runner import Module, Runner
from src.cctv_core.schemas import BBox, Detection, DetectorOutput
from src.cctv_core.zones import HeightModel, Zone, load_zones, parse_hours, save_zone
from src.policies.out_of_area import OutOfAreaPolicy
from tests.fakes import FakeDetector, temp_event_system
from tests.test_runner import Clock, FakeSource

sys.path.insert(0, str(Path("tools").resolve()))
import define_zone  # noqa: E402

W = H = 1000
SQUARE = [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]]          # bottom-right quarter
CAL = {"near": {"foot_y": 0.9, "height": 0.6}, "far": {"foot_y": 0.5, "height": 0.3}}
AT_0930 = datetime(2026, 10, 5, 9, 30).timestamp()
AT_1200 = datetime(2026, 10, 5, 12, 0).timestamp()


def person(cx, y1, y2, conf=0.9, half_w=60):
    return Detection("person", conf, BBox(cx - half_w, y1, cx + half_w, y2))


CHILD = dict(cx=750, y1=550, y2=900)   # 0.35 tall where an adult is 0.60 -> ratio 0.58
ADULT = dict(cx=750, y1=320, y2=900)   # 0.58 tall -> ratio 0.97


def output(*dets, cam="cam0", ts=AT_0930):
    return DetectorOutput("hazard_object", cam, 0, ts, list(dets), frame_width=W, frame_height=H)


def zone(**kw):
    z = {"name": "gate", "polygon": SQUARE, "calibration": CAL}
    z.update(kw)
    return z


def policy(*zones, cam="cam0", **kw):
    return OutOfAreaPolicy(zones={cam: list(zones)}, **kw)


class ZoneTests(unittest.TestCase):
    def test_contains_including_concave(self):
        l_shape = Zone("L", [(0, 0), (0.6, 0), (0.6, 0.3), (0.3, 0.3), (0.3, 1), (0, 1)])
        self.assertTrue(l_shape.contains(0.1, 0.9))
        self.assertTrue(l_shape.contains(0.5, 0.1))
        self.assertFalse(l_shape.contains(0.5, 0.8))     # inside the notch

    def test_hours(self):
        self.assertEqual(parse_hours("09:00-15:00"), (540, 900))
        z = Zone("z", [(0, 0), (1, 0), (1, 1)], active_hours=["09:00-10:00"])
        self.assertTrue(z.is_active(AT_0930))
        self.assertFalse(z.is_active(AT_1200))
        night = Zone("n", [(0, 0), (1, 0), (1, 1)], active_hours=["22:00-06:00"])
        self.assertTrue(night.is_active(datetime(2026, 10, 5, 23, 0).timestamp()))
        self.assertTrue(night.is_active(datetime(2026, 10, 5, 5, 0).timestamp()))
        self.assertFalse(night.is_active(AT_1200))
        with self.assertRaises(ValueError):
            parse_hours("9-15")

    def test_validation(self):
        with self.assertRaises(ValueError):
            Zone("x", [(0, 0), (1, 1)])
        with self.assertRaises(ValueError):
            Zone("x", [(0, 0), (1, 0), (640, 480)])      # pixels instead of fractions

    def test_height_model(self):
        m = HeightModel.from_dict(CAL)
        self.assertAlmostEqual(m.expected(0.9), 0.6)
        self.assertAlmostEqual(m.expected(0.7), 0.45)   # in between
        self.assertAlmostEqual(m.expected(0.3), 0.15)   # extrapolated further away
        self.assertGreater(m.expected(-5), 0.0)         # never zero or negative

    def test_height_model_extrapolates_only_half_a_span(self):
        m = HeightModel.from_dict(CAL)                  # calibrated between y 0.5 and 0.9
        self.assertAlmostEqual(m.expected(-5), m.expected(0.3))
        self.assertAlmostEqual(m.expected(5), m.expected(1.1))

    def test_save_and_load_keep_other_zones(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zones.yaml"
            save_zone(path, "cam0", Zone.from_dict(zone(name="gate")))
            save_zone(path, "cam0", Zone.from_dict(zone(name="kitchen")))
            save_zone(path, "cam1", Zone.from_dict(zone(name="gate")))
            save_zone(path, "cam0", Zone.from_dict(zone(name="gate", active_hours=["09:00-15:00"])))
            zones = load_zones(path)
            self.assertEqual(sorted(z.name for z in zones["cam0"]), ["gate", "kitchen"])
            gate = next(z for z in zones["cam0"] if z.name == "gate")
            self.assertEqual(gate.active_hours, ["09:00-15:00"])   # replaced, not duplicated
            self.assertEqual(len(zones["cam1"]), 1)
            self.assertEqual(load_zones(Path(tmp) / "missing.yaml"), {})

    def test_repo_zones_file_is_valid(self):
        # zones are drawn per site with tools/define_zone.py, so the file may hold any
        # number of them; it only has to load (Zone validates points and hours)
        zones = load_zones("config/zones.yaml")
        for cam, zs in zones.items():
            self.assertIsInstance(cam, str)
            for z in zs:
                self.assertGreaterEqual(len(z.polygon), 3)


class PolicyTests(unittest.TestCase):
    def test_child_in_zone(self):
        c = policy(zone()).evaluate(output(person(**CHILD)))
        self.assertEqual(len(c), 1)
        self.assertEqual((c[0].event_type, c[0].subject), ("out_of_area", "gate"))
        self.assertAlmostEqual(c[0].extra["height_ratio"], 0.58, places=2)

    def test_adult_in_zone_ignored(self):
        self.assertEqual(policy(zone()).evaluate(output(person(**ADULT))), [])

    def test_child_outside_zone(self):
        self.assertEqual(policy(zone()).evaluate(output(person(cx=200, y1=550, y2=900))), [])

    def test_outside_active_hours(self):
        p = policy(zone(active_hours=["09:00-10:00"]))
        self.assertEqual(len(p.evaluate(output(person(**CHILD), ts=AT_0930))), 1)
        self.assertEqual(p.evaluate(output(person(**CHILD), ts=AT_1200)), [])

    def test_uncalibrated_zone(self):
        no_cal = zone(calibration=None)
        self.assertEqual(policy(no_cal).evaluate(output(person(**CHILD))), [])
        anyone = zone(calibration=None, child_filter=False)
        self.assertEqual(len(policy(anyone).evaluate(output(person(**ADULT)))), 1)

    def test_box_cut_by_image_edge_is_not_measured(self):
        self.assertEqual(policy(zone()).evaluate(output(person(cx=750, y1=650, y2=1000))), [])

    def test_any_person_zone_ignores_box_cut_by_bottom_edge(self):
        # feet out of the frame: the box ends at (or within 1% of) the image edge, which
        # is not where the feet are. Real boxes end just inside the frame, e.g. y 0.995.
        anyone = zone(polygon=[[0, 0.5], [1, 0.5], [1, 1], [0, 1]], calibration=None, child_filter=False)
        self.assertEqual(policy(anyone).evaluate(output(person(cx=500, y1=300, y2=995))), [])
        self.assertEqual(len(policy(anyone).evaluate(output(person(cx=500, y1=300, y2=950)))), 1)

    def test_low_confidence_and_other_labels(self):
        p = policy(zone())
        self.assertEqual(p.evaluate(output(person(conf=0.3, **CHILD))), [])
        scissors = Detection("scissors", 0.9, BBox(700, 700, 800, 900))
        self.assertEqual(p.evaluate(output(scissors)), [])

    def test_other_camera(self):
        self.assertEqual(policy(zone(), cam="cam1").evaluate(output(person(**CHILD))), [])

    def test_two_zones_two_candidates(self):
        p = policy(zone(name="a"), zone(name="b"))
        self.assertEqual(sorted(c.subject for c in p.evaluate(output(person(**CHILD)))), ["a", "b"])

    def test_zone_child_ratio_override(self):
        strict = zone(child_ratio=0.5)                    # 0.58 is no longer "child"
        self.assertEqual(policy(strict).evaluate(output(person(**CHILD))), [])

    def test_unknown_frame_size(self):
        out = DetectorOutput("d", "cam0", 0, AT_0930, [person(**CHILD)])
        self.assertEqual(policy(zone()).evaluate(out), [])


class DefineZoneToolTests(unittest.TestCase):
    def test_build_zone_from_clicks(self):
        z = define_zone.build_zone(
            "gate", [(500, 500), (1000, 500), (1000, 1000)],
            [(700, 300), (700, 900)], [(600, 200), (600, 500)], W, H, ["09:00-15:00"], False)
        self.assertEqual(z.polygon, [(0.5, 0.5), (1.0, 0.5), (1.0, 1.0)])
        self.assertAlmostEqual(z.calibration.near_height, 0.6)
        self.assertAlmostEqual(z.calibration.far_foot_y, 0.5)

    def test_any_person_needs_no_calibration(self):
        z = define_zone.build_zone("t", [(0, 0), (10, 0), (10, 10)], None, None, 100, 100, [], True)
        self.assertFalse(z.child_filter)
        self.assertIsNone(z.calibration)

    def test_bad_calibration_click(self):
        with self.assertRaises(ValueError):
            define_zone.calibration_point((5, 50), (9, 50), 100, 100)

    def test_swapped_near_and_far_is_refused(self):
        with self.assertRaises(ValueError):
            define_zone.build_zone("gate", [(0, 0), (10, 0), (10, 10)],
                                   [(600, 200), (600, 500)], [(700, 300), (700, 900)],
                                   W, H, [], False)

    def test_near_and_far_at_the_same_spot_is_refused(self):
        with self.assertRaises(ValueError):
            define_zone.build_zone("gate", [(0, 0), (10, 0), (10, 10)],
                                   [(700, 300), (700, 900)], [(600, 310), (600, 890)],
                                   W, H, [], False)

    def test_clicks_near_the_image_edge_snap_to_it(self):
        pts = define_zone.normalise([(5, 995), (500, 970), (990, 15), (1000, 1000)], 1000, 1000)
        self.assertEqual(pts, [(0.0, 1.0), (0.5, 0.97), (1.0, 0.0), (1.0, 1.0)])

    def test_calibration_click_order_does_not_matter(self):
        head_first = define_zone.calibration_point((100, 100), (100, 400), 640, 480)
        feet_first = define_zone.calibration_point((100, 400), (100, 100), 640, 480)
        self.assertEqual(feet_first, head_first)
        self.assertAlmostEqual(head_first[0], 400 / 480, places=4)  # feet = the LOWER click


class EndToEndTests(unittest.TestCase):
    def run_frames(self, box, frames, step=0.1):
        events = temp_event_system(self)
        events.manager.type_configs["out_of_area"] = EventTypeConfig(min_duration_s=3.0)
        # keep the store open after the run so the test can read it
        events.shutdown = lambda now: (events.manager.close_all(now), events.router.close())
        det = FakeDetector()
        det.process = lambda f: DetectorOutput("hazard_object", f.camera_id, f.frame_index,
                                               AT_0930 + f.frame_index * step, [person(**box)],
                                               frame_width=W, frame_height=H)
        stats = Runner(FakeSource(frames), [Module(det, policy(zone()))], events,
                       display=False, clock=Clock(step)).run()
        return stats, events

    def test_child_staying_long_enough_alerts_once(self):
        stats, events = self.run_frames(CHILD, 60)       # 6 s in the zone
        self.assertEqual(stats.confirmed, 1)
        stored = events.store.list_events(event_type="out_of_area")
        self.assertEqual(stored[0]["subject"], "gate")

    def test_child_passing_quickly_does_not_alert(self):
        stats, _ = self.run_frames(CHILD, 20)            # 2 s < 3 s
        self.assertEqual(stats.confirmed, 0)

    def test_adult_never_alerts(self):
        stats, _ = self.run_frames(ADULT, 60)
        self.assertEqual(stats.confirmed, 0)


if __name__ == "__main__":
    unittest.main()
