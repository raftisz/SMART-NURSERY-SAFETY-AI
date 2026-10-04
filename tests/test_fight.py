import unittest

from src.cctv_core.events.manager import EventTypeConfig
from src.cctv_core.factory import build_event_system
from src.cctv_core.runner import Module, Runner, build_modules
from src.cctv_core.schemas import BBox, Detection, DetectorOutput, Keypoint
from src.policies.fight import FightPolicy, SimpleTracker, box_gap
from src.utils.config import load_config
from tests.fakes import FakeDetector
from tests.test_runner import Clock, FakeSource

H = 400  # person box height in pixels


def person(x1, wrist=None, conf=0.9):
    """A 200 x 400 px person whose left box edge is x1. wrist = (x, y) of the right wrist."""
    kps = [Keypoint(x1 + 100, 200, 0.9) for _ in range(17)]
    if wrist is not None:
        kps[10] = Keypoint(wrist[0], wrist[1], 0.9)
    return Detection("person", conf, BBox(x1, 100, x1 + 200, 100 + H), keypoints=kps)


def out(dets, t):
    return DetectorOutput("climbing_pose", "cam0", int(t * 10), t, dets, frame_width=1920, frame_height=1080)


def run_seq(policy, frames):
    """frames: list of (t, [detections]) -> candidates per frame."""
    return [policy.evaluate(out(d, t)) for t, d in frames]


class HelperTests(unittest.TestCase):
    def test_box_gap(self):
        self.assertEqual(box_gap(BBox(0, 0, 10, 10), BBox(5, 5, 20, 20)), 0)
        self.assertEqual(box_gap(BBox(0, 0, 10, 10), BBox(13, 0, 20, 10)), 3)

    def test_tracker_keeps_ids_while_moving(self):
        tr = SimpleTracker()
        a, b = person(100), person(700)
        first = [t.track_id for t in tr.update([a, b], 0.0)]
        moved = [t.track_id for t in tr.update([person(720), person(110)], 0.1)]  # order swapped
        self.assertEqual(moved, [first[1], first[0]])

    def test_tracker_drops_old_tracks(self):
        tr = SimpleTracker(max_age_s=1.0)
        old = tr.update([person(100)], 0.0)[0].track_id
        new = tr.update([person(100)], 2.0)[0].track_id
        self.assertNotEqual(old, new)


class FightRuleTests(unittest.TestCase):
    def test_fast_wrist_into_other_person_fires(self):
        # A at x 100..300, B at x 300..500 (touching). A's wrist swings from 250 into B (450).
        p = FightPolicy()
        frames = [(t / 10, [person(100, wrist=(250 if t % 4 < 2 else 450, 250)), person(300)])
                  for t in range(12)]
        res = run_seq(p, frames)
        self.assertTrue(any(res))
        fired = next(r for r in res if r)[0]
        self.assertEqual(fired.event_type, "fight")
        self.assertGreaterEqual(fired.extra["wrist_speed"], 1.5)

    def test_hug_close_but_slow_does_not_fire(self):
        p = FightPolicy()
        frames = [(t / 10, [person(100, wrist=(350, 250)), person(300)]) for t in range(12)]
        self.assertFalse(any(run_seq(p, frames)))

    def test_far_apart_fast_does_not_fire(self):
        p = FightPolicy()
        frames = [(t / 10, [person(100, wrist=(150 if t % 4 < 2 else 290, 250)), person(900)])
                  for t in range(12)]
        self.assertFalse(any(run_seq(p, frames)))

    def test_dancing_side_by_side_hands_stay_out(self):
        # close, fast wrist, but it moves up and down inside A's own box
        p = FightPolicy()
        frames = [(t / 10, [person(100, wrist=(200, 150 if t % 4 < 2 else 450)), person(320)])
                  for t in range(12)]
        self.assertFalse(any(run_seq(p, frames)))

    def test_single_person(self):
        p = FightPolicy()
        frames = [(t / 10, [person(100, wrist=(150 if t % 4 < 2 else 290, 250))]) for t in range(12)]
        self.assertFalse(any(run_seq(p, frames)))

    def test_low_confidence_keypoints_ignored(self):
        p = FightPolicy()
        frames = []
        for t in range(12):
            a = person(100, wrist=(250 if t % 4 < 2 else 450, 250))
            a.keypoints[10] = Keypoint(a.keypoints[10].x, 250, 0.1)
            frames.append((t / 10, [a, person(300)]))
        self.assertFalse(any(run_seq(p, frames)))

    def test_left_right_label_swap_is_not_speed(self):
        # A stands still, one wrist in B's box. The pose model swaps the L/R
        # labels every frame, which used to look like a wrist jumping 225 px.
        p = FightPolicy()
        frames = []
        for t in range(12):
            a = person(100)
            near_b, own = Keypoint(330, 250, 0.9), Keypoint(105, 250, 0.9)
            a.keypoints[9], a.keypoints[10] = (near_b, own) if t % 2 else (own, near_b)
            frames.append((t / 10, [a, person(300)]))
        self.assertFalse(any(run_seq(p, frames)))

    def test_track_identity_swap_is_not_speed(self):
        # Two still people almost on top of each other; their boxes jitter so the
        # IoU tracker swaps identities every frame. Each wrist is still.
        p = FightPolicy()
        frames = []
        for t in range(12):
            a, b = person(100, wrist=(300, 250)), person(110, wrist=(120, 250))
            if t % 2:
                a.bbox, b.bbox = BBox(110, 100, 310, 500), BBox(100, 100, 300, 500)
            frames.append((t / 10, [a, b]))
        self.assertFalse(any(run_seq(p, frames)))

    def test_speed_shown_on_people(self):
        p = FightPolicy()
        a = person(100)
        p.evaluate(out([a, person(700)], 0.0))
        self.assertIn("spd", a.attributes["tag"])


class EndToEndTests(unittest.TestCase):
    def run_frames(self, pattern, frames):
        events = build_event_system(load_config("config/core.yaml"), database=":memory:",
                                    async_notifications=False)
        events.manager.type_configs["fight"] = EventTypeConfig(min_duration_s=1.0, max_gap_s=0.8)
        det = FakeDetector()
        det.process = lambda f: out(pattern(f.frame_index), f.frame_index / 10)
        return Runner(FakeSource(frames), [Module(det, FightPolicy())], events,
                      display=False, clock=Clock(0.1)).run()

    def test_sustained_fight_alerts_once(self):
        swing = lambda i: [person(100, wrist=(250 if i % 4 < 2 else 450, 250)), person(300)]
        self.assertEqual(self.run_frames(swing, 40).confirmed, 1)

    def test_one_high_five_does_not_alert(self):
        def once(i):
            w = (250, 250) if i != 5 else (450, 250)
            return [person(100, wrist=w), person(300)]
        self.assertEqual(self.run_frames(once, 40).confirmed, 0)


class ConfigTests(unittest.TestCase):
    def test_pose_module_off_by_default_feeds_two_policies(self):
        cfg = load_config("config/core.yaml")
        self.assertNotIn("climbing_pose", [m.detector.name for m in build_modules(cfg)])
        pose = build_modules(cfg, only={"climbing_pose"})
        self.assertEqual(len(pose), 1)                      # named explicitly -> runs
        self.assertEqual([type(p).__name__ for p in pose[0].policies],
                         ["ClimbingPosePolicy", "FightPolicy"])


if __name__ == "__main__":
    unittest.main()
