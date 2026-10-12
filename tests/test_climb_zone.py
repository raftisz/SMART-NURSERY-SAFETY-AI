import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.cctv_core.events.manager import EventManager, EventTypeConfig
from src.cctv_core.runner import build_modules
from src.cctv_core.schemas import BBox, Detection, DetectorOutput, EventStatus, Keypoint
from src.cctv_core.zones import Zone, load_zones, save_zone
from src.detectors.climbing_pose import ClimbingPosePolicy
from src.policies.climb_zone import ClimbZonePolicy, foot_point
from src.policies.out_of_area import OutOfAreaPolicy
from src.utils.config import load_config

sys.path.insert(0, str(Path("tools").resolve()))
import define_zone  # noqa: E402

W = H = 1000
TABLE_TOP = [[0.5, 0.3], [0.9, 0.3], [0.9, 0.5], [0.5, 0.5]]      # drawn on the table's top surface
AT_0930 = datetime(2026, 10, 5, 9, 30).timestamp()
L_ANKLE, R_ANKLE = 15, 16


def person(box, ankles=None, ankle_conf=0.9):
    """box in pixels; ankles = (x, y) used for both ankles, or None for an unseen ankle."""
    kps = [Keypoint((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 0.9) for _ in range(17)]
    ax, ay = ankles if ankles else (0.0, 0.0)
    kps[L_ANKLE] = kps[R_ANKLE] = Keypoint(ax, ay, ankle_conf if ankles else 0.05)
    return Detection("person", 0.9, BBox(*box), keypoints=kps)


ON_TABLE = dict(box=(650, 50, 750, 450), ankles=(700, 440))        # feet on the table top
ON_FLOOR = dict(box=(100, 400, 200, 950), ankles=(150, 940))       # standing on the floor, left
HIDDEN_LEGS = dict(box=(650, 50, 750, 450), ankles=None)           # behind the table: ankles unseen
BOX_IN_ANKLES_OUT = dict(box=(650, 50, 750, 450), ankles=(700, 700))  # bottom edge in zone, ankles below it


def output(*dets, cam="cam0", ts=AT_0930):
    return DetectorOutput("climbing_pose", cam, 0, ts, list(dets), frame_width=W, frame_height=H)


def climb(**kw):
    z = {"name": "table", "polygon": TABLE_TOP, "kind": "climb", "child_filter": False}
    z.update(kw)
    return Zone.from_dict(z)


def exit_zone():
    return Zone.from_dict({"name": "gate", "polygon": TABLE_TOP, "child_filter": False})


class ZoneKindTests(unittest.TestCase):
    def test_kind_defaults_to_exit_and_old_files_stay_byte_identical(self):
        z = Zone.from_dict({"name": "gate", "polygon": TABLE_TOP})
        self.assertEqual(z.kind, "exit")
        self.assertNotIn("kind", z.to_dict())            # existing zones.yaml content does not change

    def test_climb_kind_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zones.yaml"
            save_zone(path, "cam0", climb())
            save_zone(path, "cam0", exit_zone())
            kinds = {z.name: z.kind for z in load_zones(path)["cam0"]}
        self.assertEqual(kinds, {"table": "climb", "gate": "exit"})

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            Zone.from_dict({"name": "x", "polygon": TABLE_TOP, "kind": "roof"})

    def test_out_of_area_ignores_climb_zones(self):
        p = OutOfAreaPolicy(zones={"cam0": [climb().to_dict(), exit_zone().to_dict()]})
        self.assertEqual([z.name for z in p.zones["cam0"]], ["gate"])

    def test_climb_policy_ignores_exit_zones(self):
        p = ClimbZonePolicy(zones={"cam0": [exit_zone()]})
        self.assertEqual(p.evaluate(output(person(**ON_TABLE))), [])

    def test_define_zone_builds_a_climb_zone_without_calibration(self):
        z = define_zone.build_zone("table", [(0, 0), (10, 0), (10, 10)], None, None, 100, 100, [], True, kind="climb")
        self.assertEqual((z.kind, z.child_filter, z.calibration), ("climb", False, None))


class ClimbZoneRuleTests(unittest.TestCase):
    def policy(self, *zones):
        return ClimbZonePolicy(zones={"cam0": list(zones or [climb()])})

    def test_visible_ankles_in_zone_is_a_candidate_for_that_zone(self):
        c = self.policy().evaluate(output(person(**ON_TABLE)))
        self.assertEqual(len(c), 1)
        self.assertEqual((c[0].event_type, c[0].subject, c[0].extra["zone"]), ("climbing", "table", "table"))

    def test_hidden_ankles_never_count_even_with_the_box_bottom_in_the_zone(self):
        self.assertEqual(self.policy().evaluate(output(person(**HIDDEN_LEGS))), [])

    def test_box_bottom_in_zone_but_ankles_outside_does_not_count(self):
        self.assertEqual(self.policy().evaluate(output(person(**BOX_IN_ANKLES_OUT))), [])

    def test_standing_on_the_floor_outside_the_zone(self):
        self.assertEqual(self.policy().evaluate(output(person(**ON_FLOOR))), [])

    def test_low_confidence_ankles_do_not_count(self):
        self.assertEqual(self.policy().evaluate(output(person(**ON_TABLE, ankle_conf=0.2))), [])

    def test_inactive_hours(self):
        p = self.policy(climb(active_hours=["13:00-14:00"]))
        self.assertEqual(p.evaluate(output(person(**ON_TABLE))), [])

    def test_other_camera_has_no_zones(self):
        self.assertEqual(self.policy().evaluate(output(person(**ON_TABLE), cam="cam1")), [])

    def test_foot_point_is_the_mean_of_visible_ankles(self):
        d = person(**ON_TABLE)
        d.keypoints[R_ANKLE] = Keypoint(720, 460, 0.9)
        self.assertEqual(foot_point(d, W, H, 0.3), (0.71, 0.45))


class DurationTests(unittest.TestCase):
    """How long is decided by the Event Manager (min_duration_s), never by the policy."""

    def run_for(self, seconds, step=0.1):
        em = EventManager(type_configs={"climbing": EventTypeConfig(min_duration_s=1.0, max_gap_s=0.8)})
        p = ClimbZonePolicy(zones={"cam0": [climb()]})
        events = []
        for i in range(int(round(seconds / step)) + 1):
            for c in p.evaluate(output(person(**ON_TABLE), ts=AT_0930 + i * step)):
                ev = em.submit(c)
                if ev is not None:
                    events.append(ev)
        return events

    def test_staying_long_enough_confirms(self):
        events = self.run_for(1.5)
        self.assertEqual([e.status for e in events], [EventStatus.CONFIRMED])
        self.assertEqual(events[0].subject, "table")

    def test_a_short_pass_does_not_confirm(self):
        self.assertEqual(self.run_for(0.6), [])


class ModeTests(unittest.TestCase):
    def test_default_mode_is_the_old_pose_rule(self):
        self.assertEqual(ClimbingPosePolicy().mode, "pose")

    def test_zone_mode_delegates_to_the_zone_rule(self):
        p = ClimbingPosePolicy(mode="zone", zones={"cam0": [climb()]})
        c = p.evaluate(output(person(**ON_TABLE)))
        self.assertEqual((len(c), c[0].subject), (1, "table"))

    def test_zone_mode_does_not_use_the_pose_rule(self):
        # hands above nose + ankle above hips = DANGER in pose mode, but not in a climb zone
        d = person(box=(100, 400, 200, 950), ankles=(150, 410))
        self.assertEqual(ClimbingPosePolicy(mode="zone", zones={"cam0": [climb()]}).evaluate(output(d)), [])

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            ClimbingPosePolicy(mode="radar")

    def test_core_yaml_keeps_the_old_behaviour_and_stays_off(self):
        cfg = load_config("config/core.yaml")
        self.assertNotIn("climbing_pose", [m.detector.name for m in build_modules(cfg)])
        pose = build_modules(cfg, only={"climbing_pose"})[0]
        climbing = next(p for p in pose.policies if isinstance(p, ClimbingPosePolicy))
        self.assertEqual((climbing.mode, climbing.require_both), ("pose", True))
        self.assertEqual(cfg["events"]["types"]["climbing"], {"severity": "HIGH"})


if __name__ == "__main__":
    unittest.main()
