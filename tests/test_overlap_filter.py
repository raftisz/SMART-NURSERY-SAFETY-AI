"""cigarette boxes that overlap a scissors/knife box are not counted (switch, default off)."""

import unittest

from src.cctv_core.events.policies import LabelMatchPolicy
from src.cctv_core.runner import Module, Runner, build_modules
from src.cctv_core.schemas import BBox, Detection, DetectorOutput
from src.utils.config import load_config
from tests.fakes import FakeDetector, temp_event_system
from tests.test_runner import Clock, FakeSource

SHARP = {"enabled": True, "labels": ["cigarette"], "with": ["scissors", "knife"]}


def d(label, conf, box):
    return Detection(label, conf, BBox(*box))


def smoking_output(dets, hazard=None):
    out = DetectorOutput("smoking", "cam0", 0, 1.0, dets)
    if hazard is not None:
        out.context["hazard_object"] = DetectorOutput("hazard_object", "cam0", 0, 1.0, hazard)
    return out


def policy(enabled=True):
    return LabelMatchPolicy("smoking", ["smoking", "cigarette"], 0.5,
                            ignore_if_overlapping=dict(SHARP, enabled=enabled))


class PolicyTests(unittest.TestCase):
    def test_cigarette_on_scissors_is_not_counted(self):
        out = smoking_output([d("cigarette", 0.60, (100, 100, 140, 160))],
                             hazard=[d("scissors", 0.88, (90, 90, 150, 170))])
        self.assertEqual(policy().evaluate(out), [])

    def test_cigarette_away_from_scissors_still_counts(self):
        out = smoking_output([d("cigarette", 0.60, (400, 100, 440, 160))],
                             hazard=[d("scissors", 0.88, (90, 90, 150, 170))])
        self.assertEqual(len(policy().evaluate(out)), 1)

    def test_smoking_label_is_never_filtered(self):
        out = smoking_output([d("smoking", 0.86, (100, 100, 140, 160))],
                             hazard=[d("knife", 0.5, (90, 90, 150, 170))])
        self.assertEqual(len(policy().evaluate(out)), 1)

    def test_switch_off_counts_everything(self):
        out = smoking_output([d("cigarette", 0.60, (100, 100, 140, 160))],
                             hazard=[d("scissors", 0.88, (90, 90, 150, 170))])
        self.assertEqual(len(policy(enabled=False).evaluate(out)), 1)

    def test_no_hazard_output_counts_everything(self):
        out = smoking_output([d("cigarette", 0.60, (100, 100, 140, 160))])
        self.assertEqual(len(policy().evaluate(out)), 1)

    def test_touching_edges_do_not_overlap(self):
        out = smoking_output([d("cigarette", 0.60, (150, 100, 190, 160))],
                             hazard=[d("scissors", 0.88, (90, 90, 150, 170))])
        self.assertEqual(len(policy().evaluate(out)), 1)


class FixedDetector(FakeDetector):
    def __init__(self, name, dets, **kw):
        super().__init__(**kw)
        self.name, self.dets = name, dets

    def process(self, frame):
        return DetectorOutput(self.name, frame.camera_id, frame.frame_index, frame.timestamp, list(self.dets))


class RunnerContextTests(unittest.TestCase):
    def run_pair(self, enabled):
        events = temp_event_system(self)
        hazard = Module(FixedDetector("hazard_object", [d("scissors", 0.88, (90, 90, 150, 170))]),
                        LabelMatchPolicy("hazard_object", ["scissors", "knife"]))
        smoking = Module(FixedDetector("smoking", [d("cigarette", 0.60, (100, 100, 140, 160))]),
                         policy(enabled))
        return Runner(FakeSource(20), [hazard, smoking], events, display=False, clock=Clock()).run()

    def test_runner_gives_later_modules_this_frames_hazard_boxes(self):
        self.assertEqual(self.run_pair(enabled=False).confirmed, 2)   # hazard + smoking
        self.assertEqual(self.run_pair(enabled=True).confirmed, 1)    # hazard only


class ConfigTests(unittest.TestCase):
    def test_switch_is_in_core_yaml_and_off(self):
        cfg = load_config("config/core.yaml")
        (module,) = build_modules(cfg, only={"smoking"})
        (p,) = module.policies
        self.assertFalse(p.ignore_if_overlapping["enabled"])
        self.assertEqual(p.ignore_if_overlapping["labels"], ["cigarette"])
        self.assertEqual(p.ignore_if_overlapping["with"], ["scissors", "knife"])

    def test_hazard_runs_before_smoking(self):
        names = [m.detector.name for m in build_modules(load_config("config/core.yaml"))]
        self.assertLess(names.index("hazard_object"), names.index("smoking"))


if __name__ == "__main__":
    unittest.main()
