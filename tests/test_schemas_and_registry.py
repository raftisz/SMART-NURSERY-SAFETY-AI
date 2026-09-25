import json
import unittest
from types import SimpleNamespace

from src.cctv_core.registry import build_detector, build_policy, import_object
from src.cctv_core.schemas import BBox, Detection, Event, EventStatus, Severity
from src.cctv_core.events.policies import LabelMatchPolicy
from src.cctv_core.schemas import DetectorOutput, Frame
from tests.fakes import FakeDetector, det


class SchemaTests(unittest.TestCase):
    def test_bbox_geometry(self):
        b = BBox(10, 20, 40, 60)
        self.assertEqual((b.width, b.height, b.area), (30, 40, 1200))

    def test_from_legacy_detection(self):
        legacy = SimpleNamespace(class_name="scissors", confidence=0.61, x1=1, y1=2, x2=30, y2=40)
        d = Detection.from_legacy(legacy)
        self.assertEqual(d.label, "scissors")
        self.assertAlmostEqual(d.confidence, 0.61)
        self.assertEqual((d.bbox.x1, d.bbox.y2), (1.0, 40.0))

    def test_severity_order_and_parse(self):
        self.assertTrue(Severity.HIGH >= Severity.MEDIUM)
        self.assertTrue(Severity.LOW < Severity.MEDIUM)
        self.assertIs(Severity.parse("high"), Severity.HIGH)

    def test_event_to_dict_is_json(self):
        e = Event("smoking", "cam0", "fake", Severity.MEDIUM, 1.0, 1.0, peak_detection=det("smoking", 0.7))
        data = e.to_dict()
        json.dumps(data, ensure_ascii=False)
        self.assertEqual(data["severity"], "MEDIUM")
        self.assertEqual(data["status"], EventStatus.CANDIDATE.value)
        self.assertEqual(data["schema_version"], "1.0")


class RegistryTests(unittest.TestCase):
    def test_build_detector_from_config(self):
        d = build_detector({"class": "tests.fakes:FakeDetector", "params": {"run_every_n_frames": 3}})
        self.assertIsInstance(d, FakeDetector)
        self.assertTrue(d.should_run(0))
        self.assertFalse(d.should_run(1))

    def test_build_policy_from_config(self):
        p = build_policy({
            "class": "src.cctv_core.events.policies:LabelMatchPolicy",
            "params": {"event_type": "hazard_object", "labels": ["knife"]},
        })
        self.assertIsInstance(p, LabelMatchPolicy)

    def test_wrong_base_class_rejected(self):
        with self.assertRaises(TypeError):
            build_detector({"class": "src.cctv_core.events.policies:LabelMatchPolicy",
                            "params": {"event_type": "x", "labels": ["y"]}})

    def test_bad_path(self):
        with self.assertRaises(ValueError):
            import_object("no_colon_here")


class PolicyTests(unittest.TestCase):
    def output(self, labels):
        d = FakeDetector(labels=labels)
        return d.process(Frame("cam0", 0, 100.0))

    def test_matches_label_and_threshold(self):
        p = LabelMatchPolicy("hazard_object", ["scissors", "knife"], min_confidence=0.4)
        out = self.output([("person", 0.9), ("scissors", 0.35), ("knife", 0.6)])
        cands = p.evaluate(out)
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0].detection.label, "knife")
        self.assertEqual(cands[0].extra["match_count"], 1)

    def test_case_insensitive_and_peak(self):
        p = LabelMatchPolicy("smoking", ["Smoking", "cigarette"])
        cands = p.evaluate(self.output([("cigarette", 0.5), ("smoking", 0.8)]))
        self.assertAlmostEqual(cands[0].confidence, 0.8)

    def test_person_alone_is_not_an_event(self):
        p = LabelMatchPolicy("hazard_object", ["scissors", "knife"])
        self.assertEqual(p.evaluate(self.output([("person", 0.99)])), [])


if __name__ == "__main__":
    unittest.main()
