import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.cctv_core.events.policies import LabelMatchPolicy
from src.cctv_core.notifications.templates import format_message, title_for
from src.cctv_core.runner import build_modules
from src.cctv_core.schemas import BBox, Detection, EventCandidate, EventStatus
from src.detectors.smoke import SmokeDetector
from src.detectors.yolo_label import YoloLabelDetector
from src.utils.config import load_config

from .fakes import temp_event_system

V2 = Path("models/smoke/smoke_custom_yolo11n_v2.pt")
V2_SHA = "2a97455c6f96252dfcedf92c149154db1b32d8eee0ef0028bb2d9c80cad95876"


def smoke_entry(cfg):
    return next(m for m in cfg["modules"] if m["detector"]["class"].endswith(":SmokeDetector"))


class SmokeConfigTests(unittest.TestCase):
    def test_off_by_default(self):
        cfg = load_config("config/core.yaml")
        self.assertIs(smoke_entry(cfg).get("enabled"), False)
        self.assertNotIn("smoke", [m.detector.name for m in build_modules(cfg)])

    def test_modules_flag_turns_it_on_with_the_planned_settings(self):
        modules = build_modules(load_config("config/core.yaml"), only={"smoke"})
        self.assertEqual([m.detector.name for m in modules], ["smoke"])
        det, policies = modules[0].detector, modules[0].policies
        self.assertEqual((det.weights, det.sha256), (str(V2), V2_SHA))
        self.assertEqual((det.conf, det.imgsz, det.target_classes), (0.25, 640, ["smoke"]))
        self.assertIsInstance(policies[0], LabelMatchPolicy)
        self.assertEqual((policies[0].event_type, policies[0].labels, policies[0].min_confidence),
                         ("smoke", {"smoke"}, 0.25))

    def test_smoke_is_the_last_module(self):
        """Placed after the others so it never changes the context smoking reads."""
        cfg = load_config("config/core.yaml")
        self.assertTrue(cfg["modules"][-1]["detector"]["class"].endswith(":SmokeDetector"))

    def test_event_settings(self):
        smoke = load_config("config/core.yaml")["events"]["types"]["smoke"]
        self.assertEqual(smoke, {"severity": "MEDIUM", "min_duration_s": 2.0,
                                 "max_gap_s": 1.0, "cooldown_s": 60.0})

    @unittest.skipUnless(V2.is_file(), "v2 weights not on this machine")
    def test_v2_weights_hash(self):
        self.assertEqual(hashlib.sha256(V2.read_bytes()).hexdigest(), V2_SHA)


class SmokeLoadTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.weights = Path(tmp.name) / "w.pt"
        self.weights.write_bytes(b"not a model")
        self.sha = hashlib.sha256(b"not a model").hexdigest()

    def test_wrong_hash_stops_before_the_model_is_loaded(self):
        with mock.patch.object(YoloLabelDetector, "load") as inner:
            with self.assertRaises(ValueError):
                SmokeDetector(weights=str(self.weights), sha256="0" * 64).load()
            inner.assert_not_called()

    def test_no_expected_hash_is_refused(self):
        with mock.patch.object(YoloLabelDetector, "load") as inner:
            with self.assertRaises(ValueError):
                SmokeDetector(weights=str(self.weights)).load()
            inner.assert_not_called()

    def test_missing_file_is_refused(self):
        with self.assertRaises(FileNotFoundError):
            SmokeDetector(weights=str(self.weights.with_name("nope.pt")), sha256=self.sha).load()

    def test_model_without_smoke_class_is_an_error(self):
        def fake_load(det):
            det._yolo = SimpleNamespace(missing_classes=["smoke"])
        with mock.patch.object(YoloLabelDetector, "load", fake_load):
            with self.assertRaises(ValueError):
                SmokeDetector(weights=str(self.weights), sha256=self.sha).load()

    def test_matching_hash_and_class_loads(self):
        def fake_load(det):
            det._yolo = SimpleNamespace(missing_classes=[])
        with mock.patch.object(YoloLabelDetector, "load", fake_load):
            SmokeDetector(weights=str(self.weights), sha256=self.sha).load()


class SmokeEventTests(unittest.TestCase):
    def submit(self, manager, t):
        d = Detection("smoke", 0.4, BBox(0, 0, 10, 10))
        return manager.submit(EventCandidate("smoke", "cam0", "smoke", t, 0.4, detection=d))

    def test_needs_two_seconds_in_a_row(self):
        manager = temp_event_system(self).manager
        results = [self.submit(manager, 100.0 + i * 0.2) for i in range(10)]   # 0.0 .. 1.8 s
        self.assertTrue(all(r is None for r in results))
        event = self.submit(manager, 102.0)
        self.assertEqual(event.status, EventStatus.CONFIRMED)

    def test_a_gap_over_one_second_restarts_the_count(self):
        manager = temp_event_system(self).manager
        for t in (100.0, 100.5, 101.0, 102.5, 103.0, 103.5, 104.0):          # gap 1.5 s at 101.0
            self.assertIsNone(self.submit(manager, t))
        self.assertEqual(self.submit(manager, 104.5).status, EventStatus.CONFIRMED)

    def test_message_says_possible_smoke(self):
        manager = temp_event_system(self).manager
        event = None
        for i in range(11):
            event = self.submit(manager, 100.0 + i * 0.2) or event
        self.assertIn("possible smoke", title_for("smoke"))
        self.assertIn("possible smoke", format_message(event))


if __name__ == "__main__":
    unittest.main()
