import unittest

from src.cctv_core.factory import build_event_system
from src.cctv_core.schemas import EventCandidate, Severity
from src.utils.config import load_config
from tests.fakes import det


class FactoryTests(unittest.TestCase):
    def test_core_yaml_builds_and_runs_end_to_end(self):
        cfg = load_config("config/core.yaml")
        system = build_event_system(cfg, database=":memory:", async_notifications=False)
        self.assertIn("console", system.router.channels)
        self.assertNotIn("line", system.router.channels)  # disabled by default
        self.assertIs(system.manager.config_for("hazard_object").severity, Severity.HIGH)
        self.assertEqual(system.camera_names["cam0"], "Notebook Camera")

        ev = None
        for t in (0.0, 0.5, 1.0):
            ev = system.manager.submit(
                EventCandidate("hazard_object", "cam0", "fake", t, 0.8, det("knife", 0.8)))
        self.assertIsNotNone(ev)
        stored = system.store.get(ev.event_id)
        self.assertEqual(stored["notifications"][0]["channel"], "console")
        system.shutdown(20.0)


class ArchitectureRuleTests(unittest.TestCase):
    def test_core_detector_layer_does_not_import_alerts(self):
        """Detectors must never talk to LINE directly."""
        from pathlib import Path
        for path in list(Path("src/cctv_core").glob("detector_base.py")) + list(Path("src/detectors").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("line_notifier", text, f"{path} imports the LINE notifier")
            self.assertNotIn("LineChannel", text, f"{path} uses a notification channel")


if __name__ == "__main__":
    unittest.main()
