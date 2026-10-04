import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("tools").resolve()))
import preflight  # noqa: E402

CAL = {"near": {"foot_y": 0.9, "height": 0.6}, "far": {"foot_y": 0.5, "height": 0.3}}
GATE = {"name": "gate", "polygon": [[0.5, 0.5], [1, 0.5], [1, 1]], "calibration": CAL}


def cfg_with(weights="yolo11n.pt", zones=None, line=False):
    policies = [{"class": "src.cctv_core.events.policies:LabelMatchPolicy",
                 "params": {"event_type": "hazard_object", "labels": ["scissors"]}}]
    if zones is not None:
        policies.append({"class": "src.policies.out_of_area:OutOfAreaPolicy",
                         "params": {"zones_file": None, "zones": zones}})
    return {
        "cameras": [{"id": "cam0", "source": 0}],
        "modules": [{"detector": {"class": "src.detectors.hazard_object:HazardObjectDetector",
                                  "params": {"weights": weights}},
                     "policies": policies}],
        "notifications": {"channels": {"line": {"enabled": line}}},
    }


class FakeLine:
    def __init__(self, configured):
        self.configured = configured


def statuses(checks):
    return [c.status for c in checks]


class WeightTests(unittest.TestCase):
    def test_missing_weights_fail(self):
        checks = preflight.check_weights(cfg_with(weights="no_such_model.pt"))
        self.assertEqual(statuses(checks), ["FAIL"])
        self.assertIn("internet", checks[0].detail)

    def test_present_weights_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp) / "m.pt"
            w.write_bytes(b"weights")
            self.assertEqual(statuses(preflight.check_weights(cfg_with(weights=str(w)))), ["OK"])


class ZoneTests(unittest.TestCase):
    def test_out_of_area_without_zone_warns(self):
        checks = preflight.check_zones(cfg_with(zones={}), "cam0")
        self.assertEqual(statuses(checks), ["WARN"])

    def test_calibrated_zone_ok(self):
        checks = preflight.check_zones(cfg_with(zones={"cam0": [GATE]}), "cam0")
        self.assertEqual(statuses(checks), ["OK"])

    def test_uncalibrated_child_zone_warns(self):
        zone = {k: v for k, v in GATE.items() if k != "calibration"}
        checks = preflight.check_zones(cfg_with(zones={"cam0": [zone]}), "cam0")
        self.assertEqual(statuses(checks), ["WARN"])

    def test_no_out_of_area_module_means_nothing_to_check(self):
        self.assertEqual(preflight.check_zones(cfg_with(), "cam0"), [])


class LineTests(unittest.TestCase):
    def test_line_off(self):
        self.assertEqual(statuses(preflight.check_line(cfg_with(line=False))), ["OK"])

    def test_line_on_without_credentials_fails(self):
        checks = preflight.check_line(cfg_with(line=True), line_factory=lambda: FakeLine(False))
        self.assertEqual(statuses(checks), ["FAIL"])

    def test_line_on_with_credentials_ok_and_never_shows_values(self):
        checks = preflight.check_line(cfg_with(line=True), line_factory=lambda: FakeLine(True))
        self.assertEqual(statuses(checks), ["OK"])
        self.assertNotIn("=", checks[0].detail)


class SourceTests(unittest.TestCase):
    def test_camera_that_cannot_open_fails(self):
        def boom(*a, **k):
            raise RuntimeError("Could not open camera 0")
        check = preflight.check_source(0, {}, open_fn=boom)
        self.assertEqual(check.status, "FAIL")

    def test_camera_that_reads_a_frame_ok(self):
        class Src:
            def read(self):
                return np.zeros((480, 640, 3), dtype=np.uint8)

            def release(self):
                pass
        check = preflight.check_source(0, {}, open_fn=lambda *a, **k: Src())
        self.assertEqual(check.status, "OK")
        self.assertIn("640x480", check.detail)

    def test_missing_fallback_clip_fails(self):
        self.assertEqual(preflight.check_fallback("no/such/backup.mp4").status, "FAIL")


class StorageTests(unittest.TestCase):
    def test_writable_storage_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"storage": {"database": f"{tmp}/e.db", "snapshots_dir": f"{tmp}/snaps"}}
            self.assertEqual(set(statuses(preflight.check_storage(cfg))), {"OK"})


class OverallTests(unittest.TestCase):
    def test_exit_code(self):
        ok = preflight.Check("a", "OK", "")
        warn = preflight.Check("b", "WARN", "")
        fail = preflight.Check("c", "FAIL", "")
        self.assertEqual(preflight.exit_code([ok, warn]), 0)
        self.assertEqual(preflight.exit_code([ok, fail]), 1)


if __name__ == "__main__":
    unittest.main()
