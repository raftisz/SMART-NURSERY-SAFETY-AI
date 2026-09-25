import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.cctv_core.events.policies import LabelMatchPolicy
from src.cctv_core.events.snapshot import SnapshotWriter
from src.cctv_core.factory import build_event_system
from src.cctv_core.runner import Module, Runner, build_modules
from src.cctv_core.stream.camera_source import CameraSource
from src.utils.config import load_config
from tests.fakes import FakeDetector


class FakeSource:
    def __init__(self, n):
        self.n = n
        self.released = False

    def read(self):
        if self.n <= 0:
            return None
        self.n -= 1
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class CountingDetector(FakeDetector):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        return super().process(frame)


class ExplodingDetector(FakeDetector):
    name = "boom"

    def process(self, frame):
        raise RuntimeError("model crashed")


class Clock:
    def __init__(self, step=0.1):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.events = build_event_system(load_config("config/core.yaml"), database=":memory:",
                                         async_notifications=False)
        self.events.manager.snapshots = SnapshotWriter(self.tmp.name)
        self.store = self.events.store

    def tearDown(self):
        self.tmp.cleanup()

    def run_with(self, modules, frames=30):
        src = FakeSource(frames)
        stats = Runner(src, modules, self.events, display=False, clock=Clock()).run()
        return src, stats

    def test_end_to_end_one_event_with_snapshot(self):
        det = CountingDetector(labels=[("scissors", 0.9)])
        policy = LabelMatchPolicy("hazard_object", ["scissors", "knife"])
        src, stats = self.run_with([Module(det, policy)])
        self.assertTrue(det.loaded)
        self.assertTrue(src.released)
        self.assertEqual(stats.frames, 30)
        self.assertEqual(stats.confirmed, 1)          # 30 sightings -> ONE event
        self.assertEqual(len(list(Path(self.tmp.name).glob("*.jpg"))), 1)

    def test_person_only_produces_no_event(self):
        det = FakeDetector(labels=[("person", 0.99)])
        _, stats = self.run_with([Module(det, LabelMatchPolicy("hazard_object", ["scissors"]))])
        self.assertEqual(stats.confirmed, 0)

    def test_run_every_n_frames(self):
        det = CountingDetector(run_every_n_frames=3)
        self.run_with([Module(det, LabelMatchPolicy("x", ["none"]))], frames=30)
        self.assertEqual(det.calls, 10)

    def test_broken_module_does_not_stop_others(self):
        good = CountingDetector(labels=[("scissors", 0.9)])
        _, stats = self.run_with([
            Module(ExplodingDetector(), LabelMatchPolicy("x", ["scissors"])),
            Module(good, LabelMatchPolicy("hazard_object", ["scissors"])),
        ])
        self.assertEqual(stats.errors["boom"], 30)
        self.assertEqual(good.calls, 30)
        self.assertEqual(stats.confirmed, 1)

    def test_max_frames(self):
        det = CountingDetector()
        stats = Runner(FakeSource(100), [Module(det, LabelMatchPolicy("x", ["none"]))],
                       self.events, display=False, max_frames=5, clock=Clock()).run()
        self.assertEqual(stats.frames, 5)


class ConfigModulesTests(unittest.TestCase):
    def test_core_yaml_modules_build_without_loading(self):
        modules = build_modules(load_config("config/core.yaml"))
        self.assertEqual([m.detector.name for m in modules], ["hazard_object", "smoking"])
        self.assertEqual(modules[1].detector.run_every_n_frames, 2)

    def test_only_filter(self):
        modules = build_modules(load_config("config/core.yaml"), only={"smoking"})
        self.assertEqual([m.detector.name for m in modules], ["smoking"])

class CameraSourceTests(unittest.TestCase):
    def test_still_image_source_read_only(self):
        img = sorted(Path("data/test_frames").glob("*.jpg"))[0]
        before = img.stat().st_mtime
        src = CameraSource(str(img), loop_image=False)
        self.assertEqual(src.kind, "image")
        self.assertFalse(src.flip)             # only webcams are mirrored by default
        self.assertIsNotNone(src.read())
        self.assertIsNone(src.read())
        src.release()
        self.assertEqual(img.stat().st_mtime, before)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            CameraSource("no/such/file.mp4")


if __name__ == "__main__":
    unittest.main()
