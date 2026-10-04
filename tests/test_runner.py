import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.cctv_core.events.policies import LabelMatchPolicy
from src.cctv_core.events.snapshot import SnapshotWriter
from src.cctv_core.factory import build_event_system
from src.cctv_core.runner import Module, ModuleLoadError, Runner, build_modules
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

    def test_failing_module_leaves_no_stale_boxes_and_shows_errors(self):
        class BreaksLater(FakeDetector):
            name = "boom"

            def process(self, frame):
                if frame.frame_index >= 5:
                    raise RuntimeError("model crashed")
                return super().process(frame)

        runner = Runner(FakeSource(10), [Module(BreaksLater(), LabelMatchPolicy("x", ["none"]))],
                        self.events, display=False, clock=Clock())
        runner.run()
        self.assertNotIn("boom", runner._last_output)  # old boxes are not drawn forever
        self.assertIn("boom ERR 5", runner.status_text())

    def test_event_system_error_does_not_stop_the_loop(self):
        self.events.manager.submit = mock.Mock(side_effect=RuntimeError("database is locked"))
        det = CountingDetector(labels=[("scissors", 0.9)])
        _, stats = self.run_with([Module(det, LabelMatchPolicy("hazard_object", ["scissors"]))],
                                 frames=10)
        self.assertEqual(stats.frames, 10)
        self.assertEqual(stats.errors["events"], 10)

    def test_load_failure_cleans_up_and_names_the_module(self):
        class NoWeights(FakeDetector):
            name = "climbing_pose"

            def load(self):
                raise FileNotFoundError("yolo11n-pose.pt")

        src = FakeSource(10)
        shutdown = mock.Mock(wraps=self.events.shutdown)
        self.events.shutdown = shutdown
        with self.assertRaises(ModuleLoadError) as ctx:
            Runner(src, [Module(NoWeights(), LabelMatchPolicy("x", ["none"]))],
                   self.events, display=False, clock=Clock()).run()
        self.assertIn("climbing_pose", str(ctx.exception))
        self.assertTrue(src.released)          # camera handed back
        shutdown.assert_called_once()          # event system closed

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

    def live_source(self, raw, *captures, **kw):
        """A webcam/stream source whose successive (re)opened captures are `captures`."""
        caps = iter(captures)
        with mock.patch.object(CameraSource, "_open_capture", side_effect=lambda: next(caps)):
            src = CameraSource(raw, retry_delay_s=0.0, **kw)
            src.read_result = src.read()
        return src

    def test_live_source_survives_a_failed_read(self):
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        src = self.live_source("0", FakeCapture([None, None, frame]))
        self.assertIsNotNone(src.read_result)  # two hiccups do not end the run

    def test_live_source_gives_up_after_many_failures(self):
        src = self.live_source("0", *[FakeCapture([]) for _ in range(5)], max_read_failures=5)
        self.assertIsNone(src.read_result)

    def test_dead_capture_is_reopened(self):
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        src = self.live_source("rtsp://cam/1", FakeCapture([]), FakeCapture([frame]))
        self.assertIsNotNone(src.read_result)  # recovered on the reopened capture


class FakeCapture:
    """cv2.VideoCapture stand-in: None in `reads` = a failed read."""

    def __init__(self, reads):
        self.reads = list(reads)

    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def read(self):
        frame = self.reads.pop(0) if self.reads else None
        return frame is not None, frame

    def release(self):
        pass


if __name__ == "__main__":
    unittest.main()
