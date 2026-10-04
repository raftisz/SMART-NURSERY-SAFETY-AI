import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.cctv_core.events.manager import EventManager, EventTypeConfig
from src.cctv_core.events.snapshot import SnapshotWriter
from src.cctv_core.events.store import EventStore
from src.cctv_core.schemas import EventCandidate, EventStatus, Severity
from tests.fakes import RecordingDispatcher, det

CFG = EventTypeConfig(severity=Severity.HIGH, confirm_hits=3, confirm_window_s=2.0,
                      close_after_s=10.0, cooldown_s=30.0)


def cand(t, conf=0.8, cam="cam0", etype="hazard_object"):
    return EventCandidate(etype, cam, "fake", t, conf, det("scissors", conf))


class EventManagerTests(unittest.TestCase):
    def setUp(self):
        self.store = EventStore(":memory:")
        self.disp = RecordingDispatcher()
        self.mgr = EventManager({"hazard_object": CFG}, store=self.store, dispatcher=self.disp)

    def test_single_sighting_is_not_confirmed(self):
        self.assertIsNone(self.mgr.submit(cand(0.0)))
        self.assertEqual(self.disp.events, [])
        self.assertEqual(self.store.list_events(), [])

    def test_confirms_after_n_hits_in_window(self):
        self.mgr.submit(cand(0.0))
        self.mgr.submit(cand(0.5))
        ev = self.mgr.submit(cand(1.0, conf=0.95))
        self.assertIsNotNone(ev)
        self.assertIs(ev.status, EventStatus.CONFIRMED)
        self.assertIs(ev.severity, Severity.HIGH)
        self.assertEqual(ev.hit_count, 3)
        self.assertAlmostEqual(ev.peak_confidence, 0.95)
        self.assertEqual(self.disp.events, [ev])
        self.assertEqual(self.store.get(ev.event_id)["status"], "CONFIRMED")

    def test_hits_spread_beyond_window_do_not_confirm(self):
        for t in (0.0, 2.5, 5.0, 7.5):  # never 3 within 2 s
            self.assertIsNone(self.mgr.submit(cand(t)))
        self.assertEqual(self.disp.events, [])

    def test_continuing_sightings_merge_into_one_event(self):
        for i in range(20):
            self.mgr.submit(cand(i * 0.1))
        self.assertEqual(len(self.disp.events), 1)
        self.assertEqual(self.disp.events[0].hit_count, 20)

    def test_close_after_gap(self):
        for t in (0.0, 0.5, 1.0):
            self.mgr.submit(cand(t))
        self.assertEqual(self.mgr.tick(5.0), [])
        closed = self.mgr.tick(11.0)
        self.assertEqual(len(closed), 1)
        self.assertIs(closed[0].status, EventStatus.CLOSED)
        self.assertEqual(self.store.get(closed[0].event_id)["status"], "CLOSED")

    def test_unconfirmed_candidate_is_discarded(self):
        self.mgr.submit(cand(0.0))
        self.assertEqual(self.mgr.tick(20.0), [])
        self.assertEqual(self.mgr.open_events, [])
        self.assertEqual(self.store.list_events(), [])

    def test_cooldown_suppresses_second_event(self):
        for t in (0.0, 0.5, 1.0):
            self.mgr.submit(cand(t))
        self.mgr.tick(12.0)                      # first event closes
        for t in (15.0, 15.5, 16.0):             # new event, still inside 30 s cooldown
            ev = self.mgr.submit(cand(t))
        self.assertIs(ev.status, EventStatus.SUPPRESSED)
        self.assertEqual(len(self.disp.events), 1)
        self.assertEqual(len(self.store.list_events()), 2)

    def test_after_cooldown_notifies_again(self):
        for t in (0.0, 0.5, 1.0):
            self.mgr.submit(cand(t))
        self.mgr.tick(12.0)
        for t in (40.0, 40.5, 41.0):
            ev = self.mgr.submit(cand(t))
        self.assertIs(ev.status, EventStatus.CONFIRMED)
        self.assertEqual(len(self.disp.events), 2)

    def test_suppressed_event_notifies_once_cooldown_ends(self):
        for t in (0.0, 0.5, 1.0):
            self.mgr.submit(cand(t))
        self.mgr.tick(12.0)                      # first event closes
        for t in (15.0, 15.5, 16.0):             # second event starts inside the cooldown
            self.mgr.submit(cand(t))
        promoted = None
        t = 16.5
        while t <= 60.0:                         # ...and is still seen after the cooldown ends
            ev = self.mgr.submit(cand(t))
            if ev is not None and promoted is None:
                promoted = (t, ev)
            t += 0.5
        self.assertIsNotNone(promoted)
        at, ev = promoted
        self.assertGreaterEqual(at, 30.0)        # not before the cooldown is over
        self.assertIs(ev.status, EventStatus.CONFIRMED)
        self.assertEqual(len(self.disp.events), 2)  # once, not on every later sighting
        self.assertEqual(self.store.get(ev.event_id)["status"], "CONFIRMED")

    def test_cameras_are_independent(self):
        for t in (0.0, 0.5, 1.0):
            self.mgr.submit(cand(t, cam="cam0"))
            self.mgr.submit(cand(t, cam="cam1"))
        self.assertEqual(sorted(e.camera_id for e in self.disp.events), ["cam0", "cam1"])

    def test_event_types_are_independent(self):
        for t in (0.0, 0.5, 1.0):
            self.mgr.submit(cand(t, etype="hazard_object"))
            self.mgr.submit(cand(t, etype="smoking"))
        types = {e.event_type: e for e in self.disp.events}
        self.assertEqual(set(types), {"hazard_object", "smoking"})
        self.assertIs(types["smoking"].severity, Severity.MEDIUM)  # default config

    def test_snapshot_saved_only_with_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = EventManager({"hazard_object": CFG}, snapshots=SnapshotWriter(tmp))
            img = np.zeros((48, 64, 3), dtype=np.uint8)
            for t in (0.0, 0.5):
                mgr.submit(cand(t), image=img)
            ev = mgr.submit(cand(1.0), image=img)
            self.assertTrue(ev.snapshot_path and Path(ev.snapshot_path).is_file())

            mgr2 = EventManager({"hazard_object": CFG}, snapshots=SnapshotWriter(tmp))
            for t in (0.0, 0.5):
                mgr2.submit(cand(t))
            self.assertIsNone(mgr2.submit(cand(1.0)).snapshot_path)

    def test_close_all(self):
        for t in (0.0, 0.5, 1.0):
            self.mgr.submit(cand(t))
        self.assertEqual(len(self.mgr.close_all(2.0)), 1)
        self.assertEqual(self.mgr.open_events, [])


if __name__ == "__main__":
    unittest.main()
