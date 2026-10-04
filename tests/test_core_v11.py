"""Core additions for schema 1.1: subject, continuous-duration confirmation, multi-policy modules."""

import unittest

import numpy as np

from src.cctv_core.events.manager import EventManager, EventTypeConfig
from src.cctv_core.events.policies import LabelMatchPolicy
from src.cctv_core.notifications.templates import format_message
from src.cctv_core.runner import Module, Runner, build_modules
from src.cctv_core.schemas import EventCandidate, EventStatus
from tests.fakes import RecordingDispatcher, det, temp_event_system
from tests.test_runner import Clock, CountingDetector, FakeSource


def cand(t, subject=None, etype="out_of_area"):
    return EventCandidate(etype, "cam0", "fake", t, 0.8, det("person", 0.8), subject=subject)


class DurationTests(unittest.TestCase):
    def setUp(self):
        self.disp = RecordingDispatcher()
        cfg = EventTypeConfig(confirm_hits=3, confirm_window_s=2.0, min_duration_s=5.0,
                              max_gap_s=1.5, close_after_s=10.0)
        self.mgr = EventManager({"out_of_area": cfg}, dispatcher=self.disp)

    def test_not_confirmed_before_duration(self):
        for i in range(40):                      # 0.0 .. 3.9 s, continuous
            self.assertIsNone(self.mgr.submit(cand(i * 0.1)))

    def test_confirmed_after_continuous_duration(self):
        ev = None
        for i in range(60):                      # 0.0 .. 5.9 s
            ev = self.mgr.submit(cand(i * 0.1)) or ev
        self.assertIsNotNone(ev)
        self.assertGreaterEqual(ev.confirmed_at, 5.0)

    def test_gap_restarts_the_count(self):
        for i in range(40):                      # 0 .. 3.9 s
            self.mgr.submit(cand(i * 0.1))
        ev = None
        for i in range(40):                      # 6.0 .. 9.9 s after a 2.1 s gap
            ev = self.mgr.submit(cand(6.0 + i * 0.1)) or ev
        self.assertIsNone(ev)                    # only 3.9 s since the restart
        for i in range(15):                      # 10.0 .. 11.4 s
            ev = self.mgr.submit(cand(10.0 + i * 0.1)) or ev
        self.assertIsNotNone(ev)

    def test_default_config_keeps_old_behaviour(self):
        mgr = EventManager({}, dispatcher=RecordingDispatcher())
        ev = None
        for t in (0.0, 0.5, 1.0):
            ev = mgr.submit(cand(t, etype="hazard_object")) or ev
        self.assertIs(ev.status, EventStatus.CONFIRMED)


class SubjectTests(unittest.TestCase):
    def test_two_zones_are_two_events(self):
        disp = RecordingDispatcher()
        mgr = EventManager({}, dispatcher=disp)
        for t in (0.0, 0.5, 1.0):
            mgr.submit(cand(t, "gate"))
            mgr.submit(cand(t, "kitchen"))
        self.assertEqual(sorted(e.subject for e in disp.events), ["gate", "kitchen"])

    def test_message_shows_zone(self):
        disp = RecordingDispatcher()
        mgr = EventManager({}, dispatcher=disp)
        for t in (0.0, 0.5, 1.0):
            mgr.submit(cand(t, "main gate"))
        msg = format_message(disp.events[0])
        self.assertIn("ตรวจพบเด็กออกนอกพื้นที่", msg)
        self.assertIn("📍 โซน: main gate", msg)
        self.assertEqual(disp.events[0].to_dict()["subject"], "main gate")

    def test_no_zone_line_without_subject(self):
        disp = RecordingDispatcher()
        mgr = EventManager({}, dispatcher=disp)
        for t in (0.0, 0.5, 1.0):
            mgr.submit(cand(t, None, "hazard_object"))
        self.assertNotIn("📍", format_message(disp.events[0]))


class MultiPolicyTests(unittest.TestCase):
    def test_one_detector_feeds_two_policies(self):
        events = temp_event_system(self)
        det_ = CountingDetector(labels=[("scissors", 0.9), ("person", 0.9)])
        module = Module(det_, [LabelMatchPolicy("hazard_object", ["scissors"]),
                               LabelMatchPolicy("person_seen", ["person"])])
        stats = Runner(FakeSource(20), [module], events, display=False, clock=Clock()).run()
        self.assertEqual(det_.calls, 20)         # the model ran once per frame, not twice
        self.assertEqual(stats.confirmed, 2)

    def test_config_policies_list_and_single_policy_both_work(self):
        cfg = {"modules": [
            {"detector": {"class": "tests.fakes:FakeDetector"},
             "policies": [{"class": "src.cctv_core.events.policies:LabelMatchPolicy",
                           "params": {"event_type": "a", "labels": ["x"]}},
                          {"class": "src.cctv_core.events.policies:LabelMatchPolicy",
                           "params": {"event_type": "b", "labels": ["y"]}}]},
            {"detector": {"class": "tests.fakes:FakeDetector"},
             "policy": {"class": "src.cctv_core.events.policies:LabelMatchPolicy",
                        "params": {"event_type": "c", "labels": ["z"]}}},
        ]}
        mods = build_modules(cfg)
        self.assertEqual([len(m.policies) for m in mods], [2, 1])


if __name__ == "__main__":
    unittest.main()
