import io
import unittest

from src.cctv_core.events.store import EventStore
from src.cctv_core.notifications.console_channel import ConsoleChannel
from src.cctv_core.notifications.line_channel import LineChannel
from src.cctv_core.notifications.router import NotificationRouter, Route
from src.cctv_core.notifications.templates import format_message
from src.cctv_core.schemas import Event, EventStatus, Severity
from tests.fakes import FakeLineNotifier, RecordingChannel, det


def event(etype="hazard_object", sev=Severity.HIGH, snapshot=None):
    e = Event(etype, "cam0", "fake", sev, 1_700_000_000.0, 1_700_000_001.0,
              status=EventStatus.CONFIRMED, confirmed_at=1_700_000_001.0,
              peak_confidence=0.87, peak_detection=det("scissors", 0.87))
    e.snapshot_path = snapshot
    return e


class TemplateTests(unittest.TestCase):
    def test_message_content(self):
        msg = format_message(event(), {"cam0": "Notebook Camera"})
        self.assertIn("🚨 AI CCTV ALERT 🚨", msg)
        self.assertIn("ตรวจพบวัตถุอันตราย", msg)
        self.assertIn("🎯 ประเภท: scissors", msg)
        self.assertIn("0.87", msg)
        self.assertIn("📹 กล้อง: Notebook Camera", msg)

    def test_evidence_line_only_when_saved(self):
        self.assertNotIn("บันทึกภาพ", format_message(event()))
        self.assertIn("บันทึกภาพ", format_message(event(snapshot="data/snapshots/a.jpg")))

    def test_unknown_type_has_fallback_title(self):
        self.assertIn("ตรวจพบเหตุการณ์: new_module", format_message(event(etype="new_module")))


class RouterTests(unittest.TestCase):
    def test_routes_by_severity_and_type(self):
        a, b = RecordingChannel("a"), RecordingChannel("b")
        router = NotificationRouter(
            {"a": a, "b": b},
            [Route(["a"]), Route(["b"], {"smoking"}, Severity.HIGH)],
            async_mode=False,
        )
        router.dispatch(event("smoking", Severity.MEDIUM))
        router.dispatch(event("smoking", Severity.HIGH))
        router.dispatch(event("hazard_object", Severity.CRITICAL))
        self.assertEqual(len(a.sent), 3)
        self.assertEqual(len(b.sent), 1)

    def test_channel_listed_once_even_if_two_routes_match(self):
        a = RecordingChannel("a")
        router = NotificationRouter({"a": a}, [Route(["a"]), Route(["a"])], async_mode=False)
        router.dispatch(event())
        self.assertEqual(len(a.sent), 1)

    def test_disabled_channel_in_route_is_skipped(self):
        a = RecordingChannel("a")
        router = NotificationRouter({"a": a}, [Route(["a", "line"])], async_mode=False)
        self.assertEqual(router.targets_for(event()), ["a"])

    def test_broken_channel_does_not_stop_others(self):
        bad, good = RecordingChannel("bad", explode=True), RecordingChannel("good")
        router = NotificationRouter({"bad": bad, "good": good}, [Route(["bad", "good"])],
                                    async_mode=False)
        e = event()
        router.dispatch(e)
        self.assertEqual(len(good.sent), 1)
        self.assertEqual([r.success for r in e.notifications], [False, True])

    def test_async_delivery_and_store_record(self):
        store = EventStore(":memory:")
        a = RecordingChannel("a")
        router = NotificationRouter({"a": a}, [Route(["a"])], store=store, async_mode=True)
        e = event()
        router.dispatch(e)
        self.assertTrue(router.wait_idle(2.0))
        router.close()
        self.assertEqual(store.get(e.event_id)["notifications"][0]["channel"], "a")

    def test_route_from_config(self):
        r = Route.from_dict({"channels": ["line"], "event_types": "*", "min_severity": "medium"})
        self.assertIsNone(r.event_types)
        self.assertIs(r.min_severity, Severity.MEDIUM)


class ChannelTests(unittest.TestCase):
    def test_console_prints(self):
        buf = io.StringIO()
        rec = ConsoleChannel(stream=buf).send(event())
        self.assertTrue(rec.success)
        self.assertIn("AI CCTV ALERT", buf.getvalue())

    def test_line_uses_send_alert_with_snapshot(self):
        fake = FakeLineNotifier()
        rec = LineChannel(notifier=fake).send(event(snapshot="data/snapshots/x.jpg"))
        self.assertTrue(rec.success)
        self.assertEqual(fake.calls[0][1], "data/snapshots/x.jpg")

    def test_line_without_credentials_does_not_send(self):
        fake = FakeLineNotifier(configured=False)
        rec = LineChannel(notifier=fake).send(event())
        self.assertFalse(rec.success)
        self.assertEqual(fake.calls, [])

    def test_line_failure_recorded(self):
        rec = LineChannel(notifier=FakeLineNotifier(ok=False)).send(event())
        self.assertFalse(rec.success)


if __name__ == "__main__":
    unittest.main()
