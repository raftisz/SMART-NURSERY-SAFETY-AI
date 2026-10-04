import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.cctv_core.events.store import EventStore
from src.cctv_core.schemas import DeliveryRecord, Event, EventStatus, Severity

sys.path.insert(0, str(Path("tools").resolve()))
import events_report  # noqa: E402

T0 = datetime(2026, 10, 4, 13, 57, 0).timestamp()


def event(etype, start, confirmed, hits, sent=None, subject=None):
    e = Event(etype, "cam0", "fake", Severity.HIGH, start, start + 5, status=EventStatus.CLOSED,
              confirmed_at=confirmed, hit_count=hits, peak_confidence=0.8, subject=subject,
              closed_at=start + 20)
    if sent is not None:
        e.notifications = [DeliveryRecord("console", confirmed, True)] if sent == "ok" else \
                          [DeliveryRecord("line", confirmed, False, "boom")]
    return e


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "events.db"
        store = EventStore(self.db)
        for e in (
            event("smoking", T0, T0 + 2, 40, sent="ok"),
            event("smoking", T0 + 25, T0 + 27, 10),                 # 25 s after a sent alert -> cooldown
            event("hazard_object", T0, T0 + 1, 100, sent="fail"),
            event("out_of_area", T0 + 60, T0 + 65, 300, subject="door"),  # nothing sent before -> unknown
            event("smoking", T0 - 86400, T0 - 86400 + 2, 5, sent="ok"),   # yesterday
        ):
            store.save(e)
        store.close()
        self.rows = events_report.load_events(self.db, since=datetime(2026, 10, 4))

    def test_reads_only_the_requested_period(self):
        self.assertEqual(len(self.rows), 4)

    def test_outcomes(self):
        out = {(r.event_type, round(r.started_at - T0)): (r.outcome, r.reason) for r in self.rows}
        self.assertEqual(out[("smoking", 0)], ("SENT", "console"))
        self.assertEqual(out[("smoking", 25)], ("NOT SENT", "cooldown"))
        self.assertEqual(out[("hazard_object", 0)][0], "SEND FAILED")
        self.assertEqual(out[("out_of_area", 60)], ("NOT SENT", "unknown"))

    def test_summary_funnel(self):
        s = events_report.summarise(self.rows)
        self.assertEqual(s["smoking"].sightings, 50)
        self.assertEqual((s["smoking"].events, s["smoking"].sent, s["smoking"].not_sent), (2, 1, 1))
        total = events_report.funnel_line(s)
        self.assertIn("450", total)                       # 40 + 10 + 100 + 300 sightings
        self.assertIn("4 events", total)
        self.assertIn("1 alerts sent", total)

    def test_database_is_opened_read_only(self):
        conn = events_report.connect_read_only(self.db)
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("DELETE FROM events")
        conn.close()

    def test_writes_csv_and_markdown(self):
        csv_path = Path(self.tmp.name) / "r.csv"
        md_path = Path(self.tmp.name) / "r.md"
        events_report.write_csv(self.rows, csv_path)
        events_report.write_markdown(self.rows, events_report.summarise(self.rows), md_path, "test")
        self.assertEqual(len(csv_path.read_text().splitlines()), 5)   # header + 4
        self.assertIn("NOT SENT", md_path.read_text())


if __name__ == "__main__":
    unittest.main()
