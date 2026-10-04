"""Read-only report of data/events.db: what was seen, what became an event, what was sent.

Every stored event was confirmed (unconfirmed candidates are never stored); its
final status is CLOSED, so "was an alert actually sent?" is answered from the
delivery records the router keeps in data_json:
    SENT         at least one channel delivered it
    SEND FAILED  delivery was attempted and every channel failed
    NOT SENT     no delivery record:
                   reason "cooldown" - the same camera/type/zone was alerted within cooldown_s
                   reason "unknown"  - e.g. the program stopped before the alert was delivered
The database is opened read-only; this tool never changes it.

Usage (from the repo root):
    python tools/events_report.py                       # today
    python tools/events_report.py --since 2026-10-04 --type smoking
    python tools/events_report.py --since 2026-10-04 --md docs/eval/events_2026-10-04.md --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cctv_core.events.manager import EventTypeConfig  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402


@dataclass
class Row:
    event_id: str
    event_type: str
    camera_id: str
    subject: str | None
    started_at: float
    confirmed_at: float | None
    last_seen_at: float
    hits: int
    peak: float
    label: str | None
    channels: list[str]
    outcome: str = ""
    reason: str = ""
    snapshot: str | None = None

    @property
    def key(self) -> tuple:
        return (self.camera_id, self.event_type, self.subject)


def connect_read_only(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def cooldowns_from(config: str | Path = "config/core.yaml") -> tuple[dict[str, float], float]:
    ev = load_config(resolve_path(config)).get("events") or {}
    default = EventTypeConfig.from_dict(ev.get("default") or {})
    types = {k: EventTypeConfig.from_dict(v or {}, base=default).cooldown_s
             for k, v in (ev.get("types") or {}).items()}
    return types, default.cooldown_s


def load_events(db: Path, since: datetime | None = None, until: datetime | None = None,
                event_type: str | None = None, config: str | Path = "config/core.yaml") -> list[Row]:
    conn = connect_read_only(db)
    try:
        records = conn.execute("SELECT * FROM events ORDER BY started_at").fetchall()
    finally:
        conn.close()
    lo = since.timestamp() if since else float("-inf")
    hi = until.timestamp() if until else float("inf")

    rows: list[Row] = []
    for r in records:
        if not (lo <= r["started_at"] < hi):
            continue
        if event_type and r["event_type"] != event_type:
            continue
        data = json.loads(r["data_json"])
        deliveries = data.get("notifications") or []
        peak = data.get("peak_detection") or {}
        row = Row(r["event_id"], r["event_type"], r["camera_id"], data.get("subject"),
                  r["started_at"], r["confirmed_at"], r["last_seen_at"], r["hit_count"],
                  r["peak_confidence"], peak.get("label"),
                  [d["channel"] for d in deliveries if d.get("success")], snapshot=r["snapshot_path"])
        if row.channels:
            row.outcome, row.reason = "SENT", ",".join(row.channels)
        elif deliveries:
            row.outcome = "SEND FAILED"
            row.reason = "; ".join(f"{d['channel']}: {d.get('error') or 'failed'}" for d in deliveries)
        else:
            row.outcome = "NOT SENT"
        rows.append(row)
    _explain_not_sent(rows, *cooldowns_from(config))
    return rows


def _explain_not_sent(rows: list[Row], cooldowns: dict[str, float], default_cooldown: float) -> None:
    """A NOT SENT event is 'cooldown' when the same key was dispatched within cooldown_s before it."""
    dispatched: dict[tuple, float] = {}
    for row in sorted(rows, key=lambda r: r.confirmed_at or r.started_at):
        t = row.confirmed_at or row.started_at
        if row.outcome in ("SENT", "SEND FAILED"):
            dispatched[row.key] = t
            continue
        last = dispatched.get(row.key)
        cooldown = cooldowns.get(row.event_type, default_cooldown)
        row.reason = "cooldown" if last is not None and 0 <= t - last < cooldown else "unknown"


@dataclass
class TypeSummary:
    sightings: int = 0
    events: int = 0
    sent: int = 0
    failed: int = 0
    not_sent: int = 0


def summarise(rows: list[Row]) -> dict[str, TypeSummary]:
    out: dict[str, TypeSummary] = {}
    for r in rows:
        s = out.setdefault(r.event_type, TypeSummary())
        s.sightings += r.hits
        s.events += 1
        s.sent += r.outcome == "SENT"
        s.failed += r.outcome == "SEND FAILED"
        s.not_sent += r.outcome == "NOT SENT"
    return dict(sorted(out.items()))


def funnel_line(summary: dict[str, TypeSummary]) -> str:
    seen = sum(s.sightings for s in summary.values())
    events = sum(s.events for s in summary.values())
    sent = sum(s.sent for s in summary.values())
    return f"seen {seen:,} times -> {events} events -> {sent} alerts sent"


def _t(ts: float | None) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "-"


def write_csv(rows: list[Row], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["started", "confirmed", "event_type", "subject", "seconds_seen", "sightings",
                    "peak", "peak_label", "outcome", "reason", "snapshot"])
        for r in rows:
            w.writerow([_t(r.started_at), _t(r.confirmed_at), r.event_type, r.subject or "",
                        f"{r.last_seen_at - r.started_at:.1f}", r.hits, f"{r.peak:.2f}", r.label or "",
                        r.outcome, r.reason, Path(r.snapshot).name if r.snapshot else ""])


def write_markdown(rows: list[Row], summary: dict[str, TypeSummary], path: Path, title: str) -> None:
    lines = [f"# Event log: {title}", "", f"**{funnel_line(summary)}**", "",
             "| event type | sightings | events | sent | send failed | not sent |", "|---|---|---|---|---|---|"]
    lines += [f"| {t} | {s.sightings:,} | {s.events} | {s.sent} | {s.failed} | {s.not_sent} |"
              for t, s in summary.items()]
    lines += ["", "| started | type | zone | seen (s) | sightings | peak | outcome | snapshot |",
              "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {_t(r.started_at)} | {r.event_type} | {r.subject or ''} | "
              f"{r.last_seen_at - r.started_at:.1f} | {r.hits} | {r.peak:.2f} {r.label or ''} | "
              f"{r.outcome}{' (' + r.reason + ')' if r.reason else ''} | "
              f"{Path(r.snapshot).name if r.snapshot else ''} |" for r in rows]
    lines += ["", "Read-only report of data/events.db (tools/events_report.py). Sightings are frames in "
              "which the rule matched; they are not accuracy figures.", ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only report of the event log")
    ap.add_argument("--db", default="data/events.db")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--until", default=None, help="YYYY-MM-DD, exclusive")
    ap.add_argument("--type", default=None, help="only this event type")
    ap.add_argument("--config", default="config/core.yaml", help="for cooldown_s")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    db = resolve_path(args.db)
    if not db.is_file():
        print(f"no database at {db}")
        return 1
    since = datetime.fromisoformat(args.since) if args.since else \
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    until = datetime.fromisoformat(args.until) if args.until else None
    rows = load_events(db, since, until, args.type, args.config)
    summary = summarise(rows)

    print(funnel_line(summary))
    for t, s in summary.items():
        print(f"  {t:<14} sightings {s.sightings:>6,}  events {s.events:>3}  sent {s.sent:>3}  "
              f"failed {s.failed:>2}  not sent {s.not_sent:>2}")
    print()
    for r in rows:
        where = f" [{r.subject}]" if r.subject else ""
        print(f"{_t(r.started_at)}  {r.event_type}{where:<10} seen {r.last_seen_at - r.started_at:5.1f}s "
              f"hits {r.hits:>4}  peak {r.peak:.2f} {r.label or '':<9} {r.outcome}"
              f"{' (' + r.reason + ')' if r.reason else ''}")
    if args.csv:
        write_csv(rows, resolve_path(args.csv))
        print(f"wrote {args.csv}")
    if args.md:
        write_markdown(rows, summary, resolve_path(args.md), f"from {since:%Y-%m-%d}")
        print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
