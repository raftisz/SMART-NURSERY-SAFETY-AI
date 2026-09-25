"""SQLite event log (standard library only)."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from ..schemas import Event

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    event_type      TEXT NOT NULL,
    severity        TEXT NOT NULL,
    status          TEXT NOT NULL,
    camera_id       TEXT NOT NULL,
    source_detector TEXT NOT NULL,
    started_at      REAL NOT NULL,
    last_seen_at    REAL NOT NULL,
    confirmed_at    REAL,
    closed_at       REAL,
    hit_count       INTEGER NOT NULL,
    peak_confidence REAL NOT NULL,
    snapshot_path   TEXT,
    data_json       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_started ON events(started_at);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
"""


class EventStore:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def save(self, event: Event) -> None:
        """Insert or update one event."""
        data = event.to_dict()
        row = (
            event.event_id, event.event_type, data["severity"], data["status"],
            event.camera_id, event.source_detector, event.started_at, event.last_seen_at,
            event.confirmed_at, event.closed_at, event.hit_count, event.peak_confidence,
            event.snapshot_path, json.dumps(data, ensure_ascii=False, default=str),
        )
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row
            )
            self._conn.commit()

    def get(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute("SELECT data_json FROM events WHERE event_id = ?", (event_id,))
            row = cur.fetchone()
        return json.loads(row["data_json"]) if row else None

    def list_events(self, limit: int = 100, event_type: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT data_json FROM events"
        args: tuple = ()
        if event_type:
            sql += " WHERE event_type = ?"
            args = (event_type,)
        sql += " ORDER BY started_at DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, args + (int(limit),)).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
