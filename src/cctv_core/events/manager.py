"""Central Event Manager.

Every EventCandidate goes through here. The manager decides when repeated
sightings become one confirmed event, stores it, saves the evidence frame,
and hands it to the notification dispatcher. Nothing else sends alerts.

Flow per (camera_id, event_type, subject):
    candidate -> open event (CANDIDATE)
              -> >= confirm_hits sightings within confirm_window_s
                 (and, if min_duration_s > 0, seen continuously for that long;
                  a gap longer than max_gap_s restarts the count)
                 -> CONFIRMED (notify)  or  SUPPRESSED (inside cooldown, stored only)
              -> a SUPPRESSED event still seen after the cooldown -> CONFIRMED (notify)
              -> not seen for close_after_s -> CLOSED
    A CANDIDATE that times out without confirming is discarded.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

from ..schemas import Event, EventCandidate, EventStatus, Severity


@dataclass
class EventTypeConfig:
    severity: Severity = Severity.MEDIUM
    confirm_hits: int = 3
    confirm_window_s: float = 2.0
    close_after_s: float = 10.0
    cooldown_s: float = 30.0
    min_duration_s: float = 0.0  # 0 = off (original behaviour)
    max_gap_s: float = 1.5       # only used when min_duration_s > 0

    @classmethod
    def from_dict(cls, data: dict[str, Any], base: "EventTypeConfig | None" = None) -> "EventTypeConfig":
        base = base or cls()
        cfg = cls(
            severity=Severity.parse(data.get("severity", base.severity)),
            confirm_hits=int(data.get("confirm_hits", base.confirm_hits)),
            confirm_window_s=float(data.get("confirm_window_s", base.confirm_window_s)),
            close_after_s=float(data.get("close_after_s", base.close_after_s)),
            cooldown_s=float(data.get("cooldown_s", base.cooldown_s)),
            min_duration_s=float(data.get("min_duration_s", base.min_duration_s)),
            max_gap_s=float(data.get("max_gap_s", base.max_gap_s)),
        )
        if cfg.confirm_hits < 1:
            raise ValueError("confirm_hits must be >= 1")
        return cfg


class Dispatcher(Protocol):
    def dispatch(self, event: Event) -> None: ...


class EventManager:
    def __init__(
        self,
        type_configs: dict[str, EventTypeConfig] | None = None,
        default: EventTypeConfig | None = None,
        store: Any = None,
        snapshots: Any = None,
        dispatcher: Dispatcher | None = None,
    ):
        self.type_configs = type_configs or {}
        self.default = default or EventTypeConfig()
        self.store = store
        self.snapshots = snapshots
        self.dispatcher = dispatcher
        self._open: dict[tuple, Event] = {}
        self._hits: dict[tuple, deque[float]] = {}
        self._streak: dict[tuple, float] = {}  # start of the current continuous sighting
        self._last_notified: dict[tuple, float] = {}
        self._lock = threading.Lock()

    def config_for(self, event_type: str) -> EventTypeConfig:
        return self.type_configs.get(event_type, self.default)

    @property
    def open_events(self) -> list[Event]:
        with self._lock:
            return list(self._open.values())

    # -- main entry --------------------------------------------------------

    def submit(self, candidate: EventCandidate, image: Any = None) -> Event | None:
        """Feed one candidate. Returns the event if it was confirmed/suppressed on this call."""
        key = (candidate.camera_id, candidate.event_type, candidate.subject)
        cfg = self.config_for(candidate.event_type)
        now = candidate.timestamp

        with self._lock:
            event = self._open.get(key)
            if event is not None and now - event.last_seen_at >= cfg.close_after_s:
                self._close_locked(key, now)
                event = None

            if event is None:
                event = Event(
                    event_type=candidate.event_type,
                    camera_id=candidate.camera_id,
                    source_detector=candidate.source_detector,
                    severity=cfg.severity,
                    started_at=now,
                    last_seen_at=now,
                    model=candidate.model,
                    subject=candidate.subject,
                )
                self._open[key] = event
                self._hits[key] = deque()
                self._streak[key] = now

            event.last_seen_at = now
            event.hit_count += 1
            if candidate.confidence >= event.peak_confidence:
                event.peak_confidence = candidate.confidence
                event.peak_detection = candidate.detection
            if candidate.extra:
                event.extra.update(candidate.extra)

            if event.status is EventStatus.SUPPRESSED:
                # Still seen after the cooldown ended: notify now instead of staying silent.
                last = self._last_notified.get(key)
                if last is not None and now - last < cfg.cooldown_s:
                    return None
                event.status = EventStatus.CONFIRMED
                event.confirmed_at = now
                self._last_notified[key] = now
            elif event.status is not EventStatus.CANDIDATE:
                return None
            else:
                if not self._confirm_locked(key, cfg, now):
                    return None
                event.confirmed_at = now
                last = self._last_notified.get(key)
                if last is not None and now - last < cfg.cooldown_s:
                    event.status = EventStatus.SUPPRESSED
                else:
                    event.status = EventStatus.CONFIRMED
                    self._last_notified[key] = now

        # Outside the lock: file and network work must not block other submits.
        if self.snapshots is not None and image is not None:
            event.snapshot_path = self.snapshots.save(event, image)
        if self.store is not None:
            self.store.save(event)
        if event.status is EventStatus.CONFIRMED and self.dispatcher is not None:
            self.dispatcher.dispatch(event)
        return event

    def _confirm_locked(self, key: tuple, cfg: EventTypeConfig, now: float) -> bool:
        """Record one sighting of a CANDIDATE; True once it meets the confirmation rule."""
        hits = self._hits[key]
        if cfg.min_duration_s > 0 and hits and now - hits[-1] > cfg.max_gap_s:
            hits.clear()                 # sighting was interrupted: start again
            self._streak[key] = now
        hits.append(now)
        while hits and now - hits[0] > cfg.confirm_window_s:
            hits.popleft()
        if len(hits) < cfg.confirm_hits:
            return False
        if cfg.min_duration_s > 0 and now - self._streak[key] < cfg.min_duration_s:
            return False
        return True

    # -- closing -----------------------------------------------------------

    def tick(self, now: float) -> list[Event]:
        """Close events not seen for close_after_s. Call regularly (e.g. every frame)."""
        closed: list[Event] = []
        with self._lock:
            for key, event in list(self._open.items()):
                if now - event.last_seen_at >= self.config_for(event.event_type).close_after_s:
                    done = self._close_locked(key, now)
                    if done is not None:
                        closed.append(done)
        return closed

    def close_all(self, now: float) -> list[Event]:
        """Close everything, e.g. at shutdown."""
        closed: list[Event] = []
        with self._lock:
            for key in list(self._open):
                done = self._close_locked(key, now)
                if done is not None:
                    closed.append(done)
        return closed

    def _close_locked(self, key: tuple, now: float) -> Event | None:
        event = self._open.pop(key)
        self._hits.pop(key, None)
        self._streak.pop(key, None)
        if event.status is EventStatus.CANDIDATE:
            return None  # never confirmed: discarded, not stored
        event.status = EventStatus.CLOSED
        event.closed_at = now
        if self.store is not None:
            self.store.save(event)
        return event
