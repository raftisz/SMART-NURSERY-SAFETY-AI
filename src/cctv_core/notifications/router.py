"""Routes confirmed events to channels, on a background thread.

The video loop only enqueues; slow or failing channels can never stall it.
Every delivery attempt is recorded on the event and saved to the store.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..schemas import DeliveryRecord, Event, Severity
from .base import NotificationChannel


@dataclass
class Route:
    channels: list[str]
    event_types: set[str] | None = None  # None = every event type
    min_severity: Severity = Severity.INFO

    def matches(self, event: Event) -> bool:
        if self.event_types is not None and event.event_type not in self.event_types:
            return False
        return event.severity >= self.min_severity

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Route":
        types = data.get("event_types")
        if types in (None, "*", ["*"]):
            types = None
        return cls(
            channels=list(data.get("channels") or []),
            event_types=set(types) if types is not None else None,
            min_severity=Severity.parse(data.get("min_severity", "INFO")),
        )


class NotificationRouter:
    def __init__(
        self,
        channels: dict[str, NotificationChannel],
        routes: list[Route],
        store: Any = None,
        async_mode: bool = True,
    ):
        self.channels = channels
        self.store = store
        self.async_mode = async_mode
        self.routes: list[Route] = []
        for route in routes:
            known = [c for c in route.channels if c in channels]
            for missing in set(route.channels) - set(known):
                print(f"[NOTIFY] channel '{missing}' is not enabled - skipped in route")
            if known:
                self.routes.append(Route(known, route.event_types, route.min_severity))

        self._queue: "queue.Queue[Event | None]" = queue.Queue()
        self._worker: threading.Thread | None = None
        if async_mode:
            self._worker = threading.Thread(target=self._run, daemon=True, name="notify-router")
            self._worker.start()

    def targets_for(self, event: Event) -> list[str]:
        names: list[str] = []
        for route in self.routes:
            if route.matches(event):
                names += [c for c in route.channels if c not in names]
        return names

    def dispatch(self, event: Event) -> None:
        if self.async_mode:
            self._queue.put(event)
        else:
            self._deliver(event)

    def _run(self) -> None:
        while True:
            event = self._queue.get()
            try:
                if event is None:
                    return
                self._deliver(event)
            finally:
                self._queue.task_done()

    def _deliver(self, event: Event) -> None:
        for name in self.targets_for(event):
            channel = self.channels[name]
            try:
                record = channel.send(event)
            except Exception as exc:  # a broken channel must not stop the others
                record = DeliveryRecord(name, time.time(), False, f"{type(exc).__name__}: {exc}")
            event.notifications.append(record)
            status = "sent" if record.success else f"FAILED ({record.error})"
            print(f"[NOTIFY] {name}: {event.event_type} @ {event.camera_id} {status}")
        if self.store is not None:
            try:
                self.store.save(event)
            except Exception as exc:
                print(f"[NOTIFY] store error {type(exc).__name__}: {exc}")

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Block until queued events are delivered. True if drained in time."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    def close(self, timeout: float = 5.0) -> None:
        if self._worker is not None and self._worker.is_alive():
            self.wait_idle(timeout)
            self._queue.put(None)
            self._worker.join(timeout)
