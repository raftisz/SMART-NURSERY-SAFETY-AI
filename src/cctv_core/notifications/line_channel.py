"""LINE channel: formats an Event and sends it through the existing LineNotifier.

Only LineNotifier.send_alert() is used. Its maybe_alert() (class filter and
cooldown) is not, because those decisions now belong to the Event Manager.
src/alerts/line_notifier.py itself is not modified.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..schemas import DeliveryRecord, Event
from .base import NotificationChannel
from .templates import format_message


class LineChannel(NotificationChannel):
    name = "line"

    def __init__(
        self,
        camera_names: dict[str, str] | None = None,
        notifier: Any = None,
        env_path: str | Path | None = None,
        timeout: float = 5.0,
    ):
        if notifier is None:
            from src.alerts.line_notifier import LineNotifier

            notifier = LineNotifier(env_path=Path(env_path) if env_path else None, timeout=timeout)
        self.notifier = notifier
        self.camera_names = camera_names or {}

    @property
    def configured(self) -> bool:
        return bool(getattr(self.notifier, "configured", False))

    def send(self, event: Event) -> DeliveryRecord:
        now = time.time()
        if not self.configured:
            return DeliveryRecord(self.name, now, False, "LINE credentials missing in .env")
        ok = self.notifier.send_alert(
            format_message(event, self.camera_names), image_path=event.snapshot_path
        )
        # LineNotifier prints the exact failure reason; it only returns a bool.
        return DeliveryRecord(self.name, now, bool(ok), None if ok else "send failed (see [LINE] log)")
