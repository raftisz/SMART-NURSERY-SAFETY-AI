"""Print alerts to the terminal. The safe default while developing."""

from __future__ import annotations

import sys
import time
from typing import TextIO

from ..schemas import DeliveryRecord, Event
from .base import NotificationChannel
from .templates import format_message


class ConsoleChannel(NotificationChannel):
    name = "console"

    def __init__(self, camera_names: dict[str, str] | None = None, stream: TextIO | None = None):
        self.camera_names = camera_names or {}
        self.stream = stream

    def send(self, event: Event) -> DeliveryRecord:
        out = self.stream or sys.stdout
        print("-" * 40, file=out)
        print(format_message(event, self.camera_names), file=out)
        if event.snapshot_path:
            print(f"(snapshot: {event.snapshot_path})", file=out)
        print("-" * 40, file=out, flush=True)
        return DeliveryRecord(channel=self.name, sent_at=time.time(), success=True)
