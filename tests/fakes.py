"""Test doubles. No camera, no model, no network."""

from __future__ import annotations

import time

from src.cctv_core.detector_base import BaseDetector
from src.cctv_core.notifications.base import NotificationChannel
from src.cctv_core.schemas import BBox, DeliveryRecord, Detection, DetectorOutput, Frame


def det(label: str, conf: float) -> Detection:
    return Detection(label=label, confidence=conf, bbox=BBox(0, 0, 10, 10))


class FakeDetector(BaseDetector):
    name = "fake"

    def __init__(self, labels=None, **kwargs):
        super().__init__(**kwargs)
        self.labels = labels or [("scissors", 0.9)]
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def process(self, frame: Frame) -> DetectorOutput:
        return DetectorOutput(
            detector=self.name,
            camera_id=frame.camera_id,
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
            detections=[det(label, conf) for label, conf in self.labels],
        )


class RecordingChannel(NotificationChannel):
    def __init__(self, name="rec", fail=False, explode=False):
        self.name = name
        self.fail = fail
        self.explode = explode
        self.sent = []

    def send(self, event):
        if self.explode:
            raise RuntimeError("boom")
        self.sent.append(event)
        return DeliveryRecord(self.name, time.time(), not self.fail, "fail" if self.fail else None)


class RecordingDispatcher:
    def __init__(self):
        self.events = []

    def dispatch(self, event):
        self.events.append(event)


class FakeLineNotifier:
    """Stands in for src.alerts.line_notifier.LineNotifier."""

    def __init__(self, configured=True, ok=True):
        self.configured = configured
        self.ok = ok
        self.calls = []

    def send_alert(self, message, image_path=None):
        self.calls.append((message, image_path))
        return self.ok
