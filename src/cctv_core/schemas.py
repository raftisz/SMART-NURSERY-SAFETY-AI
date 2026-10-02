"""Shared data types for every detector, event and notification.

These are the contract between modules. A new detector only has to produce
`DetectorOutput`; everything downstream works on these types alone.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1.1"  # 1.1: optional subject (e.g. zone name); frame size on DetectorOutput


# --------------------------------------------------------------------------
# Frames and detections
# --------------------------------------------------------------------------


@dataclass
class Frame:
    """One image from one camera. `image` is a BGR numpy array (or None in tests)."""

    camera_id: str
    frame_index: int
    timestamp: float  # seconds (time.time())
    image: Any = None
    width: int = 0
    height: int = 0


@dataclass
class BBox:
    """Pixel box in the coordinates of the frame given to the detector."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class Detection:
    """One thing a detector saw in one frame.

    `label` is the model's class name exactly as the model reports it.
    """

    label: str
    confidence: float
    bbox: BBox
    keypoints: list[Keypoint] | None = None
    track_id: int | None = None  # reserved for future tracking
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, det: Any) -> "Detection":
        """Convert src.detection.yolo_detector.Detection (duck-typed, no import).

        The legacy class exposes class_name, confidence, x1, y1, x2, y2.
        """
        return cls(
            label=str(det.class_name),
            confidence=float(det.confidence),
            bbox=BBox(float(det.x1), float(det.y1), float(det.x2), float(det.y2)),
        )


@dataclass
class ModelInfo:
    weights: str
    sha256: str | None = None
    class_names: dict[int, str] = field(default_factory=dict)


@dataclass
class DetectorOutput:
    detector: str
    camera_id: str
    frame_index: int
    timestamp: float
    detections: list[Detection]
    inference_ms: float = 0.0
    model: ModelInfo | None = None
    frame_width: int = 0   # size of the frame the detector saw (for normalised geometry)
    frame_height: int = 0


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


class Severity(Enum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def parse(cls, value: "str | Severity") -> "Severity":
        if isinstance(value, Severity):
            return value
        return cls[str(value).strip().upper()]

    def __ge__(self, other: "Severity") -> bool:
        return self.value >= other.value

    def __gt__(self, other: "Severity") -> bool:
        return self.value > other.value

    def __le__(self, other: "Severity") -> bool:
        return self.value <= other.value

    def __lt__(self, other: "Severity") -> bool:
        return self.value < other.value


class EventStatus(Enum):
    CANDIDATE = "CANDIDATE"    # seen, not yet confirmed
    CONFIRMED = "CONFIRMED"    # confirmed and sent to notification
    SUPPRESSED = "SUPPRESSED"  # confirmed, but inside cooldown: stored, not notified
    CLOSED = "CLOSED"          # no longer seen


@dataclass
class EventCandidate:
    """What an EventPolicy produces from one DetectorOutput."""

    event_type: str
    camera_id: str
    source_detector: str
    timestamp: float
    confidence: float
    detection: Detection | None = None
    model: ModelInfo | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    #: optional sub-key, e.g. a zone name. Events are tracked per
    #: (camera_id, event_type, subject), so two zones never merge.
    subject: str | None = None


@dataclass
class DeliveryRecord:
    channel: str
    sent_at: float
    success: bool
    error: str | None = None


@dataclass
class Event:
    event_type: str
    camera_id: str
    source_detector: str
    severity: Severity
    started_at: float
    last_seen_at: float
    status: EventStatus = EventStatus.CANDIDATE
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    schema_version: str = SCHEMA_VERSION
    closed_at: float | None = None
    confirmed_at: float | None = None
    hit_count: int = 0
    peak_confidence: float = 0.0
    peak_detection: Detection | None = None
    model: ModelInfo | None = None
    snapshot_path: str | None = None
    notifications: list[DeliveryRecord] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    subject: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (enums become their names)."""
        data = asdict(self)
        data["severity"] = self.severity.name
        data["status"] = self.status.value
        return data
