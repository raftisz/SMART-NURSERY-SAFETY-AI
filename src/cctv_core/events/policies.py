"""Reusable EventPolicy implementations."""

from __future__ import annotations

from ..detector_base import EventPolicy
from ..schemas import DetectorOutput, EventCandidate


class LabelMatchPolicy(EventPolicy):
    """One candidate per frame when any of `labels` is seen at >= min_confidence.

    The highest-confidence matching detection is attached as evidence.
    Matching is case-insensitive on the model's own class names.
    """

    def __init__(self, event_type: str, labels: list[str], min_confidence: float = 0.0):
        if not labels:
            raise ValueError("labels must not be empty")
        self.event_type = event_type
        self.labels = {label.lower() for label in labels}
        self.min_confidence = float(min_confidence)

    def evaluate(self, output: DetectorOutput) -> list[EventCandidate]:
        matches = [
            d for d in output.detections
            if d.label.lower() in self.labels and d.confidence >= self.min_confidence
        ]
        if not matches:
            return []
        top = max(matches, key=lambda d: d.confidence)
        return [
            EventCandidate(
                event_type=self.event_type,
                camera_id=output.camera_id,
                source_detector=output.detector,
                timestamp=output.timestamp,
                confidence=top.confidence,
                detection=top,
                model=output.model,
                extra={"match_count": len(matches)},
            )
        ]
