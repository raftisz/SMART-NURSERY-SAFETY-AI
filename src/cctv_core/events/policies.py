"""Reusable EventPolicy implementations."""

from __future__ import annotations

from typing import Any

from ..detector_base import EventPolicy
from ..schemas import BBox, Detection, DetectorOutput, EventCandidate


def boxes_overlap(a: BBox, b: BBox) -> bool:
    """True when the boxes share some area (touching edges do not count)."""
    return min(a.x2, b.x2) > max(a.x1, b.x1) and min(a.y2, b.y2) > max(a.y1, b.y1)


class LabelMatchPolicy(EventPolicy):
    """One candidate per frame when any of `labels` is seen at >= min_confidence.

    The highest-confidence matching detection is attached as evidence.
    Matching is case-insensitive on the model's own class names.

    `ignore_if_overlapping` (off unless enabled): a detection whose label is in
    `labels` is not counted when its box overlaps a box with a label in `with`
    reported by another module on the same frame, e.g.
        {enabled: true, labels: [cigarette], with: [scissors, knife]}
    The other module must run earlier in config order; if it did not run on this
    frame, nothing is ignored.
    """

    def __init__(self, event_type: str, labels: list[str], min_confidence: float = 0.0,
                 ignore_if_overlapping: dict[str, Any] | None = None):
        if not labels:
            raise ValueError("labels must not be empty")
        self.event_type = event_type
        self.labels = {label.lower() for label in labels}
        self.min_confidence = float(min_confidence)
        opts = dict(ignore_if_overlapping or {})
        self.ignore_if_overlapping = {
            "enabled": bool(opts.get("enabled", False)),
            "labels": [str(x).lower() for x in opts.get("labels") or []],
            "with": [str(x).lower() for x in opts.get("with") or []],
        }

    def _ignored(self, det: Detection, output: DetectorOutput) -> bool:
        rule = self.ignore_if_overlapping
        if not rule["enabled"] or det.label.lower() not in rule["labels"]:
            return False
        return any(o.label.lower() in rule["with"] and boxes_overlap(det.bbox, o.bbox)
                   for other in output.context.values() for o in other.detections)

    def evaluate(self, output: DetectorOutput) -> list[EventCandidate]:
        matches = [
            d for d in output.detections
            if d.label.lower() in self.labels and d.confidence >= self.min_confidence
            and not self._ignored(d, output)
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
