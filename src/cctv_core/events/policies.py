"""Reusable EventPolicy implementations."""

from __future__ import annotations

from collections import deque
from typing import Any

from ..detector_base import EventPolicy
from ..schemas import BBox, Detection, DetectorOutput, EventCandidate


def boxes_overlap(a: BBox, b: BBox) -> bool:
    """True when the boxes share some area (touching edges do not count)."""
    return min(a.x2, b.x2) > max(a.x1, b.x1) and min(a.y2, b.y2) > max(a.y1, b.y1)


def center_shift(a: BBox, b: BBox) -> float:
    """Centre distance between two boxes, in units of the larger box diagonal."""
    ax, ay = (a.x1 + a.x2) / 2, (a.y1 + a.y2) / 2
    bx, by = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
    diag = max((a.width ** 2 + a.height ** 2) ** 0.5, (b.width ** 2 + b.height ** 2) ** 0.5, 1.0)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / diag


class LabelMatchPolicy(EventPolicy):
    """One candidate per frame when any of `labels` is seen at >= min_confidence.

    The highest-confidence matching detection is attached as evidence.
    Matching is case-insensitive on the model's own class names.

    `per_label_confidence` (optional): a threshold per label that overrides
    `min_confidence`, e.g. {knife: 0.35, scissors: 0.30}. Classes are not equally
    hard, so one threshold for all of them trades one class's recall for another's
    false alarms. Pick the values on a validation set (tools/eval_hazard_model.py
    prints them), never on the test set.

    `persistence` (off unless enabled): a detection only counts when the same
    label was also seen at about the same place in the previous frames, e.g.
        {enabled: true, min_frames: 2, window_s: 1.0, max_shift: 1.5}
    = seen in >= 2 frames within 1 s, centre moved <= 1.5 box diagonals per step.
    The Event Manager's confirm_hits counts sightings anywhere in the picture, so
    three one-frame false boxes on three different objects can still confirm an
    event; a real object stays put (or moves smoothly), flicker does not.

    `ignore_if_overlapping` (off unless enabled): a detection whose label is in
    `labels` is not counted when its box overlaps a box with a label in `with`
    reported by another module on the same frame, e.g.
        {enabled: true, labels: [cigarette], with: [scissors, knife]}
    The other module must run earlier in config order; if it did not run on this
    frame, nothing is ignored.
    """

    def __init__(self, event_type: str, labels: list[str], min_confidence: float = 0.0,
                 ignore_if_overlapping: dict[str, Any] | None = None,
                 per_label_confidence: dict[str, float] | None = None,
                 persistence: dict[str, Any] | None = None):
        if not labels:
            raise ValueError("labels must not be empty")
        self.event_type = event_type
        self.labels = {label.lower() for label in labels}
        self.min_confidence = float(min_confidence)
        self.per_label_confidence = {str(k).lower(): float(v)
                                     for k, v in (per_label_confidence or {}).items()}
        unknown = set(self.per_label_confidence) - self.labels
        if unknown:
            raise ValueError(f"per_label_confidence for labels not in labels: {sorted(unknown)}")
        opts = dict(ignore_if_overlapping or {})
        self.ignore_if_overlapping = {
            "enabled": bool(opts.get("enabled", False)),
            "labels": [str(x).lower() for x in opts.get("labels") or []],
            "with": [str(x).lower() for x in opts.get("with") or []],
        }
        p = dict(persistence or {})
        self.persistence = {
            "enabled": bool(p.get("enabled", False)),
            "min_frames": int(p.get("min_frames", 2)),
            "window_s": float(p.get("window_s", 1.0)),
            "max_shift": float(p.get("max_shift", 1.5)),
        }
        if self.persistence["min_frames"] < 1:
            raise ValueError("persistence.min_frames must be >= 1")
        # camera -> list of (timestamp, [(label, bbox), ...]) of passing detections
        self._history: dict[str, deque] = {}

    def threshold(self, label: str) -> float:
        return self.per_label_confidence.get(label.lower(), self.min_confidence)

    def _persistent(self, det: Detection, frames: list[tuple[float, list[tuple[str, BBox]]]]) -> bool:
        """True when `det` can be chained back through min_frames-1 earlier frames."""
        need = self.persistence["min_frames"] - 1
        if need <= 0:
            return True
        label, box, chained = det.label.lower(), det.bbox, 0
        for _, seen in reversed(frames):
            near = [b for lab, b in seen if lab == label
                    and center_shift(b, box) <= self.persistence["max_shift"]]
            if not near:
                continue  # a missed frame is allowed inside the window
            box = min(near, key=lambda b: center_shift(b, box))
            chained += 1
            if chained >= need:
                return True
        return False

    def _ignored(self, det: Detection, output: DetectorOutput) -> bool:
        rule = self.ignore_if_overlapping
        if not rule["enabled"] or det.label.lower() not in rule["labels"]:
            return False
        return any(o.label.lower() in rule["with"] and boxes_overlap(det.bbox, o.bbox)
                   for other in output.context.values() for o in other.detections)

    def evaluate(self, output: DetectorOutput) -> list[EventCandidate]:
        matches = [
            d for d in output.detections
            if d.label.lower() in self.labels and d.confidence >= self.threshold(d.label)
            and not self._ignored(d, output)
        ]
        if self.persistence["enabled"]:
            hist = self._history.setdefault(output.camera_id, deque())
            t = output.timestamp
            while hist and t - hist[0][0] > self.persistence["window_s"]:
                hist.popleft()
            frames = list(hist)
            hist.append((t, [(d.label.lower(), d.bbox) for d in matches]))
            matches = [d for d in matches if self._persistent(d, frames)]
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
