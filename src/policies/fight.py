"""Fight v1: a rule over pose keypoints and simple tracking (no training).

A frame counts as "fight-like" when, for two people:
  1. they are close (boxes overlap, or the gap is small for their size),
  2. one of them moves a wrist fast (in body-heights per second), and
  3. that fast wrist is inside (or at the edge of) the OTHER person's box.
How long it must last is the Event Manager's job (min_duration_s of 'fight').

Why each part: hugging is close but slow; dancing side by side is fast but the
hands stay out of the other's box; a single high-five is fast and reaching but
brief, so the duration filter drops it.

Known limits: rough play, wrestling games and tickling look like fighting;
people hidden behind each other lose keypoints; thresholds are starting
values to be tuned on QA clips. No accuracy claim until measured.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import count

from src.cctv_core.detector_base import EventPolicy
from src.cctv_core.schemas import BBox, Detection, DetectorOutput, EventCandidate

L_WRIST, R_WRIST = 9, 10


def iou(a: BBox, b: BBox) -> float:
    iw = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    ih = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def box_gap(a: BBox, b: BBox) -> float:
    """Distance between two boxes (0 when they touch or overlap)."""
    dx = max(0.0, max(a.x1, b.x1) - min(a.x2, b.x2))
    dy = max(0.0, max(a.y1, b.y1) - min(a.y2, b.y2))
    return (dx * dx + dy * dy) ** 0.5


def inside(x: float, y: float, b: BBox, margin: float) -> bool:
    mx, my = b.width * margin, b.height * margin
    return b.x1 - mx <= x <= b.x2 + mx and b.y1 - my <= y <= b.y2 + my


@dataclass
class Track:
    track_id: int
    bbox: BBox
    last_t: float
    # (t, wrist index, x, y) for confident wrists
    wrists: deque = field(default_factory=lambda: deque(maxlen=60))


class SimpleTracker:
    """Greedy IoU matching; enough for a handful of people in one room."""

    def __init__(self, iou_min: float = 0.2, max_age_s: float = 1.0):
        self.iou_min = iou_min
        self.max_age_s = max_age_s
        self.tracks: dict[int, Track] = {}
        self._ids = count(1)

    def update(self, dets: list[Detection], t: float) -> list[Track]:
        for tid in [k for k, tr in self.tracks.items() if t - tr.last_t > self.max_age_s]:
            del self.tracks[tid]
        pairs = sorted(((iou(tr.bbox, d.bbox), tid, i) for tid, tr in self.tracks.items()
                        for i, d in enumerate(dets)), reverse=True)
        used_t, used_d, out = set(), set(), [None] * len(dets)
        for score, tid, i in pairs:
            if score < self.iou_min or tid in used_t or i in used_d:
                continue
            used_t.add(tid)
            used_d.add(i)
            out[i] = self.tracks[tid]
        for i, d in enumerate(dets):
            if out[i] is None:
                tr = Track(next(self._ids), d.bbox, t)
                self.tracks[tr.track_id] = tr
                out[i] = tr
            out[i].bbox, out[i].last_t = d.bbox, t
            d.track_id = out[i].track_id
        return out


class FightPolicy(EventPolicy):
    def __init__(
        self,
        event_type: str = "fight",
        min_confidence: float = 0.4,
        keypoint_conf: float = 0.3,
        close_gap: float = 0.15,        # max gap between boxes, x mean box height
        speed_threshold: float = 1.5,   # wrist speed, body heights per second
        speed_window_s: float = 0.5,    # speed measured over this time span
        reach_margin: float = 0.05,     # how far outside the other box still counts
    ):
        self.event_type = event_type
        self.min_confidence = float(min_confidence)
        self.keypoint_conf = float(keypoint_conf)
        self.close_gap = float(close_gap)
        self.speed_threshold = float(speed_threshold)
        self.speed_window_s = float(speed_window_s)
        self.reach_margin = float(reach_margin)
        self._trackers: dict[str, SimpleTracker] = {}

    def _record_wrists(self, tr: Track, d: Detection, t: float) -> None:
        for w in (L_WRIST, R_WRIST):
            if d.keypoints and len(d.keypoints) > w and d.keypoints[w].confidence > self.keypoint_conf:
                k = d.keypoints[w]
                tr.wrists.append((t, w, k.x, k.y))

    def fast_wrists(self, tr: Track, t: float) -> tuple[float, list[tuple[float, float]]]:
        """(top speed, positions of wrists currently moving faster than the threshold)."""
        height = max(1.0, tr.bbox.height)
        best, fast = 0.0, []
        for w in (L_WRIST, R_WRIST):
            samples = [s for s in tr.wrists if s[1] == w]
            if len(samples) < 2 or samples[-1][0] != t:
                continue
            now = samples[-1]
            old = [s for s in samples if now[0] - s[0] >= self.speed_window_s * 0.5]
            if not old:
                continue
            ref = old[-1]
            dt = now[0] - ref[0]
            if dt <= 0:
                continue
            dist = ((now[2] - ref[2]) ** 2 + (now[3] - ref[3]) ** 2) ** 0.5
            speed = dist / height / dt
            best = max(best, speed)
            if speed >= self.speed_threshold:
                fast.append((now[2], now[3]))
        return best, fast

    def evaluate(self, output: DetectorOutput) -> list[EventCandidate]:
        people = [d for d in output.detections
                  if d.label.lower() == "person" and d.confidence >= self.min_confidence and d.keypoints]
        tracker = self._trackers.setdefault(output.camera_id, SimpleTracker())
        t = output.timestamp
        tracks = tracker.update(people, t)
        for tr, d in zip(tracks, people):
            # forget samples older than twice the window
            while tr.wrists and t - tr.wrists[0][0] > 2 * self.speed_window_s:
                tr.wrists.popleft()
            self._record_wrists(tr, d, t)

        info = [self.fast_wrists(tr, t) for tr in tracks]
        for d, tr, (speed, _) in zip(people, tracks, info):
            d.attributes["tag"] = f"id{tr.track_id} spd {speed:.1f}"
        if len(people) < 2:
            return []

        hits = []
        for i in range(len(people)):
            for j in range(len(people)):
                if i == j:
                    continue
                a, b = people[i], people[j]
                mean_h = (a.bbox.height + b.bbox.height) / 2
                if box_gap(a.bbox, b.bbox) > self.close_gap * mean_h:
                    continue
                speed, fast = info[i]
                if any(inside(x, y, b.bbox, self.reach_margin) for x, y in fast):
                    hits.append((i, j, speed))
        if not hits:
            return []

        i, j, speed = max(hits, key=lambda h: h[2])
        a, b = people[i], people[j]
        a.attributes["tag"] = f"id{tracks[i].track_id} FIGHT? spd {speed:.1f}"
        return [EventCandidate(
            event_type=self.event_type, camera_id=output.camera_id,
            source_detector=output.detector, timestamp=t,
            confidence=min(a.confidence, b.confidence), detection=a, model=output.model,
            extra={"track_ids": sorted([tracks[i].track_id, tracks[j].track_id]),
                   "wrist_speed": round(speed, 2), "pairs": len(hits)},
        )]
