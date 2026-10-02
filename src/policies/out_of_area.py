"""Child Out-of-Area: a child's feet inside an exit zone during active hours.

Reuses the `person` boxes the hazard_object model already produces, so it
adds no model run. For each person:
  1. foot point = bottom-centre of the box (where they stand on the floor)
  2. inside a zone of this camera, and the zone is active now?
  3. child? height / expected-standing-adult-height-at-that-spot < child_ratio
How long they must stay is the Event Manager's job (min_duration_s of the
'out_of_area' event type in config/core.yaml).

Known limits (v1): an adult sitting or crouching in the zone looks like a
child; a child walking IN from outside also triggers (no direction yet);
boxes cut by the top/bottom image edge are skipped because their height is
unreliable. No accuracy claim until it is measured.
"""

from __future__ import annotations

from typing import Any

from src.cctv_core.detector_base import EventPolicy
from src.cctv_core.schemas import Detection, DetectorOutput, EventCandidate
from src.cctv_core.zones import Zone, load_zones
from src.utils.config import resolve_path

EDGE_MARGIN = 0.01  # boxes touching the top/bottom 1% are not measured


class OutOfAreaPolicy(EventPolicy):
    def __init__(
        self,
        event_type: str = "out_of_area",
        zones_file: str | None = "config/zones.yaml",
        zones: dict[str, list[dict[str, Any]]] | None = None,
        person_label: str = "person",
        min_confidence: float = 0.4,
        child_ratio: float = 0.75,
    ):
        self.event_type = event_type
        self.person_label = person_label.lower()
        self.min_confidence = float(min_confidence)
        self.child_ratio = float(child_ratio)
        if zones is not None:
            self.zones = {str(c): [Zone.from_dict(z) for z in zs] for c, zs in zones.items()}
        elif zones_file:
            self.zones = load_zones(resolve_path(zones_file))
        else:
            self.zones = {}
        self._warned: set[str] = set()
        total = sum(len(z) for z in self.zones.values())
        print(f"[out_of_area] {total} zone(s) loaded"
              + ("" if total else " - run tools/define_zone.py to create one"))

    def _warn_once(self, key: str, text: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            print(f"[out_of_area] {text}")

    def child_score(self, zone: Zone, det: Detection, w: int, h: int) -> float | None:
        """height / expected adult height; None if it cannot be measured."""
        if zone.calibration is None:
            return None
        if det.bbox.y1 <= EDGE_MARGIN * h or det.bbox.y2 >= (1 - EDGE_MARGIN) * h:
            return None
        foot_y = det.bbox.y2 / h
        return (det.bbox.height / h) / zone.calibration.expected(foot_y)

    def evaluate(self, output: DetectorOutput) -> list[EventCandidate]:
        zones = self.zones.get(output.camera_id)
        if not zones:
            return []
        w, h = output.frame_width, output.frame_height
        if not w or not h:
            self._warn_once("size", "frame size unknown - skipped")
            return []

        per_zone: dict[str, list[tuple[Detection, float | None]]] = {}
        for d in output.detections:
            if d.label.lower() != self.person_label or d.confidence < self.min_confidence:
                continue
            fx = (d.bbox.x1 + d.bbox.x2) / 2 / w
            fy = d.bbox.y2 / h
            for zone in zones:
                if not zone.is_active(output.timestamp) or not zone.contains(fx, fy):
                    continue
                score = None
                if zone.child_filter:
                    if zone.calibration is None:
                        self._warn_once(f"cal:{zone.name}",
                                        f"zone '{zone.name}' is not calibrated - ignored "
                                        "(calibrate it, or set child_filter: false)")
                        continue
                    score = self.child_score(zone, d, w, h)
                    if score is None:
                        continue
                    limit = zone.child_ratio if zone.child_ratio is not None else self.child_ratio
                    d.attributes["tag"] = f"{'CHILD' if score < limit else 'adult'} {score:.2f}"
                    if score >= limit:
                        continue
                per_zone.setdefault(zone.name, []).append((d, score))

        out: list[EventCandidate] = []
        for name, hits in per_zone.items():
            top, score = max(hits, key=lambda x: x[0].confidence)
            out.append(EventCandidate(
                event_type=self.event_type, camera_id=output.camera_id,
                source_detector=output.detector, timestamp=output.timestamp,
                confidence=top.confidence, detection=top, model=output.model,
                extra={"zone": name, "height_ratio": None if score is None else round(score, 2),
                       "people_in_zone": len(hits)},
                subject=name,
            ))
        return out

    def draw(self, image: Any, camera_id: str) -> None:
        import time

        import cv2
        import numpy as np

        zones = self.zones.get(camera_id) or []
        if not zones:
            return
        hh, ww = image.shape[:2]
        now = time.time()
        for zone in zones:
            pts = np.array([[int(x * ww), int(y * hh)] for x, y in zone.polygon], dtype=np.int32)
            color = (0, 165, 255) if zone.is_active(now) else (150, 150, 150)
            cv2.polylines(image, [pts], True, color, 2)
            x, y = pts[0]
            cv2.putText(image, f"zone {zone.name}", (int(x) + 4, max(15, int(y) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
