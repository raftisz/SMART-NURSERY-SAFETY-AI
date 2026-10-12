"""Climbing, zone mode: a person's visible ankles on a furniture "climb" zone.

Why: the pose rule (wrist above nose AND ankle above hip) reads posture, and a
person standing on a table has the same posture as one standing on the floor.
What differs is WHERE the feet are. Climb zones are drawn per camera on the top
surface of tables, shelves, window sills (tools/define_zone.py --kind climb) and
stored in config/zones.yaml next to the Out-of-Area zones (kind: climb).

Rule per frame: the mean of the person's ankles with confidence > keypoint_conf
lies inside an active climb zone -> one candidate per zone (subject = zone name).
`foot_offset` (default 0 = off) moves that point down by a fraction of the person
box height, because the ankle joint sits above the sole: 0.039 x body height in
Drillis & Contini (1966), reproduced as Fig. 4.1 in Winter, Biomechanics and Motor
Control of Human Movement. The box height stands in for body height, so for a
crouching person the shift is smaller.
There is NO fallback to the bottom of the person box: when the ankles cannot be
seen (legs hidden behind the table, cut by the image edge) the person does not
count, because a box cut by a table edge would put its bottom on the table top.
How long the feet must stay is the Event Manager's job (min_duration_s of the
'climbing' event type), like out_of_area.

Known limits: an adult (teacher) standing on a chair to reach a shelf counts,
sitting on a table inside the zone counts, a jump that lands in the zone counts
if it stays longer than min_duration_s, and zones must be redrawn when the
camera or the furniture moves. No accuracy claim until measured.
"""

from __future__ import annotations

from typing import Any

from src.cctv_core.detector_base import EventPolicy
from src.cctv_core.schemas import Detection, DetectorOutput, EventCandidate
from src.cctv_core.zones import Zone, load_zones
from src.utils.config import resolve_path

L_ANKLE, R_ANKLE = 15, 16


def foot_point(det: Detection, w: int, h: int, min_conf: float,
               foot_offset: float = 0.0) -> tuple[float, float] | None:
    """Normalised mean of the confident ankles, moved down by foot_offset x the person box height
    (the ankle joint sits above the sole); None when no ankle is seen."""
    kps = det.keypoints or []
    pts = [kps[i] for i in (L_ANKLE, R_ANKLE) if len(kps) > i and kps[i].confidence > min_conf]
    if not pts:
        return None
    y = sum(p.y for p in pts) / len(pts) + foot_offset * max(0.0, det.bbox.height)
    return (round(sum(p.x for p in pts) / len(pts) / w, 4), round(y / h, 4))


class ClimbZonePolicy(EventPolicy):
    def __init__(
        self,
        event_type: str = "climbing",
        zones_file: str | None = "config/zones.yaml",
        zones: dict[str, list[Any]] | None = None,
        keypoint_conf: float = 0.3,
        person_label: str = "person",
        foot_offset: float = 0.0,
    ):
        if not 0.0 <= foot_offset <= 0.2:
            raise ValueError("foot_offset must be a fraction of the person's height between 0 and 0.2")
        self.event_type = event_type
        self.keypoint_conf = float(keypoint_conf)
        self.foot_offset = float(foot_offset)
        self.person_label = person_label.lower()
        if zones is not None:
            loaded = {str(c): [z if isinstance(z, Zone) else Zone.from_dict(z) for z in zs] for c, zs in zones.items()}
        elif zones_file:
            loaded = load_zones(resolve_path(zones_file))
        else:
            loaded = {}
        self.zones = {cam: [z for z in zs if z.kind == "climb"] for cam, zs in loaded.items()}
        total = sum(len(z) for z in self.zones.values())
        print(f"[climbing] zone mode: {total} climb zone(s) loaded"
              + ("" if total else " - draw one with tools/define_zone.py --kind climb"))

    def evaluate(self, output: DetectorOutput) -> list[EventCandidate]:
        zones = self.zones.get(output.camera_id)
        if not zones or not output.frame_width or not output.frame_height:
            return []
        per_zone: dict[str, list[Detection]] = {}
        for d in output.detections:
            if d.label.lower() != self.person_label or not d.keypoints:
                continue
            foot = foot_point(d, output.frame_width, output.frame_height, self.keypoint_conf, self.foot_offset)
            if foot is None:
                continue
            for zone in zones:
                if zone.is_active(output.timestamp) and zone.contains(*foot):
                    d.attributes["tag"] = f"CLIMB ZONE {zone.name}"
                    per_zone.setdefault(zone.name, []).append(d)
        return [
            EventCandidate(
                event_type=self.event_type, camera_id=output.camera_id, source_detector=output.detector,
                timestamp=output.timestamp, confidence=max(d.confidence for d in hits),
                detection=max(hits, key=lambda d: d.confidence), model=output.model,
                extra={"zone": name, "mode": "zone", "people_in_zone": len(hits)}, subject=name,
            )
            for name, hits in per_zone.items()
        ]

    def draw(self, image: Any, camera_id: str) -> None:
        import time

        import cv2
        import numpy as np

        hh, ww = image.shape[:2]
        now = time.time()
        for zone in self.zones.get(camera_id) or []:
            pts = np.array([[int(x * ww), int(y * hh)] for x, y in zone.polygon], dtype=np.int32)
            color = (255, 0, 255) if zone.is_active(now) else (150, 150, 150)
            cv2.polylines(image, [pts], True, color, 2)
            x, y = pts[0]
            cv2.putText(image, f"climb {zone.name}", (int(x) + 4, max(15, int(y) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
