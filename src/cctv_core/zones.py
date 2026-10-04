"""Zones drawn on a camera image, plus a per-camera adult-height model.

All geometry is NORMALISED (0..1 of image width/height), so a zone stays
correct if the camera resolution changes. Zones are stored in
config/zones.yaml, written by tools/define_zone.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

Point = tuple[float, float]


def parse_hours(text: str) -> tuple[int, int]:
    """"09:00-15:00" -> (540, 900) minutes. A range may cross midnight ("22:00-06:00")."""
    try:
        start, end = (part.strip() for part in text.split("-"))
        sh, sm = (int(v) for v in start.split(":"))
        eh, em = (int(v) for v in end.split(":"))
    except ValueError as exc:
        raise ValueError(f"bad time range {text!r}, expected HH:MM-HH:MM") from exc
    for h, m in ((sh, sm), (eh, em)):
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError(f"bad time in {text!r}")
    return sh * 60 + sm, eh * 60 + em


@dataclass
class HeightModel:
    """Expected height of a STANDING ADULT as a function of where their feet are.

    Built from two reference measurements (an adult standing near the camera and
    far from it); in between, the height is interpolated linearly. Beyond them it
    is extrapolated by at most half the calibrated span, then held constant, so a
    bad or narrow calibration cannot produce absurd expected heights.
    Values are fractions of the image height.
    """

    near_foot_y: float
    near_height: float
    far_foot_y: float
    far_height: float

    def expected(self, foot_y: float) -> float:
        dy = self.near_foot_y - self.far_foot_y
        if abs(dy) < 1e-6:
            return max(0.02, (self.near_height + self.far_height) / 2)
        slope = (self.near_height - self.far_height) / dy
        lo, hi = sorted((self.far_foot_y, self.near_foot_y))
        margin = (hi - lo) * 0.5
        foot_y = min(max(foot_y, lo - margin), hi + margin)
        return max(0.02, self.far_height + slope * (foot_y - self.far_foot_y))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeightModel":
        near, far = data["near"], data["far"]
        return cls(float(near["foot_y"]), float(near["height"]),
                   float(far["foot_y"]), float(far["height"]))

    def to_dict(self) -> dict[str, Any]:
        return {"near": {"foot_y": round(self.near_foot_y, 4), "height": round(self.near_height, 4)},
                "far": {"foot_y": round(self.far_foot_y, 4), "height": round(self.far_height, 4)}}


@dataclass
class Zone:
    name: str
    polygon: list[Point]
    active_hours: list[str] = field(default_factory=list)  # empty = always active
    child_filter: bool = True        # False = any person counts (no calibration needed)
    child_ratio: float | None = None  # None = use the policy default
    calibration: HeightModel | None = None

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(f"zone {self.name!r} needs at least 3 points")
        for x, y in self.polygon:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"zone {self.name!r}: points must be normalised 0..1")
        self._ranges = [parse_hours(h) for h in self.active_hours]

    def contains(self, x: float, y: float) -> bool:
        """Point-in-polygon (ray casting) on normalised coordinates."""
        inside = False
        pts = self.polygon
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_cross:
                    inside = not inside
            j = i
        return inside

    def is_active(self, timestamp: float) -> bool:
        if not self._ranges:
            return True
        t = datetime.fromtimestamp(timestamp)
        minute = t.hour * 60 + t.minute
        for start, end in self._ranges:
            if start <= end:
                if start <= minute < end:
                    return True
            elif minute >= start or minute < end:  # crosses midnight
                return True
        return False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Zone":
        cal = data.get("calibration")
        return cls(
            name=str(data["name"]),
            polygon=[(float(p[0]), float(p[1])) for p in data["polygon"]],
            active_hours=list(data.get("active_hours") or []),
            child_filter=bool(data.get("child_filter", True)),
            child_ratio=float(data["child_ratio"]) if data.get("child_ratio") is not None else None,
            calibration=HeightModel.from_dict(cal) if cal else None,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "polygon": [[round(x, 4), round(y, 4)] for x, y in self.polygon],
            "active_hours": list(self.active_hours),
            "child_filter": self.child_filter,
        }
        if self.child_ratio is not None:
            data["child_ratio"] = self.child_ratio
        if self.calibration is not None:
            data["calibration"] = self.calibration.to_dict()
        return data


ZONES_HEADER = (
    "# Zones per camera - written by tools/define_zone.py (you may also edit by hand)\n"
    "# Coordinates are fractions (0..1) of the image width/height.\n"
)


def load_zones(path: str | Path) -> dict[str, list[Zone]]:
    """{camera_id: [Zone, ...]}. A missing file means no zones."""
    p = Path(path)
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {str(cam): [Zone.from_dict(z) for z in (zones or [])]
            for cam, zones in (data.get("zones") or {}).items()}


def save_zone(path: str | Path, camera_id: str, zone: Zone) -> None:
    """Add or replace (same name) one zone for one camera, keeping all others."""
    p = Path(path)
    data = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.is_file() else {}
    zones = data.setdefault("zones", {}) or {}
    data["zones"] = zones
    cam = [z for z in (zones.get(camera_id) or []) if z.get("name") != zone.name]
    cam.append(zone.to_dict())
    zones[camera_id] = cam
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(ZONES_HEADER + yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                 encoding="utf-8")
