"""Save the evidence frame of a confirmed event."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ..schemas import Event


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "x"


class SnapshotWriter:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def save(self, event: Event, image: Any) -> str | None:
        """Write a JPG. Returns its path, or None if writing failed."""
        import cv2  # imported here so the rest of the Core runs without OpenCV

        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromtimestamp(event.confirmed_at or event.last_seen_at).strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}_{_safe(event.camera_id)}_{_safe(event.event_type)}_{event.event_id[:8]}.jpg"
        path = self.directory / name
        try:
            ok = cv2.imwrite(str(path), image)
        except Exception as exc:  # never let evidence saving stop detection
            print(f"[SNAPSHOT] error {type(exc).__name__}: {exc}")
            return None
        return str(path) if ok else None
