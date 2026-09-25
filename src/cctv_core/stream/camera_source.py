"""Frames from a webcam index, an RTSP/HTTP URL, a video file, or a still image.

Webcam backend choice mirrors detect.py: AVFoundation on macOS, CAP_ANY elsewhere.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class CameraSource:
    def __init__(self, source: Any, width: int | None = None, height: int | None = None,
                 flip: bool | None = None, loop_image: bool = True):
        import cv2

        self._cv2 = cv2
        raw = str(source)
        self.source = raw
        self.is_webcam = raw.isdigit()
        self.is_stream = "://" in raw
        self.is_image = not (self.is_webcam or self.is_stream) and Path(raw).suffix.lower() in IMAGE_SUFFIXES
        self.loop_image = loop_image
        # Mirror a live self-view webcam by default, like detect.py (flip_webcam: true).
        self.flip = self.is_webcam if flip is None else bool(flip)
        self._still = None
        self._cap = None
        self._served_image = False

        if self.is_webcam:
            api = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
            self._cap = cv2.VideoCapture(int(raw), api)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Could not open camera {raw}. Another app may be using it, or the "
                    "terminal has no camera permission (System Settings > Privacy & Security > Camera)."
                )
            if width:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        elif self.is_stream:
            self._cap = cv2.VideoCapture(raw)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open stream: {raw}")
        else:
            path = Path(raw)
            if not path.is_file():
                raise FileNotFoundError(f"Source not found: {raw}")
            if self.is_image:
                self._still = cv2.imread(str(path))
                if self._still is None:
                    raise RuntimeError(f"Could not decode image: {raw}")
            else:
                self._cap = cv2.VideoCapture(str(path))
                if not self._cap.isOpened():
                    raise RuntimeError(f"Could not open video: {raw}")

    @property
    def kind(self) -> str:
        if self.is_webcam:
            return "webcam"
        if self.is_stream:
            return "stream"
        return "image" if self.is_image else "video"

    def read(self):
        """Next BGR frame, or None when the source has ended."""
        if self._still is not None:
            if self._served_image and not self.loop_image:
                return None
            self._served_image = True
            frame = self._still.copy()
        else:
            ok, frame = self._cap.read()
            if not ok:
                return None
        if self.flip:
            # Flip BEFORE inference so boxes match what is displayed.
            frame = self._cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
