"""Frames from a webcam index, an RTSP/HTTP URL, a video file, or a still image.

Webcam backend choice mirrors detect.py: AVFoundation on macOS, CAP_ANY elsewhere.
A live source (webcam or stream) retries failed reads and reopens the capture,
so one dropped frame does not end the run; a video file still ends at its end.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class CameraSource:
    def __init__(self, source: Any, width: int | None = None, height: int | None = None,
                 flip: bool | None = None, loop_image: bool = True, loop_video: bool = False,
                 max_read_failures: int = 50, retry_delay_s: float = 0.1, reopen_every: int = 10):
        import cv2

        self._cv2 = cv2
        raw = str(source)
        self.source = raw
        self.is_webcam = raw.isdigit()
        self.is_stream = "://" in raw
        self.is_image = not (self.is_webcam or self.is_stream) and Path(raw).suffix.lower() in IMAGE_SUFFIXES
        self.loop_image = loop_image
        self.loop_video = loop_video  # restart a video file at its end (backup clip for demos)
        self.width, self.height = width, height
        self.max_read_failures = max(1, int(max_read_failures))
        self.retry_delay_s = float(retry_delay_s)
        self.reopen_every = max(1, int(reopen_every))
        # Mirror a live self-view webcam by default, like detect.py (flip_webcam: true).
        self.flip = self.is_webcam if flip is None else bool(flip)
        self._still = None
        self._cap = None
        self._served_image = False

        if self.is_webcam:
            self._cap = self._open_capture()
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Could not open camera {raw}. Another app may be using it, or the "
                    "terminal has no camera permission (System Settings > Privacy & Security > Camera)."
                )
        elif self.is_stream:
            self._cap = self._open_capture()
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

    def _open_capture(self):
        """Open the webcam or stream capture (also used to reconnect)."""
        cv2 = self._cv2
        if self.is_webcam:
            api = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(self.source), api)
            if cap.isOpened():
                if self.width:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                if self.height:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return cap
        return cv2.VideoCapture(self.source)

    @property
    def fps(self) -> float | None:
        """Frame rate reported by a video file or stream; None if unknown (image, some webcams)."""
        if self._cap is None:
            return None
        value = float(self._cap.get(self._cv2.CAP_PROP_FPS) or 0.0)
        return value if value > 0 else None

    @property
    def is_live(self) -> bool:
        return self.is_webcam or self.is_stream

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
        elif self.is_live:
            frame = self._read_live()
            if frame is None:
                return None
        else:
            ok, frame = self._cap.read()
            if not ok and self.loop_video:
                self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                return None
        if self.flip:
            # Flip BEFORE inference so boxes match what is displayed.
            frame = self._cv2.flip(frame, 1)
        return frame

    def _read_live(self):
        """Read a live frame, retrying and reopening the capture before giving up."""
        failures = 0
        while True:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                if failures:
                    print(f"[SOURCE] recovered after {failures} failed read(s)")
                return frame
            failures += 1
            if failures >= self.max_read_failures:
                print(f"[SOURCE] {failures} failed reads in a row - giving up")
                return None
            if failures == 1:
                print("[SOURCE] frame read failed - retrying")
            if failures % self.reopen_every == 0:
                print("[SOURCE] reopening the capture")
                self._cap.release()
                self._cap = self._open_capture()
            time.sleep(self.retry_delay_s)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
