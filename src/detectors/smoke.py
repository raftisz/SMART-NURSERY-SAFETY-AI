"""Smoke: the project's own one-class model models/smoke/smoke_custom_yolo11n_v2.pt.

Trained outside the pipeline (models/smoke/README.md); conf 0.25 and imgsz 640
match the live test. The weights file is a pickle, so its SHA-256 must match the
value in config before Ultralytics opens it, and the model must have a 'smoke'
class (looked up in model.names) - otherwise the module refuses to load.
"""

from __future__ import annotations

from .yolo_label import YoloLabelDetector, sha256_of
from src.utils.config import resolve_path


class SmokeDetector(YoloLabelDetector):
    name = "smoke"

    def __init__(self, weights: str = "models/smoke/smoke_custom_yolo11n_v2.pt",
                 target_classes: list[str] | None = None, conf: float = 0.25,
                 sha256: str | None = None, **kwargs):
        super().__init__(weights, target_classes or ["smoke"], conf=conf, **kwargs)
        self.sha256 = sha256

    def load(self) -> None:
        path = resolve_path(self.weights)
        if not path.is_file():
            raise FileNotFoundError(f"{self.weights} not found")
        if not self.sha256:
            raise ValueError(f"{self.name}: no expected sha256 in config - not loading {self.weights}")
        digest = sha256_of(path)
        if digest != self.sha256:
            raise ValueError(f"{self.name}: SHA-256 of {self.weights} is {digest}, expected {self.sha256}")
        super().load()
        if self._yolo.missing_classes:
            raise ValueError(f"{self.name}: {self.weights} has no class "
                             f"{', '.join(self._yolo.missing_classes)}")
