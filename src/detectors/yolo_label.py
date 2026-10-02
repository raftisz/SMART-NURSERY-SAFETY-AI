"""Generic adapter: any Ultralytics YOLO *detection* model -> BaseDetector.

Reuses src/detection/yolo_detector.py (YoloDetector) unchanged, so class names
are still resolved from model.names and the extra class-wise NMS still applies.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from src.cctv_core.detector_base import BaseDetector
from src.cctv_core.schemas import Detection, DetectorOutput, Frame, ModelInfo
from src.utils.config import resolve_path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_weights(weights: str) -> tuple[str, str | None]:
    """Project file -> absolute path + SHA-256. Otherwise pass the name through
    unchanged so Ultralytics can download stock weights (e.g. yolo11n.pt)."""
    candidate = resolve_path(weights)
    if candidate.is_file():
        return str(candidate), sha256_of(candidate)
    return weights, None


class YoloLabelDetector(BaseDetector):
    name = "yolo_label"

    def __init__(
        self,
        weights: str,
        target_classes: list[str],
        conf: float = 0.25,
        imgsz: int = 640,
        nms_iou: float | None = 0.5,
        device: str = "auto",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.weights = weights
        self.target_classes = list(target_classes)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.nms_iou = nms_iou
        self.device = device
        self._yolo: Any = None
        self.model_info: ModelInfo | None = None

    def load(self) -> None:
        from src.detection.yolo_detector import YoloDetector  # imports torch

        path, digest = resolve_weights(self.weights)
        self._yolo = YoloDetector(path, self.target_classes, self.device)
        if self._yolo.missing_classes:
            print(f"[{self.name}] WARN classes not in {self.weights}: "
                  f"{', '.join(self._yolo.missing_classes)}")
        self.model_info = ModelInfo(weights=self.weights, sha256=digest,
                                    class_names=dict(self._yolo.names))
        print(f"[{self.name}] loaded {self.weights} on {self._yolo.device} "
              f"-> {sorted(self._yolo.class_ids)}")

    def process(self, frame: Frame) -> DetectorOutput:
        if self._yolo is None:
            raise RuntimeError(f"{self.name}: load() was not called")
        started = time.perf_counter()
        legacy, _infer_ms, _raw = self._yolo.detect(
            frame.image, self.conf, imgsz=self.imgsz, nms_iou=self.nms_iou
        )
        return DetectorOutput(
            detector=self.name,
            camera_id=frame.camera_id,
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
            detections=[Detection.from_legacy(d) for d in legacy],
            inference_ms=(time.perf_counter() - started) * 1000.0,
            model=self.model_info,
            frame_width=frame.width,
            frame_height=frame.height,
        )
