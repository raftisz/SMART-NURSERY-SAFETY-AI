"""Child climbing: YOLO pose model + the keypoint rule from legacy/kids_acsident.py.

Rule = the FIRST version in the legacy notebook (analyze_climbing_risk):
  hands_up      : a wrist above the nose (keypoint confidence > 0.3)
  feet_climbing : an ankle above the average hip height (confidence > 0.3)
  both          -> "DANGER: CLIMBING DETECTED!"  -> event
  one of them   -> "WARNING: HIGH RISK POSE"     -> event only if require_both=False
Two guards added on top of the legacy rule:
  - a body whose shoulder-to-hip line is more horizontal than vertical is
    LYING (e.g. nap time) and never counts as climbing
  - hips below keypoint confidence are not used as the "feet raised" reference

The project's custom pose weights live on the original author's Google Drive
and are not in this repo; until they are added, pretrained yolo11n-pose.pt is
used (no training). No accuracy claim is made for either.
"""

from __future__ import annotations

import time
from typing import Any

from src.cctv_core.detector_base import BaseDetector, EventPolicy
from src.cctv_core.schemas import (BBox, Detection, DetectorOutput, EventCandidate,
                                   Frame, Keypoint, ModelInfo)

from .yolo_label import resolve_weights

# COCO-17 keypoint indices
NOSE, L_SHOULDER, R_SHOULDER = 0, 5, 6
L_WRIST, R_WRIST, L_HIP, R_HIP, L_ANKLE, R_ANKLE = 9, 10, 11, 12, 15, 16


def _midpoint(kpts: list[Keypoint], a: int, b: int, min_conf: float) -> tuple[float, float] | None:
    """Mean position of the confident keypoints among a and b; None if neither is."""
    pts = [kpts[i] for i in (a, b) if kpts[i].confidence > min_conf]
    if not pts:
        return None
    return sum(p.x for p in pts) / len(pts), sum(p.y for p in pts) / len(pts)


def is_lying(kpts: list[Keypoint], min_conf: float = 0.3) -> bool:
    """True when the torso (shoulders -> hips) is more horizontal than vertical."""
    shoulders = _midpoint(kpts, L_SHOULDER, R_SHOULDER, min_conf)
    hips = _midpoint(kpts, L_HIP, R_HIP, min_conf)
    if shoulders is None or hips is None:
        return False  # orientation unknown: fall back to the legacy rule
    return abs(shoulders[0] - hips[0]) > abs(shoulders[1] - hips[1])


def classify_pose(kpts: list[Keypoint], min_conf: float = 0.3) -> str:
    """Port of legacy analyze_climbing_risk (version 1). Returns DANGER/WARNING/SAFE/LYING/NORMAL.

    Image y grows downwards, so "above" means a smaller y.
    """
    if len(kpts) < 17:
        return "NORMAL"
    if is_lying(kpts, min_conf):
        return "LYING"
    nose = kpts[NOSE]
    nose_y = nose.y if nose.confidence > min_conf else None
    hips = _midpoint(kpts, L_HIP, R_HIP, min_conf)

    hands_up = False
    if nose_y is not None:
        hands_up = any(kpts[i].y < nose_y and kpts[i].confidence > min_conf
                       for i in (L_WRIST, R_WRIST))
    feet_climbing = hips is not None and any(
        kpts[i].y < hips[1] and kpts[i].confidence > min_conf for i in (L_ANKLE, R_ANKLE))

    if hands_up and feet_climbing:
        return "DANGER"
    if hands_up or feet_climbing:
        return "WARNING"
    return "SAFE"


class ClimbingPoseDetector(BaseDetector):
    name = "climbing_pose"

    def __init__(self, weights: str = "yolo11n-pose.pt", conf: float = 0.3,
                 imgsz: int = 640, device: str = "auto", **kwargs: Any):
        super().__init__(**kwargs)
        self.weights = weights
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = device
        self._model: Any = None
        self._device = "cpu"
        self.model_info: ModelInfo | None = None

    def load(self) -> None:
        from ultralytics import YOLO
        from src.detection.yolo_detector import select_device

        path, digest = resolve_weights(self.weights)
        self._model = YOLO(path)
        self._device = select_device(self.device)
        self.model_info = ModelInfo(self.weights, digest, dict(self._model.names))
        print(f"[{self.name}] loaded {self.weights} on {self._device}")

    def process(self, frame: Frame) -> DetectorOutput:
        if self._model is None:
            raise RuntimeError(f"{self.name}: load() was not called")
        started = time.perf_counter()
        result = self._model.predict(frame.image, conf=self.conf, imgsz=self.imgsz,
                                     device=self._device, verbose=False)[0]
        detections: list[Detection] = []
        if result.boxes is not None and result.keypoints is not None:
            kp_all = result.keypoints.data.cpu().numpy()
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                kpts = [Keypoint(float(p[0]), float(p[1]), float(p[2])) for p in kp_all[i]] \
                    if i < len(kp_all) else None
                detections.append(Detection(
                    label=self._model.names.get(int(box.cls[0]), "person"),
                    confidence=float(box.conf[0]),
                    bbox=BBox(x1, y1, x2, y2),
                    keypoints=kpts,
                ))
        return DetectorOutput(self.name, frame.camera_id, frame.frame_index, frame.timestamp,
                              detections, (time.perf_counter() - started) * 1000.0,
                              self.model_info, frame.width, frame.height)


class ClimbingPosePolicy(EventPolicy):
    def __init__(self, event_type: str = "climbing", require_both: bool = True,
                 keypoint_conf: float = 0.3):
        self.event_type = event_type
        self.require_both = require_both
        self.keypoint_conf = keypoint_conf

    def evaluate(self, output: DetectorOutput) -> list[EventCandidate]:
        hits: list[tuple[Detection, str]] = []
        for d in output.detections:
            if not d.keypoints:
                continue
            status = classify_pose(d.keypoints, self.keypoint_conf)
            d.attributes["pose_status"] = status
            if status == "DANGER" or (status == "WARNING" and not self.require_both):
                hits.append((d, status))
        if not hits:
            return []
        top, status = max(hits, key=lambda h: h[0].confidence)
        return [EventCandidate(self.event_type, output.camera_id, output.detector,
                               output.timestamp, top.confidence, top, output.model,
                               {"pose_status": status, "people_at_risk": len(hits)})]
