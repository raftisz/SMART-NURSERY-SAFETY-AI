"""Hazardous objects: pretrained COCO YOLO11n (person / scissors / knife).

Defaults copy config.yaml used by detect.py in Phase 2 (conf 0.25, imgsz 640,
extra NMS IoU 0.5) so behaviour matches the tested harness.
"""

from __future__ import annotations

from .yolo_label import YoloLabelDetector


class HazardObjectDetector(YoloLabelDetector):
    name = "hazard_object"

    def __init__(self, weights: str = "yolo11n.pt",
                 target_classes: list[str] | None = None, **kwargs):
        super().__init__(weights, target_classes or ["person", "scissors", "knife"], **kwargs)
