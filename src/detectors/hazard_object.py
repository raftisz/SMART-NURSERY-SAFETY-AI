"""Hazardous objects: YOLO (person / scissors / knife), optionally with person-guided zoom.

Defaults copy config.yaml used by detect.py in Phase 2 (conf 0.25, imgsz 640,
extra NMS IoU 0.5) so behaviour matches the tested harness.

weights: the stock COCO `yolo11n.pt`, or a fine-tuned COCO-80 model from
tools/train_hazard_model.py (models/hazard/*.pt) - same class names, so nothing
downstream changes.

zoom (off unless enabled): see src/detectors/hazard_zoom.py. Re-runs the model
on crops around small (far-away) people so a small knife/scissors in a hand is
seen at 3-6x the size. Detections found only by the zoom pass carry
attributes["zoom"] = True.
"""

from __future__ import annotations

import time
from typing import Any

from src.cctv_core.schemas import BBox, Detection, DetectorOutput, Frame

from .hazard_zoom import ZoomConfig, zoom_detect
from .yolo_label import YoloLabelDetector


class HazardObjectDetector(YoloLabelDetector):
    name = "hazard_object"

    def __init__(self, weights: str = "yolo11n.pt",
                 target_classes: list[str] | None = None,
                 zoom: dict[str, Any] | None = None, **kwargs):
        super().__init__(weights, target_classes or ["person", "scissors", "knife"], **kwargs)
        self.zoom = ZoomConfig.from_dict(zoom)
        self._runs = 0
        self.zoom_crops = 0  # total crops processed (for reports)

    def _hazard_labels(self) -> list[str]:
        return [c for c in self.target_classes if c.lower() != "person"]

    def _predict_crops(self, crops: list) -> list[list[tuple[str, float, tuple]]]:
        ids = sorted(self._yolo.class_ids[c] for c in self._hazard_labels() if c in self._yolo.class_ids)
        if not ids:
            return [[] for _ in crops]
        results = self._yolo.model.predict(crops, conf=self.conf, imgsz=self.zoom.crop_imgsz,
                                           classes=ids, device=self._yolo.device, verbose=False)
        out = []
        for r in results:
            out.append([(self._yolo.names[int(c)], float(p), tuple(float(v) for v in b))
                        for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                           r.boxes.conf.tolist())])
        return out

    def process(self, frame: Frame) -> DetectorOutput:
        output = super().process(frame)
        self._runs += 1
        if not self.zoom.enabled or frame.image is None or (self._runs - 1) % self.zoom.every_n:
            return output
        started = time.perf_counter()
        full = [(d.label, d.confidence, (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2))
                for d in output.detections]
        merged, n = zoom_detect(frame.image, full, "person", self.zoom, self._predict_crops)
        if n:
            self.zoom_crops += n
            dets: list[Detection] = []
            originals = {(d.label, d.confidence, (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2)): d
                         for d in output.detections}
            for label, conf, box in merged:
                orig = originals.get((label, conf, box))
                if orig is not None:
                    dets.append(orig)
                else:
                    dets.append(Detection(label, conf, BBox(*box), attributes={"zoom": True, "tag": "zoom"}))
            output.detections = dets
        output.inference_ms += (time.perf_counter() - started) * 1000.0
        return output
