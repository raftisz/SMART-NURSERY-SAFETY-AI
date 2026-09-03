"""Thin wrapper around a pretrained Ultralytics YOLO model.

Phase 2 only: run a pretrained model and report what it sees. No training,
no fine-tuning, no risk logic.
"""

import time
from dataclasses import dataclass

import torch
from ultralytics import YOLO


@dataclass
class Detection:
    """One detected box, in pixel coordinates of the frame passed to detect()."""

    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Box area in pixels - the evidence for confidence-vs-object-size."""
        return self.width * self.height


def _iou(a: "Detection", b: "Detection") -> float:
    """Intersection-over-union of two boxes."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def class_wise_nms(detections: list["Detection"], iou_thres: float) -> list["Detection"]:
    """Drop duplicate boxes of the SAME class that overlap by more than iou_thres.

    Ultralytics already runs NMS internally, but at a looser default IoU, so a
    second box on one object can survive. This is a stricter pass on top.

    It is deliberately class-wise: a hand inside a person box, or scissors on a
    table overlapping a person, are different objects and must both survive.
    """
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if any(k.class_id == det.class_id and _iou(k, det) > iou_thres for k in kept):
            continue
        kept.append(det)
    return kept


def select_device(preference: str = "auto") -> str:
    """Pick the inference device.

    "auto" prefers CUDA, then Apple MPS, then CPU. An explicit preference is
    returned unchanged so the caller can force a device for benchmarking.
    """
    if preference and preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloDetector:
    """Loads the model once and resolves target class names to integer IDs.

    Class IDs are looked up from model.names rather than hardcoded, because
    the mapping is a property of the weights, not a constant.
    """

    def __init__(self, weights: str, target_class_names: list[str], device: str = "auto"):
        self.weights = weights
        self.device = select_device(device)
        self.model = YOLO(weights)
        self.names: dict[int, str] = self.model.names

        # name -> id, case-insensitive, for the classes we care about.
        lookup = {name.lower(): idx for idx, name in self.names.items()}
        self.class_ids: dict[str, int] = {}
        self.missing_classes: list[str] = []
        for wanted in target_class_names:
            key = wanted.lower()
            if key in lookup:
                self.class_ids[wanted] = lookup[key]
            else:
                self.missing_classes.append(wanted)

        # Passed to predict(classes=...) so filtering happens inside the model
        # rather than over a full 80-class result set.
        self.filter_ids: list[int] = sorted(self.class_ids.values())

        # Settings read BACK off the predictor after each call, so callers can
        # report what inference actually used instead of what was requested.
        # These are the single source of truth for the exit summary.
        self.effective_imgsz: int | None = None
        self.effective_conf: float | None = None
        self.effective_model_iou: float | None = None
        self.ultra_inference_ms: float | None = None
        self.model_input_hw: tuple[int, int] | None = None

    def describe(self) -> str:
        """One-off startup summary of the resolved mapping."""
        lines = [
            f"Model        : {self.weights}",
            f"Device       : {self.device}",
            f"Total classes: {len(self.names)}",
            "Resolved target classes (name -> id, from model.names):",
        ]
        for name, idx in sorted(self.class_ids.items(), key=lambda kv: kv[1]):
            lines.append(f"    {name:<10} -> {idx}")
        if self.missing_classes:
            lines.append(f"  !! NOT in this model: {', '.join(self.missing_classes)}")
        return "\n".join(lines)

    def detect(
        self, frame, conf: float, imgsz: int = 640, nms_iou: float | None = None
    ) -> tuple[list[Detection], float, int]:
        """Run inference on one BGR frame.

        Returns (detections, inference_ms, raw_count) where raw_count is the
        number of boxes before the extra NMS pass, so duplicate suppression can
        be reported.

        The frame is used exactly as given - any flipping must already have
        happened, so returned boxes match what the caller displays.

        imgsz is the size the model resizes to internally. It changes both
        accuracy and speed, so it must be held constant across a test matrix
        and recorded alongside the results.
        """
        started = time.perf_counter()
        results = self.model.predict(
            frame,
            conf=conf,
            imgsz=imgsz,
            classes=self.filter_ids or None,
            device=self.device,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - started) * 1000.0

        # Read the settings back off the predictor that just ran. If anything
        # between argparse and predict() drops or overrides a value, this is
        # where it becomes visible, because it comes from the call itself.
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and getattr(predictor, "args", None) is not None:
            eff = predictor.args.imgsz
            # imgsz may come back as [h, w]; normalise to the long side.
            self.effective_imgsz = max(eff) if isinstance(eff, (list, tuple)) else eff
            self.effective_conf = predictor.args.conf
            self.effective_model_iou = predictor.args.iou
            # Record the real letterboxed tensor dimensions by wrapping
            # preprocess once. args.imgsz says what was asked of the predictor;
            # this says what the network actually received. Populated from the
            # second call onward, since the predictor only exists after the first.
            if not getattr(predictor, "_ng_wrapped", False):
                original_preprocess = predictor.preprocess

                def _recording_preprocess(ims, _orig=original_preprocess, _self=self):
                    tensor = _orig(ims)
                    try:
                        _self.model_input_hw = (int(tensor.shape[-2]), int(tensor.shape[-1]))
                    except (AttributeError, IndexError, TypeError):
                        pass
                    return tensor

                predictor.preprocess = _recording_preprocess
                predictor._ng_wrapped = True

        speed = getattr(results[0], "speed", None)
        if speed:
            self.ultra_inference_ms = speed.get("inference")

        detections: list[Detection] = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.names.get(class_id, str(class_id)),
                    confidence=float(box.conf[0]),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        raw_count = len(detections)
        if nms_iou is not None:
            detections = class_wise_nms(detections, nms_iou)
        return detections, inference_ms, raw_count
