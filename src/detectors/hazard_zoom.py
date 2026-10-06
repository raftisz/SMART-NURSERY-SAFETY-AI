"""Person-guided zoom: a second, close-up look around small (far-away) people.

Why: a pair of scissors held by a child 3-4 m from the camera is ~10-20 px long
in a 1280x720 frame. After the letterbox resize to imgsz the detector sees even
fewer pixels, and the object blends into the background - this is the
"detects near, misses far" failure. People, on the other hand, are big and are
detected reliably at that distance. So: take the person boxes the same model
already produced, cut out each SMALL person (plus a margin for outstretched
arms), and run the detector again on that crop. The crop is resized up to
crop_imgsz, so the object becomes 3-6x larger to the network, and most of the
distracting background is gone.

Cost: one extra inference per crop (batched), only for far-away people, and
only every `every_n` frames if configured. Near people are skipped: they are
already large in the full frame.

This module is pure geometry + merging; the model call is passed in as a
function so it is unit-tested without a model (tests/test_hazard_zoom.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

Box = tuple[float, float, float, float]          # x1, y1, x2, y2 in pixels
Det = tuple[str, float, Box]                      # label, confidence, box


@dataclass
class ZoomConfig:
    enabled: bool = False
    max_crops: int = 3             # at most this many people per frame
    max_person_height: float = 0.6  # only people shorter than this fraction of the frame (far away)
    min_person_conf: float = 0.3
    pad_x: float = 0.6             # margin as a fraction of person width (arms reach sideways)
    pad_y: float = 0.15
    min_crop: int = 160            # px; tiny crops are grown to at least this
    crop_imgsz: int = 640
    every_n: int = 1               # run the zoom pass on every n-th frame the detector runs
    merge_iou: float = 0.5

    @classmethod
    def from_dict(cls, data: dict | None) -> "ZoomConfig":
        data = dict(data or {})
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown zoom option(s): {sorted(unknown)}")
        return cls(**data)


def iou(a: Box, b: Box) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def crop_boxes(persons: Sequence[Det], width: int, height: int, cfg: ZoomConfig) -> list[tuple[int, int, int, int]]:
    """Integer crop windows around the small, confident people (largest crops not needed)."""
    small = [p for p in persons
             if p[1] >= cfg.min_person_conf and (p[2][3] - p[2][1]) < cfg.max_person_height * height]
    small.sort(key=lambda p: -p[1])
    crops: list[tuple[int, int, int, int]] = []
    for _, _, (x1, y1, x2, y2) in small:
        w, h = x2 - x1, y2 - y1
        cx1, cy1 = x1 - cfg.pad_x * w, y1 - cfg.pad_y * h
        cx2, cy2 = x2 + cfg.pad_x * w, y2 + cfg.pad_y * h
        # grow to the minimum size around the centre
        cw, ch = cx2 - cx1, cy2 - cy1
        if cw < cfg.min_crop:
            cx1 -= (cfg.min_crop - cw) / 2
            cx2 += (cfg.min_crop - cw) / 2
        if ch < cfg.min_crop:
            cy1 -= (cfg.min_crop - ch) / 2
            cy2 += (cfg.min_crop - ch) / 2
        box = (int(max(0, cx1)), int(max(0, cy1)), int(min(width, cx2)), int(min(height, cy2)))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            continue
        # a crop mostly inside an already chosen crop adds nothing
        if any(iou(box, c) > 0.6 for c in crops):
            continue
        crops.append(box)
        if len(crops) >= cfg.max_crops:
            break
    return crops


def merge(full: list[Det], zoomed: list[Det], merge_iou: float) -> list[Det]:
    """Full-frame detections + zoom detections; a same-label pair overlapping by
    >= merge_iou keeps only the more confident box."""
    out = list(full)
    for z in sorted(zoomed, key=lambda d: -d[1]):
        clash = [i for i, f in enumerate(out) if f[0] == z[0] and iou(f[2], z[2]) >= merge_iou]
        if not clash:
            out.append(z)
            continue
        i = max(clash, key=lambda k: out[k][1])
        if z[1] > out[i][1]:
            out[i] = z
    return out


def zoom_detect(image, full: list[Det], person_label: str, cfg: ZoomConfig,
                predict: Callable[[list], list[list[Det]]]) -> tuple[list[Det], int]:
    """Run `predict` on the person crops of `image` and merge. Returns (detections, n_crops).

    `predict(crops)` gets a list of image arrays and returns, per crop, detections
    in CROP pixel coordinates (only the labels of interest)."""
    h, w = image.shape[:2]
    persons = [d for d in full if d[0].lower() == person_label]
    windows = crop_boxes(persons, w, h, cfg)
    if not windows:
        return full, 0
    crops = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in windows]
    zoomed: list[Det] = []
    for (x1, y1, _, _), dets in zip(windows, predict(crops)):
        for label, conf, (a, b, c, d) in dets:
            zoomed.append((label, conf, (a + x1, b + y1, c + x1, d + y1)))
    return merge(full, zoomed, cfg.merge_iou), len(windows)
