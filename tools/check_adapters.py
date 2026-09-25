"""Step 8 wiring check - run on the Mac (needs torch/ultralytics).

1. Runs YoloDetector exactly as detect.py does (config.yaml values) and the new
   HazardObjectDetector adapter on the same images, and checks both give the
   same labels, confidences and boxes. This proves the adapter is wired
   correctly. It is NOT an accuracy test.
2. Loads the smoking model through SmokingDetector and prints what it reports.

Images are read from data/test_frames/ (read-only). Those frames were saved by
detect.py with boxes already drawn on them, so they are only suitable for this
wiring check, never for measuring accuracy.

Usage:  python tools/check_adapters.py [--limit 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.cctv_core.schemas import Frame  # noqa: E402
from src.detection.yolo_detector import YoloDetector  # noqa: E402
from src.detectors.hazard_object import HazardObjectDetector  # noqa: E402
from src.detectors.smoking import SmokingDetector  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def key(label, conf, x1, y1, x2, y2):
    return (label, round(conf, 4), round(x1), round(y1), round(x2), round(y2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    images = sorted((ROOT / "data" / "test_frames").glob("*.jpg"))[: args.limit]
    if not images:
        print("no images in data/test_frames")
        return 1

    cfg = load_config()  # the Phase 2 config.yaml, unchanged
    d = cfg["detection"]
    legacy = YoloDetector(cfg["model"]["weights"], d["target_classes"], cfg["model"]["device"])
    adapter = HazardObjectDetector(weights=cfg["model"]["weights"], conf=d["conf"],
                                   imgsz=d["imgsz"], nms_iou=d["nms_iou"],
                                   device=cfg["model"]["device"])
    adapter.load()

    same = 0
    for i, path in enumerate(images):
        img = cv2.imread(str(path))
        old, _, _ = legacy.detect(img, d["conf"], imgsz=d["imgsz"], nms_iou=d["nms_iou"])
        new = adapter.process(Frame("check", i, 0.0, img)).detections
        a = sorted(key(x.class_name, x.confidence, x.x1, x.y1, x.x2, x.y2) for x in old)
        b = sorted(key(x.label, x.confidence, x.bbox.x1, x.bbox.y1, x.bbox.x2, x.bbox.y2) for x in new)
        ok = a == b
        same += ok
        print(f"{'MATCH   ' if ok else 'MISMATCH'} {path.name}: {[x[0] for x in b]}")
    print(f"\nhazard_object adapter vs YoloDetector: {same}/{len(images)} identical")

    smoking = SmokingDetector()
    smoking.load()
    print(f"smoking classes: {smoking.model_info.class_names}")
    print(f"smoking sha256 : {smoking.model_info.sha256}")
    for i, path in enumerate(images[:3]):
        out = smoking.process(Frame("check", i, 0.0, cv2.imread(str(path))))
        print(f"  {path.name}: {[(x.label, round(x.confidence, 2)) for x in out.detections]}")
    print("(smoking output is shown only to prove the model runs; it is not an accuracy result)")
    return 0 if same == len(images) else 2


if __name__ == "__main__":
    sys.exit(main())
