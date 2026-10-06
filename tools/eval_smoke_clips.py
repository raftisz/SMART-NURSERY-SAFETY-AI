"""Measure smoke models on the clips in ~/smoke_eval_clips (test only, never for training).

    smoke_*.mp4 = clip WITH smoke, none_*.mp4 = clip WITHOUT smoke (other files are skipped)

Criteria, fixed before measuring (do not change after seeing results):
    confidence 0.25, one count per frame however many boxes
    smoke clips : exactly 5; a clip is detected when >= 1 frame has smoke >= 0.25;
                  PASS needs >= 4 of 5 clips detected
    none clips  : at least 1; EVERY clip may have at most 2 frames with smoke >= 0.25
    overall     : PASS only if both parts pass; wrong clip counts = INCOMPLETE (not PASS)

Standalone: no Runner, Event Manager, database or LINE; nothing is written to disk.
YOLO_AUTOINSTALL is forced off. .onnx models are run with OpenCV DNN (dnn=True).

Usage (from the repo root):
    python tools/eval_smoke_clips.py --model models/smoke/fire_smoke_yolov8n_luminous0219.pt
    python tools/eval_smoke_clips.py --model A.pt --model B.pt --clips ~/smoke_eval_clips
"""

from __future__ import annotations

import os

os.environ["YOLO_AUTOINSTALL"] = "False"  # read by ultralytics at import: never auto-install

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ---- criteria: fixed before measuring ----
CONF = 0.25
POS_CLIPS, POS_MIN_DETECTED = 5, 4
NEG_MAX_FRAMES_PER_CLIP = 2
POS_PREFIX, NEG_PREFIX = "smoke_", "none_"
CLIP_SUFFIX = ".mp4"
DEFAULT_CLIPS = "~/smoke_eval_clips"


@dataclass
class ClipResult:
    name: str
    kind: str                  # "pos" (smoke_) or "neg" (none_)
    frames: int
    detected_frames: int       # frames with smoke >= CONF (one per frame)
    best_conf: float           # highest smoke confidence at >= CONF, 0 if none

    @property
    def detected(self) -> bool:
        return self.detected_frames > 0


@dataclass
class Verdict:
    pos_clips: int
    pos_detected: int
    neg_clips: int
    neg_failed: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.pos_clips != POS_CLIPS or self.neg_clips < 1:
            return "INCOMPLETE"
        ok = self.pos_detected >= POS_MIN_DETECTED and not self.neg_failed
        return "PASS" if ok else "FAIL"


def list_clips(folder: Path) -> list[Path]:
    """smoke_*.mp4 and none_*.mp4 only, sorted."""
    return sorted(p for p in Path(folder).expanduser().iterdir()
                  if p.suffix == CLIP_SUFFIX and p.name.startswith((POS_PREFIX, NEG_PREFIX)))


def smoke_class_id(names: dict) -> int:
    ids = [int(k) for k, v in names.items() if str(v).lower() == "smoke"]
    if len(ids) != 1:
        raise ValueError(f"model must have exactly one 'smoke' class, has {names}")
    return ids[0]


def evaluate_clip(path: Path, predict: Callable) -> ClipResult:
    """`predict(image)` returns [(label, confidence), ...] for one frame."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = detected = 0
    best = 0.0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            frames += 1
            confs = [c for label, c in predict(image) if str(label).lower() == "smoke" and c >= CONF]
            if confs:                       # one count per frame, however many boxes
                detected += 1
                best = max(best, max(confs))
    finally:
        cap.release()
    kind = "pos" if path.name.startswith(POS_PREFIX) else "neg"
    return ClipResult(path.name, kind, frames, detected, best)


def verdict(results: list[ClipResult]) -> Verdict:
    pos = [r for r in results if r.kind == "pos"]
    neg = [r for r in results if r.kind == "neg"]
    return Verdict(len(pos), sum(r.detected for r in pos), len(neg),
                   [r.name for r in neg if r.detected_frames > NEG_MAX_FRAMES_PER_CLIP])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_predict(model_path: Path, device: str):
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    names = dict(model.names)
    sid = smoke_class_id(names)
    onnx = model_path.suffix == ".onnx"

    def predict(image):
        r = model.predict(image, conf=CONF, classes=[sid], device="cpu" if onnx else device,
                          dnn=onnx, verbose=False)[0]
        return [(names[int(b.cls[0])], float(b.conf[0])) for b in r.boxes]

    return predict, names, sid


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure smoke models on labelled clips")
    ap.add_argument("--model", action="append", required=True, help="model file (repeat for several)")
    ap.add_argument("--clips", default=DEFAULT_CLIPS)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    folder = Path(args.clips).expanduser()
    if not folder.is_dir():
        print(f"clips folder not found: {folder}")
        return 1
    clips = list_clips(folder)
    skipped = sorted(p.name for p in folder.iterdir() if p.is_file() and p not in clips)
    print("=" * 72)
    print(f" smoke clip test | conf {CONF} | smoke clips: PASS if >= {POS_MIN_DETECTED}/{POS_CLIPS} detected "
          f"| none clips: <= {NEG_MAX_FRAMES_PER_CLIP} frames each")
    print(f" clips folder {folder}: {len(clips)} used" + (f", skipped {skipped}" if skipped else ""))
    for c in clips:
        print(f"   {c.name:<32} sha256 {sha256_file(c)[:16]}")
    print("=" * 72)

    results_by_model: dict[str, str] = {}
    for m in args.model:
        path = Path(m)
        print(f"\n### model {path}")
        if not path.is_file():
            print("   not found - skipped")
            results_by_model[str(path)] = "NOT RUN (file missing)"
            continue
        print(f"   size {path.stat().st_size:,} bytes  sha256 {sha256_file(path)}")
        try:
            predict, names, sid = make_predict(path, args.device)
        except Exception as exc:
            print(f"   cannot load: {type(exc).__name__}: {str(exc)[:200]}")
            results_by_model[str(path)] = "NOT RUN (load error)"
            continue
        print(f"   names {names} -> smoke id {sid}")
        results = []
        try:
            for c in clips:
                r = evaluate_clip(c, predict)
                results.append(r)
                print(f"   {r.name:<32} {r.kind}  frames {r.frames:>5}  smoke frames {r.detected_frames:>5}  "
                      f"best {r.best_conf:.2f}")
        except Exception as exc:
            print(f"   inference failed: {type(exc).__name__}: {str(exc)[:200]}")
            results_by_model[str(path)] = "NOT RUN (inference error)"
            continue
        v = verdict(results)
        print(f"   smoke clips detected: {v.pos_detected}/{v.pos_clips} (need >= {POS_MIN_DETECTED}/{POS_CLIPS})")
        print(f"   none clips over {NEG_MAX_FRAMES_PER_CLIP} frames: {v.neg_failed or 'none'} (of {v.neg_clips})")
        print(f"   RESULT: {v.status}")
        results_by_model[str(path)] = v.status

    print("\n" + "=" * 72)
    for m, s in results_by_model.items():
        print(f" {s:<22} {m}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
