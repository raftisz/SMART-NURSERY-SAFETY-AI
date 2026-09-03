"""
Benchmark the CAPTURE stage of a webcam, with no inference involved.

Phase 2 found that `capture` had grown into one of the largest stages of the
loop. This measures cap.read() alone under the options worth trying, so the
choice is made on numbers from the real camera rather than on theory.

It deliberately does NOT test any 4:3 resolution: 4:3 makes the model letterbox
to a LARGER tensor (640x480 instead of 640x384) and costs more inference time.
Every variant here is 16:9, so the model input stays 640x384.

Usage:
    python tools/bench_capture.py                # camera 0
    python tools/bench_capture.py --camera 1 --frames 150
"""

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# (label, width, height, buffersize, backend)
VARIANTS = [
    ("baseline 1920x1080, default buffer", 1920, 1080, None, "avfoundation"),
    ("1920x1080 + BUFFERSIZE=1", 1920, 1080, 1, "avfoundation"),
    ("1280x720, default buffer", 1280, 720, None, "avfoundation"),
    ("1280x720 + BUFFERSIZE=1", 1280, 720, 1, "avfoundation"),
    ("1280x720, CAP_ANY backend", 1280, 720, None, "any"),
]


def backend_id(name: str) -> int:
    if name == "any":
        return cv2.CAP_ANY
    if name == "avfoundation":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY


def bench(index: int, label: str, w: int, h: int, buf, backend: str, frames: int, warmup: int):
    cap = cv2.VideoCapture(index, backend_id(backend))
    if not cap.isOpened():
        print(f"  {label:38s} -> could not open camera")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    buf_note = ""
    if buf is not None:
        accepted = cap.set(cv2.CAP_PROP_BUFFERSIZE, buf)
        buf_note = "" if accepted else " (BUFFERSIZE not supported)"

    for _ in range(warmup):
        cap.read()

    times = []
    for _ in range(frames):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        times.append((time.perf_counter() - t0) * 1000.0)
        if not ok:
            break

    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    if not times:
        print(f"  {label:38s} -> no frames read")
        return None

    med = statistics.median(times)
    print(f"  {label:38s} -> {med:6.2f} ms median  "
          f"(actual {actual[0]}x{actual[1]}){buf_note}")
    return med


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark webcam capture options")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--frames", type=int, default=120, help="frames measured per variant")
    ap.add_argument("--warmup", type=int, default=25, help="frames discarded first")
    args = ap.parse_args()

    print("=" * 72)
    print(" Capture-stage benchmark (no inference)")
    print("=" * 72)
    print(f"Camera {args.camera}, {args.frames} frames per variant after {args.warmup} warm-up.")
    print("All variants are 16:9, so the model input tensor stays 640x384.\n")

    results = {}
    for label, w, h, buf, backend in VARIANTS:
        med = bench(args.camera, label, w, h, buf, backend, args.frames, args.warmup)
        if med is not None:
            results[label] = med

    if not results:
        print("\nNo variant produced a result.")
        return 1

    baseline_key = VARIANTS[0][0]
    print("\n" + "-" * 72)
    if baseline_key in results:
        base = results[baseline_key]
        print("Change vs baseline:")
        for label, med in results.items():
            if label == baseline_key:
                continue
            delta = base - med
            pct = (delta / base * 100) if base else 0.0
            verdict = "faster" if delta > 0 else "SLOWER"
            print(f"  {label:38s} {delta:+6.2f} ms ({pct:+5.1f}%) {verdict}")
    best = min(results, key=results.get)
    print(f"\nFastest variant: {best}  ({results[best]:.2f} ms)")
    print("Run this a second time to confirm - a warm machine gives different numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
