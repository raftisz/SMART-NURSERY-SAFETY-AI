"""
NurseryGuard AI - Phase 1 environment check.

Verifies that:
  1. The Python environment and core packages work.
  2. OpenCV can open the webcam.
  3. Frames can be read and displayed in a window.

No detection, no risk analysis - that comes in later phases.

Usage:
    python main.py             # use default camera (index 0)
    python main.py --camera 1  # use another camera
"""

import argparse
import platform
import sys

import cv2
import numpy as np

WINDOW_NAME = "NurseryGuard AI - Phase 1 Webcam Test"


def print_environment() -> None:
    """Check 1: report the Python environment and package versions."""
    print("=" * 55)
    print(" NurseryGuard AI - Phase 1 Environment Check")
    print("=" * 55)
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Interpreter : {sys.executable}")
    print(f"Platform    : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"OpenCV      : {cv2.__version__}")
    print(f"NumPy       : {np.__version__}")
    print("-" * 55)


def open_camera(index: int) -> cv2.VideoCapture:
    """Check 2: open the webcam and return the capture handle.

    On macOS the AVFoundation backend is used explicitly, since the default
    backend can silently fail to deliver frames.
    """
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)

    if not cap.isOpened():
        print(f"[FAIL] Could not open camera at index {index}.")
        print("       - Is another app (Zoom, Photo Booth, Teams) using the camera?")
        print("       - On macOS, grant camera access to your terminal in")
        print("         System Settings > Privacy & Security > Camera, then restart it.")
        print("       - Try a different index: python main.py --camera 1")
        return cap

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[ OK ] Camera {index} opened: {width}x{height} @ {fps:.1f} FPS")
    return cap


def show_stream(cap: cv2.VideoCapture) -> bool:
    """Check 3: read frames and display them until the user quits.

    Returns True if at least one frame was displayed successfully.
    """
    print("[INFO] Showing webcam feed. Press 'q' or ESC in the window to quit.")
    frames_shown = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[FAIL] Camera opened but no frame could be read.")
            break

        # Flip horizontally so the preview is not mirrored. This must happen
        # before anything is drawn on the frame, otherwise the overlay would be
        # reversed too - and it keeps the displayed image and any future
        # detection coordinates in the same orientation.
        frame = cv2.flip(frame, 1)

        frames_shown += 1
        cv2.putText(
            frame,
            f"Phase 1 OK - frame {frames_shown} - press 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 'q' or ESC
            break

        # The window was closed with the title-bar button.
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    if frames_shown:
        print(f"[ OK ] Displayed {frames_shown} frames.")
    return frames_shown > 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NurseryGuard AI Phase 1 webcam test")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    args = parser.parse_args()

    print_environment()

    cap = open_camera(args.camera)
    if not cap.isOpened():
        return 1

    try:
        displayed = show_stream(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("-" * 55)
    if displayed:
        print("PHASE 1 PASSED - environment and webcam are working.")
        return 0

    print("PHASE 1 FAILED - camera opened but no frames were displayed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
