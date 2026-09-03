"""
NurseryGuard AI - Phase 2: pretrained detection test harness.

Runs a pretrained YOLO11 model over a webcam, video file, or still image and
draws person / scissors / knife detections. The point of this phase is to
measure whether the pretrained model is good enough that building a custom
dataset (Phase 3) and fine-tuning (Phase 4) can be skipped. See the Decision
Gate in README.md.

This does NOT replace main.py - that remains the Phase 1 webcam check.

Usage:
    python detect.py                        # webcam 0
    python detect.py --source 1             # webcam 1
    python detect.py --source clip.mp4      # video file
    python detect.py --source photo.jpg     # still image

Keys (in the display window):
    s        save annotated frame + append a CSV row per detection
    + / -    raise / lower the confidence threshold
    h        toggle the HUD bar
    q / ESC  quit
"""

import argparse
import csv
import platform
import statistics
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.alerts.line_notifier import LineNotifier
from src.detection.yolo_detector import YoloDetector
from src.utils.config import load_config, resolve_path

WINDOW_NAME = "NurseryGuard AI - Phase 2 Detection"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Fixed column order of docs/phase2_test_results.csv.
CSV_COLUMNS = [
    "timestamp",
    "object",
    "scenario",
    "distance_m",
    "lighting",
    "background",
    "detected(y/n)",
    "confidence",
    "bbox_w_px",
    "bbox_h_px",
    "conf_threshold",
    "imgsz",
    "capture_w",
    "capture_h",
    "capture_ms",
    "inference_ms",
    "draw_ms",
    "display_ms",
    "fps",
    "fps_gate_valid",
    "model",
    "device",
    "frame_path",
    "notes",
]

# Filled in by the person running the test, not by this program.
HUMAN_COLUMNS = ["scenario", "distance_m", "lighting", "background", "notes"]

# The Decision Gate only accepts an FPS reading after this much continuous
# running, so start-up and thermal ramp cannot flatter the number.
GATE_MIN_SECONDS = 60.0

# HUD bar sits above the video, so it can never cover a detection.
HUD_HEIGHT = 116
HUD_MIN_WIDTH = 560
HUD_BG = (28, 28, 28)

# Stages timed separately, to locate time that is not inference.
STAGES = ("capture", "inference", "draw", "display")

# How often to print a timing line during a run, so thermal drift is visible
# while testing rather than only at exit.
THERMAL_LOG_SECONDS = 60.0

# How often the two-model comparison line is printed.
COMPARE_LOG_SECONDS = 5.0

# The only keys the program acts on. '=' and '_' are deliberately NOT aliases
# for '+' and '-': every accepted key maps to exactly one action.
ACCEPTED_KEYS = frozenset({ord("s"), ord("h"), ord("q"), ord("+"), ord("-"), 27})

# Per-class box colours (BGR).
COLORS = {
    "person": (0, 200, 0),
    "scissors": (0, 0, 255),
    "knife": (0, 0, 255),
}
DEFAULT_COLOR = (255, 160, 0)


# --------------------------------------------------------------------------
# Frame sources
# --------------------------------------------------------------------------


class FrameSource:
    """Yields frames from a webcam index, a video file, or a still image.

    A still image is served repeatedly, so the confidence keys stay
    interactive on a single photo.
    """

    def __init__(self, source: str, width: int | None = None, height: int | None = None,
                 buffersize: int | None = None, backend: str = "auto"):
        self.raw = source
        self.is_webcam = source.isdigit()
        self.is_image = not self.is_webcam and Path(source).suffix.lower() in IMAGE_SUFFIXES
        self.is_video = not self.is_webcam and not self.is_image
        self._cap: cv2.VideoCapture | None = None
        self._still = None

        # Resolution bookkeeping: what was asked for, what the driver reported
        # after setting, and what the first decoded frame actually measured.
        # These three are printed at startup and must agree.
        self.requested_w = width
        self.requested_h = height
        self.reported_w = 0
        self.reported_h = 0
        self.first_frame_w = 0
        self.first_frame_h = 0
        self._fallback_resize = False
        self._fallback_announced = False
        self.backend_used = "n/a"
        self.buffersize_accepted: bool | None = None

        if self.is_webcam:
            if backend == "any":
                api = cv2.CAP_ANY
            elif backend == "avfoundation":
                api = cv2.CAP_AVFOUNDATION
            else:  # auto
                api = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
            self.backend_used = backend
            self._cap = cv2.VideoCapture(int(source), api)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Could not open camera index {source}. Is it in use by another app, "
                    "or is camera permission not granted to your terminal?"
                )
            # Set the resolution immediately after opening and before any read.
            # A camera may accept the call but keep its own size, so the value
            # is always read back rather than assumed.
            if width:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # Not supported by every backend; the return value says whether the
            # driver accepted it, so a silent no-op is reported rather than assumed.
            if buffersize is not None:
                accepted = self._cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)
                self.buffersize_accepted = bool(accepted)
                print(f"[INFO] CAP_PROP_BUFFERSIZE={buffersize} "
                      f"{'accepted' if accepted else 'NOT supported by this backend'}")
            self.reported_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.reported_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return

        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")

        if self.is_image:
            self._still = cv2.imread(str(path))
            if self._still is None:
                raise RuntimeError(f"Could not decode image: {source}")
        else:
            self._cap = cv2.VideoCapture(str(path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open video: {source}")

    @property
    def kind(self) -> str:
        if self.is_webcam:
            return "webcam"
        return "image" if self.is_image else "video"

    def read(self):
        """Return the next frame, or None when the source is exhausted.

        If the camera ignored the requested resolution, the frame is resized
        here - before the caller flips it and before inference - so everything
        downstream sees the requested size and nothing re-reads a full-size
        frame.
        """
        if self._still is not None:
            frame = self._still.copy()
        else:
            ok, frame = self._cap.read()
            if not ok:
                return None

        if self.first_frame_h == 0:
            self.first_frame_h, self.first_frame_w = frame.shape[:2]

        if self.requested_w and self.requested_h:
            h, w = frame.shape[:2]
            if (w, h) != (self.requested_w, self.requested_h):
                if not self._fallback_announced:
                    print(f"[WARN] Camera delivered {w}x{h}, not the requested "
                          f"{self.requested_w}x{self.requested_h}.")
                    print("[INFO] Fallback resize is ACTIVE - every frame is resized "
                          "before flip and before inference.")
                    self._fallback_announced = True
                    self._fallback_resize = True
                frame = cv2.resize(frame, (self.requested_w, self.requested_h),
                                   interpolation=cv2.INTER_AREA)
        return frame

    @property
    def fallback_active(self) -> bool:
        return self._fallback_resize

    def report_resolution(self) -> None:
        """Print the three resolution figures and flag any disagreement."""
        req = (f"{self.requested_w}x{self.requested_h}"
               if self.requested_w and self.requested_h else "not specified")
        rep = f"{self.reported_w}x{self.reported_h}" if self.reported_w else "n/a"
        act = (f"{self.first_frame_w}x{self.first_frame_h}"
               if self.first_frame_w else "not read yet")
        print(f"Resolution requested      : {req}")
        print(f"Resolution reported by cap: {rep}")
        print(f"First frame actual shape  : {act}")

        if not (self.requested_w and self.requested_h):
            return
        mismatches = []
        if self.reported_w and (self.reported_w, self.reported_h) != \
                (self.requested_w, self.requested_h):
            mismatches.append("driver reported a different size than requested")
        if self.first_frame_w and (self.first_frame_w, self.first_frame_h) != \
                (self.requested_w, self.requested_h):
            mismatches.append("first decoded frame differs from requested")
        if mismatches:
            print(f"[WARN] Resolution disagreement: {'; '.join(mismatches)}.")
        else:
            print("[ OK ] All three resolution figures agree.")

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


# --------------------------------------------------------------------------
# Keyboard
# --------------------------------------------------------------------------


def summarise_model(weights: str, detections, infer_ms: float) -> str:
    """One side of a [COMPARE] line: detections, best confidence, area, time."""
    name = Path(weights).stem
    if not detections:
        return f"{name}: 0 det, -, -, {infer_ms:.1f} ms"
    best = max(detections, key=lambda d: d.confidence)
    return (f"{name}: {len(detections)} det, conf {best.confidence:.2f}, "
            f"{best.area}px, {infer_ms:.1f} ms")


def clamp_conf(value: float, low: float, high: float) -> float:
    """Keep the confidence threshold inside its configured range."""
    return round(min(high, max(low, value)), 2)


def decode_key(raw: int, warned: set) -> int | None:
    """Turn a cv2.waitKey return value into an ASCII code, or None.

    waitKey returns -1 when nothing was pressed, and the Unicode CODEPOINT when
    a non-Latin keyboard layout is active. Masking such a value with 0xFF - the
    usual idiom - silently folds it onto an unrelated ASCII code. With a Thai
    layout, pressing 's' sends U+0E2B (3627); 3627 & 0xFF == 43 == '+', so the
    save key acted as "raise confidence". Codes outside ASCII are therefore
    rejected here instead of masked.
    """
    if raw is None or raw < 0:
        return None
    if raw > 127:
        if "layout" not in warned:
            warned.add("layout")
            print(f"[WARN] Ignoring a non-ASCII key (code {raw}). A non-English keyboard "
                  "layout looks active.")
            print("       Switch the input source to English/ABC, or the hotkeys "
                  "cannot be read correctly.")
        return None
    return raw


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def draw_detections(frame, detections) -> None:
    """Draw one labelled box per detection, in place."""
    for det in detections:
        color = COLORS.get(det.class_name, DEFAULT_COLOR)
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)

        label = f"{det.class_name} {det.confidence:.2f} [{det.area}px]"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        # Keep the label inside the frame on both axes. The area suffix makes
        # labels long enough to run off the right edge otherwise.
        top = max(det.y1, text_h + baseline + 2)
        left = max(0, min(det.x1, frame.shape[1] - text_w - 4))
        cv2.rectangle(
            frame,
            (left, top - text_h - baseline - 2),
            (left + text_w + 4, top),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (left + 2, top - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


def compose_with_hud(
    frame,
    conf: float,
    infer_ms: float,
    fps: float,
    n_dets: int,
    device: str,
    *,
    imgsz: int,
    elapsed: float,
    gate_valid: bool,
    stages: dict[str, float],
    suppressed: int,
):
    """Return a new image: a solid stats bar on top, the video below it.

    The bar is a separate strip rather than an overlay, so HUD text can never
    sit on top of a bounding box - the frames stay readable for scoring.
    """
    h, w = frame.shape[:2]
    canvas_w = max(w, HUD_MIN_WIDTH)
    canvas = np.zeros((h + HUD_HEIGHT, canvas_w, 3), dtype=np.uint8)
    canvas[:HUD_HEIGHT, :] = HUD_BG
    canvas[HUD_HEIGHT:, :w] = frame

    if gate_valid:
        status, status_color = "GATE-VALID", (0, 255, 0)
    else:
        status = f"warming up {elapsed:.0f}/{GATE_MIN_SECONDS:.0f}s"
        status_color = (0, 200, 255)

    dup = f"  (NMS -{suppressed})" if suppressed else ""
    lines = [
        (f"conf {conf:.2f} (+/-)   imgsz {imgsz}   {w}x{h}   [{device}]", (255, 255, 255)),
        (f"loop {fps:5.1f} FPS   detections: {n_dets}{dup}", (0, 255, 0)),
        (f"cap {stages.get('capture', 0):5.1f} | inf {stages.get('inference', 0):5.1f} | "
         f"draw {stages.get('draw', 0):5.1f} | disp {stages.get('display', 0):5.1f}  ms (median)",
         (0, 255, 255)),
        (f"FPS reading: {status}    s=save  h=hide HUD  q=quit", status_color),
    ]
    for i, (line, color) in enumerate(lines):
        cv2.putText(canvas, line, (10, 24 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Separator so the bar reads as chrome, not part of the image.
    cv2.line(canvas, (0, HUD_HEIGHT - 1), (canvas_w, HUD_HEIGHT - 1), (70, 70, 70), 1)
    return canvas


# --------------------------------------------------------------------------
# CSV logging
# --------------------------------------------------------------------------


def ensure_csv(csv_path: Path) -> None:
    """Create the results CSV with headers only if it does not exist yet."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.is_file():
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(CSV_COLUMNS)


def save_event(
    frame,
    detections,
    *,
    conf: float,
    infer_ms: float,
    fps: float,
    model: str,
    device: str,
    imgsz: int,
    capture_w: int,
    capture_h: int,
    gate_valid: bool,
    stages: dict[str, float],
    frames_dir: Path,
    csv_path: Path,
) -> Path:
    """Save the annotated frame and append one CSV row per detection.

    Everything the program can know is filled in automatically. The
    judgement columns (scenario, distance, lighting, background, notes) are
    left blank for the tester. If nothing was detected, a single row is
    written with detected(y/n)=n, so a miss is recorded rather than lost.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now()
    filename = f"frame_{stamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
    image_path = frames_dir / filename
    cv2.imwrite(str(image_path), frame)

    ensure_csv(csv_path)
    base = {col: "" for col in CSV_COLUMNS}
    base.update(
        {
            "timestamp": stamp.isoformat(timespec="seconds"),
            "conf_threshold": f"{conf:.2f}",
            "imgsz": imgsz,
            "capture_w": capture_w,
            "capture_h": capture_h,
            "capture_ms": f"{stages.get('capture', 0):.1f}",
            "inference_ms": f"{infer_ms:.1f}",
            "draw_ms": f"{stages.get('draw', 0):.1f}",
            "display_ms": f"{stages.get('display', 0):.1f}",
            "fps": f"{fps:.1f}",
            "fps_gate_valid": "y" if gate_valid else "n",
            "model": model,
            "device": device,
            "frame_path": str(image_path),
        }
    )

    rows = []
    if detections:
        for det in detections:
            row = dict(base)
            row.update(
                {
                    "object": det.class_name,
                    "detected(y/n)": "y",
                    "confidence": f"{det.confidence:.2f}",
                    "bbox_w_px": det.width,
                    "bbox_h_px": det.height,
                }
            )
            rows.append(row)
    else:
        row = dict(base)
        row["detected(y/n)"] = "n"
        rows.append(row)

    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writerows(rows)

    # Count the data rows actually in the file, so the confirmation reflects
    # what is on disk rather than what this call believes it wrote.
    with open(csv_path, "r", encoding="utf-8") as handle:
        total_rows = max(0, sum(1 for _ in handle) - 1)
    print(f"[SAVE] {image_path}  (+{len(rows)} row(s); CSV rows now: {total_rows})")
    # Box size is the evidence for the confidence-vs-object-size question, so
    # the range in this frame is surfaced at the moment of capture.
    if detections:
        areas = sorted(d.area for d in detections)
        smallest = min(detections, key=lambda d: d.area)
        largest = max(detections, key=lambda d: d.area)
        print(f"[BBOX] smallest {smallest.class_name} {areas[0]}px "
              f"({smallest.width}x{smallest.height})  |  "
              f"largest {largest.class_name} {areas[-1]}px "
              f"({largest.width}x{largest.height})")
    return image_path


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NurseryGuard AI Phase 2 - pretrained YOLO detection test"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="webcam index (default: 0), or a path to a video or image file",
    )
    parser.add_argument("--model", default=None,
                        help="weights to use, e.g. yolo11n.pt / yolo11s.pt / yolo11m.pt "
                             "(default from config.yaml). Downloaded on first use.")
    parser.add_argument("--compare-models", nargs="?", const="yolo11s.pt", default=None,
                        metavar="WEIGHTS",
                        help="also run a second model on the same frames and print a "
                             "side-by-side line every 5s. Observation only - not "
                             "gate-valid. Defaults to yolo11s.pt.")
    parser.add_argument("--line", action="store_true",
                        help="push a LINE alert when scissors or knife is detected "
                             "(off by default; credentials come from .env)")
    parser.add_argument("--line-test", action="store_true",
                        help="send one LINE test message and exit, without opening "
                             "the camera")
    parser.add_argument("--conf", type=float, default=None, help="confidence threshold")
    parser.add_argument("--device", default=None, help="cpu | mps | cuda | auto")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="model input size (default from config.yaml)")
    parser.add_argument("--width", type=int, default=None, help="capture width (webcam)")
    parser.add_argument("--height", type=int, default=None, help="capture height (webcam)")
    parser.add_argument("--display-scale", type=float, default=None,
                        help="scale the DISPLAYED window only (default 1.0). Does not "
                             "affect inference, boxes, saved frames or the CSV.")
    parser.add_argument("--buffersize", type=int, default=None,
                        help="CAP_PROP_BUFFERSIZE for the camera (try 1 to drop stale frames)")
    parser.add_argument("--backend", default=None, choices=["auto", "avfoundation", "any"],
                        help="capture backend to use for a webcam")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--flip", dest="flip", action="store_true", default=None,
                        help="force horizontal flip on")
    parser.add_argument("--no-flip", dest="flip", action="store_false",
                        help="force horizontal flip off")
    parser.add_argument("--no-display", action="store_true",
                        help="run without a window (for headless checks)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N frames (0 = unlimited)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg.get("model", {})
    det_cfg = cfg.get("detection", {})
    disp_cfg = cfg.get("display", {})
    path_cfg = cfg.get("paths", {})

    conf = args.conf if args.conf is not None else float(det_cfg.get("conf", 0.25))
    # Clamped up front so an out-of-range --conf cannot start outside the band.
    conf = clamp_conf(conf, float(det_cfg.get("conf_min", 0.05)),
                      float(det_cfg.get("conf_max", 0.95)))
    imgsz = args.imgsz if args.imgsz is not None else int(det_cfg.get("imgsz", 640))
    nms_iou = float(det_cfg.get("nms_iou", 0.5))
    display_scale = (args.display_scale if args.display_scale is not None
                     else float(disp_cfg.get("display_scale", 1.0)))
    if display_scale <= 0:
        print("[FAIL] --display-scale must be greater than 0.")
        return 1
    req_w = args.width if args.width is not None else disp_cfg.get("capture_width")
    req_h = args.height if args.height is not None else disp_cfg.get("capture_height")
    conf_step = float(det_cfg.get("conf_step", 0.05))
    conf_min = float(det_cfg.get("conf_min", 0.05))
    conf_max = float(det_cfg.get("conf_max", 0.95))
    window = int(disp_cfg.get("fps_window", 30))

    frames_dir = resolve_path(path_cfg.get("frames_dir", "data/test_frames"))
    csv_path = resolve_path(path_cfg.get("results_csv", "docs/phase2_test_results.csv"))

    alert_cfg = cfg.get("alerts", {})

    print("=" * 60)
    print(" NurseryGuard AI - Phase 2 Detection")
    print("=" * 60)

    # --line-test short-circuits before any camera or model work, so
    # credentials can be checked without hardware.
    if args.line_test:
        notifier = LineNotifier(
            cooldown_seconds=float(alert_cfg.get("line_cooldown_seconds", 30)),
            hazard_classes=alert_cfg.get("hazard_classes", ["scissors", "knife"]),
            camera_name=alert_cfg.get("camera_name", "Notebook Camera"),
        )
        print(notifier.describe())
        print("-" * 60)
        if not notifier.configured:
            print("[FAIL] Cannot send: fill LINE_CHANNEL_ACCESS_TOKEN and "
                  "LINE_USER_ID in .env (copy .env.example).")
            return 1
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ok = notifier.send_alert(
            f"✅ NurseryGuard AI - LINE test message\nTime: {stamp}\n"
            "If you can read this, credentials work."
        )
        print("-" * 60)
        print("LINE test PASSED." if ok else "LINE test FAILED - see the error above.")
        return 0 if ok else 1

    try:
        source = FrameSource(
            args.source, width=req_w, height=req_h,
            buffersize=args.buffersize if args.buffersize is not None
            else disp_cfg.get("buffersize"),
            backend=args.backend or disp_cfg.get("backend", "auto"),
        )
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    detector = YoloDetector(
        weights=args.model or model_cfg.get("weights", "yolo11n.pt"),
        target_class_names=det_cfg.get("target_classes", ["person", "scissors", "knife"]),
        device=args.device or model_cfg.get("device", "auto"),
    )
    print(detector.describe())
    if detector.missing_classes:
        print("[FAIL] Those classes are not in this model - fix target_classes in config.yaml.")
        source.release()
        return 1

    # The flip exists to un-mirror a live self-view, so it defaults on for the
    # webcam and off for files. It is applied BEFORE inference either way, so
    # boxes always line up with the displayed image.
    if args.flip is None:
        flip = bool(disp_cfg.get("flip_webcam", True)) if source.is_webcam \
            else bool(disp_cfg.get("flip_file", False))
    else:
        flip = args.flip

    # Second model for observation only. Loaded after the primary so a failure
    # here cannot be mistaken for a problem with the model under test.
    comparator = None
    if args.compare_models:
        comparator = YoloDetector(
            weights=args.compare_models,
            target_class_names=det_cfg.get("target_classes",
                                           ["person", "scissors", "knife"]),
            device=args.device or model_cfg.get("device", "auto"),
        )
        if comparator.missing_classes:
            print(f"[FAIL] {args.compare_models} lacks: "
                  f"{', '.join(comparator.missing_classes)}")
            source.release()
            return 1

    print(f"Source       : {args.source} ({source.kind})")
    if comparator is not None:
        # describe() already printed the primary model; clarify the roles here.
        print(f"Primary model: {detector.weights}  (drives display and CSV)")
        print(f"Compare with : {comparator.weights}  (observation only)")
    # Alerting is opt-in. With --line absent, behaviour is exactly as before.
    notifier = None
    if args.line:
        notifier = LineNotifier(
            cooldown_seconds=float(alert_cfg.get("line_cooldown_seconds", 30)),
            hazard_classes=alert_cfg.get("hazard_classes", ["scissors", "knife"]),
            camera_name=alert_cfg.get("camera_name", "Notebook Camera"),
        )

    print(f"Flip         : {'on' if flip else 'off'} (applied before inference)")
    print(f"Conf         : {conf:.2f}")
    print(f"imgsz        : {imgsz}")
    print(f"NMS IoU      : {nms_iou:.2f} (class-wise, on top of model NMS)")
    print(notifier.describe() if notifier else "LINE alerts  : off (use --line to enable)")
    print("-" * 60)
    if comparator is not None:
        print("[WARN] --compare-models runs TWO models on every frame.")
        print("       FPS in this mode is NOT gate-valid and must not be quoted.")
        print("       Display and CSV use the primary model only.")
    if args.no_display:
        print("[WARN] --no-display: FPS from this run is DIAGNOSTIC ONLY.")
        print("       The Decision Gate requires display ON. Do not quote this number.")
    else:
        print("Keys: s=save  +/-=confidence  h=toggle HUD  q/ESC=quit")
        print(f"Gate: FPS is only valid after {GATE_MIN_SECONDS:.0f}s of continuous run.")

    infer_times: deque[float] = deque(maxlen=window)
    loop_times: deque[float] = deque(maxlen=window)
    # One rolling window per stage, so the time that is not inference is visible.
    stage_times: dict[str, deque[float]] = {s: deque(maxlen=window) for s in STAGES}
    frame_count = 0
    last_infer_ms = 0.0
    last_fps = 0.0
    run_started = time.perf_counter()
    gate_valid = False
    capture_w = capture_h = 0
    show_hud = True
    suppressed = 0
    resolution_reported = False
    display_wh: tuple[int, int] | None = None
    key_warnings: set = set()
    next_compare_log = 0.0

    def elapsed_now() -> float:
        return time.perf_counter() - run_started
    next_thermal_log = THERMAL_LOG_SECONDS
    fps_first_minute: float | None = None

    def medians() -> dict[str, float]:
        return {s: (statistics.median(v) if v else 0.0) for s, v in stage_times.items()}

    try:
        while True:
            loop_start = time.perf_counter()

            t0 = time.perf_counter()
            frame = source.read()
            if frame is None:
                print("[INFO] End of source.")
                break
            if flip:
                frame = cv2.flip(frame, 1)
            stage_times["capture"].append((time.perf_counter() - t0) * 1000.0)

            # Report the three resolution figures once the first frame exists,
            # so the actual decoded shape can be compared against the request.
            if not resolution_reported:
                source.report_resolution()
                fh, fw = frame.shape[:2]
                win_w = int(round(max(fw, HUD_MIN_WIDTH) * display_scale))
                win_h = int(round((fh + HUD_HEIGHT) * display_scale))
                print(f"Inference input size      : {fw}x{fh} (full capture, unscaled)")
                if args.no_display:
                    print(f"Display window            : none (--no-display); "
                          f"display-scale {display_scale:.2f} has no effect")
                else:
                    print(f"Display window            : ~{win_w}x{win_h} "
                          f"(display-scale {display_scale:.2f}, display only)")
                print("-" * 60)
                resolution_reported = True

            # Recorded from the frame itself, so it reflects what was really
            # captured rather than what was requested.
            capture_h, capture_w = frame.shape[:2]

            detections, infer_ms, raw_count = detector.detect(
                frame, conf, imgsz=imgsz, nms_iou=nms_iou
            )
            suppressed = raw_count - len(detections)
            infer_times.append(infer_ms)
            stage_times["inference"].append(infer_ms)

            # Observation-only second model. Runs on the SAME frame, after the
            # primary, and never feeds the display or the CSV.
            if comparator is not None and elapsed_now() >= next_compare_log:
                cmp_dets, cmp_ms, _ = comparator.detect(
                    frame, conf, imgsz=imgsz, nms_iou=nms_iou
                )
                print(f"[COMPARE] {summarise_model(detector.weights, detections, infer_ms)} | "
                      f"{summarise_model(comparator.weights, cmp_dets, cmp_ms)}")
                next_compare_log = elapsed_now() + COMPARE_LOG_SECONDS

            # Alerting is fire-and-forget on a daemon thread; it cannot block or
            # crash the loop, and it self-suppresses via its cooldown.
            if notifier is not None:
                notifier.maybe_alert(detections)

            t0 = time.perf_counter()
            draw_detections(frame, detections)
            last_infer_ms = sum(infer_times) / len(infer_times)
            last_fps = (len(loop_times) / sum(loop_times)) if loop_times else 0.0

            elapsed = time.perf_counter() - run_started
            # Display-off runs can never satisfy the gate, whatever the elapsed time.
            gate_valid = (not args.no_display) and elapsed >= GATE_MIN_SECONDS

            stage_medians = medians()
            canvas = compose_with_hud(
                frame, conf, last_infer_ms, last_fps, len(detections), detector.device,
                imgsz=imgsz, elapsed=elapsed, gate_valid=gate_valid,
                stages=stage_medians, suppressed=suppressed,
            ) if show_hud else frame
            stage_times["draw"].append((time.perf_counter() - t0) * 1000.0)

            frame_count += 1

            if not args.no_display:
                t0 = time.perf_counter()
                # Display-only downscale. `canvas` itself is never modified, so
                # inference, boxes, saved frames and the CSV are all unaffected.
                if display_scale != 1.0:
                    shown = cv2.resize(canvas, None, fx=display_scale, fy=display_scale,
                                       interpolation=cv2.INTER_AREA)
                else:
                    shown = canvas
                if display_wh is None:
                    display_wh = (shown.shape[1], shown.shape[0])
                cv2.imshow(WINDOW_NAME, shown)
                # Exactly one waitKey per frame, decoded once into one variable.
                key = decode_key(cv2.waitKey(1), key_warnings)
                stage_times["display"].append((time.perf_counter() - t0) * 1000.0)

                if key is not None:
                    shown_char = chr(key) if 32 <= key <= 126 else f"code {key}"
                    if key in ACCEPTED_KEYS:
                        print(f"[KEY] {shown_char}")

                if key == 27:  # ESC
                    break
                if key == ord("q"):
                    break
                if key == ord("h"):
                    show_hud = not show_hud
                    print(f"[HUD] {'shown' if show_hud else 'hidden'}")
                elif key == ord("s"):
                    save_event(
                        canvas,
                        detections,
                        conf=conf,
                        infer_ms=last_infer_ms,
                        fps=last_fps,
                        model=detector.weights,
                        device=detector.device,
                        imgsz=imgsz,
                        capture_w=capture_w,
                        capture_h=capture_h,
                        gate_valid=gate_valid,
                        stages=stage_medians,
                        frames_dir=frames_dir,
                        csv_path=csv_path,
                    )
                elif key == ord("+"):
                    previous = conf
                    conf = clamp_conf(conf + conf_step, conf_min, conf_max)
                    print(f"[CONF] {previous:.2f} -> {conf:.2f}")
                elif key == ord("-"):
                    previous = conf
                    conf = clamp_conf(conf - conf_step, conf_min, conf_max)
                    print(f"[CONF] {previous:.2f} -> {conf:.2f}")

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

            loop_times.append(time.perf_counter() - loop_start)

            # Periodic timing line, so a downward FPS trend is visible during a
            # long matrix run instead of only at exit.
            if elapsed >= next_thermal_log:
                m = medians()
                print(f"[{elapsed / 60:5.1f} min] FPS {last_fps:5.1f} | "
                      f"cap {m['capture']:5.1f} | inf {m['inference']:5.1f} | "
                      f"draw {m['draw']:4.1f} | disp {m['display']:5.1f} ms"
                      + ("  <- degrading" if fps_first_minute and
                         last_fps < fps_first_minute * 0.9 else ""))
                if fps_first_minute is None:
                    fps_first_minute = last_fps
                next_thermal_log += THERMAL_LOG_SECONDS

            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        source.release()
        cv2.destroyAllWindows()

    total_elapsed = time.perf_counter() - run_started
    print("-" * 60)
    print(f"Frames processed : {frame_count}")
    print(f"Run duration     : {total_elapsed:.1f} s")
    print(f"Model            : {detector.weights}  (recorded in every CSV row)")
    if comparator is not None:
        print(f"Compared against : {comparator.weights}  (observation only, not in CSV)")
    print(f"Capture size     : {capture_w}x{capture_h}")

    # imgsz is reported from the predictor that actually ran, never from the
    # config object, so the banner and the summary cannot drift apart.
    eff = detector.effective_imgsz
    shape = detector.model_input_hw
    shape_txt = f"   model input {shape[1]}x{shape[0]}" if shape else ""
    print(f"imgsz requested  : {imgsz}")
    print(f"imgsz USED       : {eff if eff is not None else 'unknown'}{shape_txt}"
          "   (read back from predict)")
    if eff is not None and eff != imgsz:
        print(f"[FAIL] imgsz mismatch: requested {imgsz} but inference used {eff}.")
    if detector.effective_conf is not None and abs(detector.effective_conf - conf) > 1e-6:
        print(f"[FAIL] conf mismatch: current {conf:.2f} but inference used "
              f"{detector.effective_conf:.2f}.")
    if infer_times:
        print(f"Mean inference   : {sum(infer_times) / len(infer_times):.1f} ms "
              f"(rolling window of last {len(infer_times)})")
    if loop_times:
        print(f"End-to-end FPS   : {len(loop_times) / sum(loop_times):.1f} "
              f"(rolling window of last {len(loop_times)})")

    # Per-stage breakdown, to locate time that is not inference.
    final = medians()
    accounted = sum(final.values())
    print("-" * 60)
    print(f"Per-stage medians (last {window} frames):")
    for stage in STAGES:
        share = (final[stage] / accounted * 100) if accounted else 0.0
        print(f"  {stage:<10}: {final[stage]:7.2f} ms  ({share:4.1f}%)")
    print(f"  {'TOTAL':<10}: {accounted:7.2f} ms")
    if source.fallback_active:
        print("  NOTE: fallback resize was active - included in the capture stage.")

    # Say plainly whether this run may be quoted for the gate, so a diagnostic
    # number never gets copied into the results by mistake.
    if args.no_display:
        print("FPS gate status  : NOT VALID - display was off (diagnostic only)")
    elif total_elapsed < GATE_MIN_SECONDS:
        print(f"FPS gate status  : NOT VALID - ran {total_elapsed:.0f}s, "
              f"needs >= {GATE_MIN_SECONDS:.0f}s")
    else:
        print("FPS gate status  : VALID - display on, ran long enough")
    return 0


if __name__ == "__main__":
    sys.exit(main())
