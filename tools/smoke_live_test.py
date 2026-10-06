"""Live webcam test of the smoke model (luminous0219 YOLOv8n fire/smoke), standalone.

Not part of the CCTV Core: no Runner, no Event Manager, no database, no LINE, no
config change. Nothing is saved to disk. The pass criteria below are fixed BEFORE
any test and printed in the checklist; they must not be changed after seeing results.

Keys (window focused; Thai layout works too):
    1 = Test A  no smoke, 60 s              (pass: 0 frames with smoke >= 0.25)
    2 = Test B  one smoke trial of 10 s     (5 trials; pass: >= 3/5 trials with >= 1 frame)
    3 = Test C  smoke-like but not smoke, 60 s (pass: <= 2 frames)
    q / ESC = finish, print the summary and the environment checks

Usage (from the repo root, in the Terminal that has camera permission):
    python tools/smoke_live_test.py                 # webcam 0, default model (luminous0219)
    python tools/smoke_live_test.py --model models/smoke/rabahdev_best.pt
    python tools/smoke_live_test.py --check-only    # checklist only, no camera
"""

from __future__ import annotations

import os

os.environ["YOLO_AUTOINSTALL"] = "False"  # read by ultralytics at import: never auto-install

import argparse
import hashlib
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "models/smoke/fire_smoke_yolov8n_luminous0219.pt"
# verified downloads: a file with one of these names must match its SHA-256 or the test stops
KNOWN_SHA256 = {
    "fire_smoke_yolov8n_luminous0219.pt": "ac0a10257b2bc1f20c9d957f8adeeb61dd6140322fc19d0b4a116cb491776d16",
    "rabahdev_best.pt": "b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f",
    # trained on Colab (T4) from ~/smoke_data/dataset, see models/smoke/smoke_custom_yolo11n_v1.train_info.json
    "smoke_custom_yolo11n_v1.pt": "dee7fb2db1a38e2133fec43a48271aa4cb498c44fc5b08c1f296ca0d193b017d",
    # v2: dataset_v2 (v1 + Zenodo Indoor Fire Smoke), see models/smoke/smoke_custom_yolo11n_v2.train_info.json
    "smoke_custom_yolo11n_v2.pt": "2a97455c6f96252dfcedf92c149154db1b32d8eee0ef0028bb2d9c80cad95876",
}
SMOKE_NAME = "smoke"

# ---- criteria: fixed before testing, do not change after seeing results ----
CONF = 0.25
A_SECONDS, A_MAX_FRAMES = 60.0, 0      # no smoke: 0 frames with smoke >= CONF
B_TRIALS, B_TRIAL_SECONDS, B_MIN_DETECTED = 5, 10.0, 3  # smoke: >= 3 of 5 trials detect >= 1 frame
C_SECONDS, C_MAX_FRAMES = 60.0, 2      # smoke-like: <= 2 frames with smoke >= CONF
IMGSZ = 640                            # the model was trained at 640

KEYS = {  # key code -> action; Thai Kedmanee layout: 1 = ๅ, 2 = /, 3 = -, q = ๆ
    ord("1"): "A", 0x0E45: "A",
    ord("2"): "B", ord("/"): "B",
    ord("3"): "C", ord("-"): "C",
    ord("q"): "quit", 27: "quit", 0x0E46: "quit",
}


@dataclass
class TimedTest:
    """A or C: one run of `seconds`; counts frames (not boxes) with smoke >= CONF."""

    name: str
    seconds: float
    max_frames: int
    started: float | None = None
    elapsed: float = 0.0
    frames: int = 0
    detected_frames: int = 0
    done: bool = False

    @property
    def active(self) -> bool:
        return self.started is not None and not self.done

    def start(self, t: float) -> bool:
        if self.started is not None:      # only the first run counts
            return False
        self.started = t
        return True

    def on_frame(self, t: float, smoke: bool, conf: float = 0.0) -> None:
        if not self.active:
            return
        self.elapsed = t - self.started
        if self.elapsed >= self.seconds:
            self.elapsed, self.done = self.seconds, True
            return
        self.frames += 1
        self.detected_frames += smoke

    def stop(self, t: float) -> None:     # quit during the run
        if self.active:
            self.elapsed = min(t - self.started, self.seconds)
            self.done = self.elapsed >= self.seconds

    def status(self) -> str:
        if not self.done:
            return "INCOMPLETE"
        return "PASS" if self.detected_frames <= self.max_frames else "FAIL"


@dataclass
class Trial:
    """Statistics of one B trial (reporting only; the pass rule uses `hit`)."""

    duration: float = 0.0
    total_frames: int = 0
    detected_frames: int = 0
    max_conf: float = 0.0
    conf_sum: float = 0.0              # over detected frames

    @property
    def hit(self) -> bool:
        return self.detected_frames > 0

    @property
    def rate(self) -> float:
        return self.detected_frames / self.total_frames if self.total_frames else 0.0

    @property
    def mean_conf(self) -> float:
        return self.conf_sum / self.detected_frames if self.detected_frames else 0.0


@dataclass
class TrialTest:
    """B: up to `trials` separate trials of `seconds`; a trial is detected with >= 1 smoke frame."""

    trials: int
    seconds: float
    min_detected: int
    results: list[Trial] = field(default_factory=list)  # one entry per finished trial
    current_start: float | None = None
    current: Trial | None = None

    @property
    def active(self) -> bool:
        return self.current_start is not None

    @property
    def current_hit(self) -> bool:
        return bool(self.current and self.current.hit)

    def start(self, t: float) -> bool:
        if self.active or len(self.results) >= self.trials:   # trials after the 5th never count
            return False
        self.current_start, self.current = t, Trial()
        return True

    def on_frame(self, t: float, smoke: bool, conf: float = 0.0) -> None:
        """`conf` = the frame's highest smoke confidence (0 if none at >= CONF)."""
        if not self.active:
            return
        if t - self.current_start >= self.seconds:
            self.current.duration = self.seconds
            self.results.append(self.current)
            self.current_start, self.current = None, None
            return
        self.current.total_frames += 1
        if smoke:
            self.current.detected_frames += 1
            self.current.max_conf = max(self.current.max_conf, conf)
            self.current.conf_sum += conf

    def stop(self, t: float) -> None:     # an unfinished trial is dropped, not counted
        self.current_start, self.current = None, None

    @property
    def detected(self) -> int:
        return sum(r.hit for r in self.results)

    def status(self) -> str:
        if len(self.results) < self.trials:
            return "INCOMPLETE"
        return "PASS" if self.detected >= self.min_detected else "FAIL"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pip_packages() -> list[str]:
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    return sorted(line for line in out.stdout.splitlines() if line.strip())


def git_status() -> set[str]:
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                         capture_output=True, text=True)
    return set(out.stdout.splitlines())


def find_smoke_id(names: dict) -> int | None:
    """The one class named 'smoke' (any case); None if there is not exactly one."""
    ids = [int(k) for k, v in names.items() if str(v).lower() == SMOKE_NAME]
    return ids[0] if len(ids) == 1 else None


def show_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def checklist(model_path: Path, size: int, sha: str, sha_note: str, names: dict, sid: int) -> str:
    return "\n".join([
        "=" * 70,
        " SMOKE MODEL LIVE TEST - checklist (criteria fixed before testing)",
        "=" * 70,
        f" model path      : {show_path(model_path)}",
        f" file size       : {size:,} bytes",
        f" SHA-256         : {sha}  ({sha_note})",
        f" model.names     : {names}",
        f" smoke class id  : {sid} ({names.get(sid)!r}, found by name)",
        f" confidence      : {CONF}  (imgsz {IMGSZ})",
        f" A no smoke      : {A_SECONDS:.0f} s, PASS if frames with smoke >= {CONF} <= {A_MAX_FRAMES}",
        f" B smoke         : {B_TRIALS} trials x {B_TRIAL_SECONDS:.0f} s, PASS if >= {B_MIN_DETECTED}/{B_TRIALS} "
        f"trials have >= 1 frame with smoke >= {CONF} (trials after the {B_TRIALS}th do not count)",
        f" C smoke-like    : {C_SECONDS:.0f} s, PASS if frames with smoke >= {CONF} <= {C_MAX_FRAMES}",
        " counting        : per frame (several boxes in one frame = 1 frame); only the first A/C run counts",
        " overall         : PASS only if A, B and C all PASS; an unfinished test is INCOMPLETE (not PASS)",
        " keys            : 1 = A, 2 = B (one trial), 3 = C, q = finish",
        " no alerts       : no LINE, no database, no Event Manager, nothing saved to disk",
        "=" * 70,
    ])


def trial_lines(b: TrialTest) -> list[str]:
    lines = []
    for i, r in enumerate(b.results, 1):
        lines += [f"  B Trial {i}",
                  f"    duration: {r.duration:.1f}s",
                  f"    total frames: {r.total_frames}",
                  f"    detected frames: {r.detected_frames}",
                  f"    detection rate: {r.rate:.1%}",
                  f"    max confidence: {r.max_conf:.2f}" if r.hit else "    max confidence: -",
                  f"    mean confidence: {r.mean_conf:.2f}" if r.hit else "    mean confidence: -",
                  f"    result: {'DETECTED' if r.hit else 'NOT DETECTED'}"]
    return lines


def summary(a: TimedTest, b: TrialTest, c: TimedTest) -> str:
    overall = "PASS" if all(t.status() == "PASS" for t in (a, b, c)) else "FAIL"
    return "\n".join([
        "", "=" * 70, " SMOKE MODEL TEST - summary", "=" * 70,
        "A (no smoke):",
        f"  {a.detected_frames}/{a.elapsed:.0f}s frames detected  ({a.frames} frames checked)",
        f"  {a.status()}",
        "B (smoke):",
        *trial_lines(b),
        f"  {b.detected}/{len(b.results)} trials detected  (needed >= {B_MIN_DETECTED}/{B_TRIALS})",
        f"  {b.status()}",
        "C (smoke-like):",
        f"  {c.detected_frames} detected frames / {c.elapsed:.0f}s  ({c.frames} frames checked)",
        f"  {c.status()}",
        "",
        f"SMOKE MODEL TEST: {overall}",
        "=" * 70,
    ])


def draw(view, boxes, hud_lines):
    import cv2

    for (x1, y1, x2, y2), conf in boxes:
        cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)
        cv2.putText(view, f"smoke {conf:.2f}", (int(x1), max(15, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    cv2.rectangle(view, (0, 0), (view.shape[1], 30 + 24 * (len(hud_lines) - 1)), (28, 28, 28), -1)
    for i, line in enumerate(hud_lines):
        cv2.putText(view, line, (8, 21 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Live webcam test of the smoke model")
    ap.add_argument("--source", default="0", help="webcam index (default 0)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"smoke model .pt (default {DEFAULT_MODEL})")
    ap.add_argument("--check-only", action="store_true", help="print the checklist and exit")
    args = ap.parse_args()

    git_before = git_status()
    pip_before = pip_packages()

    # ---- before loading: size and SHA-256 ----
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.is_file():
        print(f"model not found: {model_path}")
        return 2
    size, sha = model_path.stat().st_size, sha256_file(model_path)
    print(f"model    : {show_path(model_path)}\nfile size: {size:,} bytes\nSHA-256  : {sha}")
    expected = KNOWN_SHA256.get(model_path.name)
    if expected is not None and sha != expected:
        print(f"SHA-256 does not match the verified download ({expected}) - stop")
        return 2
    sha_note = "matches the verified download" if expected else "WARNING: not in the verified list, check its source"
    if expected is None:
        print(sha_note)

    import torch
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = dict(model.names)
    print(f"model.names: {names}")
    sid = find_smoke_id(names)
    if sid is None:
        print(f"no single class named {SMOKE_NAME!r} in {names} - stop, no test is run")
        return 2
    print(checklist(model_path, size, sha, sha_note, names, sid))
    if args.check_only:
        return 0

    import cv2

    from src.cctv_core.stream.camera_source import CameraSource

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    source = CameraSource(args.source, 1280, 720, flip=True)
    a = TimedTest("A", A_SECONDS, A_MAX_FRAMES)
    b = TrialTest(B_TRIALS, B_TRIAL_SECONDS, B_MIN_DETECTED)
    c = TimedTest("C", C_SECONDS, C_MAX_FRAMES)
    tests = {"A": a, "B": b, "C": c}
    message, times = "press 1 / 2 / 3 to start a test, q to finish", deque(maxlen=30)
    print("camera open - waiting for you in front of the camera")
    try:
        while True:
            image = source.read()
            if image is None:
                print("camera gave no frame - finishing")
                break
            t0 = time.perf_counter()
            r = model.predict(image, conf=CONF, imgsz=IMGSZ, device=device, classes=[sid], verbose=False)[0]
            boxes = [(tuple(map(float, bx.xyxy[0])), float(bx.conf[0])) for bx in r.boxes
                     if int(bx.cls[0]) == sid and float(bx.conf[0]) >= CONF]
            smoke = bool(boxes)                       # one frame counts once, however many boxes
            best = max((conf for _, conf in boxes), default=0.0)
            now = time.monotonic()
            for test in tests.values():
                test.on_frame(now, smoke, best)
            times.append(time.perf_counter() - t0)
            fps = len(times) / sum(times) if times else 0.0

            active = next((k for k, v in tests.items() if v.active), None)
            if active == "B":
                mode = f"B trial {len(b.results) + 1}/{B_TRIALS}  {now - b.current_start:4.1f}/{B_TRIAL_SECONDS:.0f}s  " \
                       f"smoke frames {b.current.detected_frames}/{b.current.total_frames}"
            elif active:
                tt = tests[active]
                mode = f"{active}  {tt.elapsed:4.1f}/{tt.seconds:.0f}s  smoke frames {tt.detected_frames}"
            else:
                mode = "idle (not counting)"
            hud = [f"{fps:4.1f} FPS (model)  |  mode: {mode}",
                   f"A {a.detected_frames} fr ({a.status()})  B {b.detected}/{len(b.results)} trials ({b.status()})  "
                   f"C {c.detected_frames} fr ({c.status()})",
                   message]
            view = image.copy()
            draw(view, boxes, hud)
            cv2.imshow("smoke model live test", view)
            action = KEYS.get(cv2.waitKeyEx(1))
            if action == "quit":
                break
            if action in tests:
                if active:
                    message = f"test {active} is running - wait for it to finish"
                elif tests[action].start(now):
                    message = f"test {action} started"
                    print(f"[TEST] {action} started")
                else:
                    message = f"test {action} already done (only the first run counts)"
    except KeyboardInterrupt:
        print("\nstopped (Ctrl+C)")
    finally:
        now = time.monotonic()
        for test in tests.values():
            test.stop(now)
        source.release()
        cv2.destroyAllWindows()

    print(summary(a, b, c))

    # ---- environment checks after the test ----
    print("temporary files : none were written (no evidence images are saved by this script)")
    new_entries = sorted(git_status() - git_before)
    print("repo files created during the test: " + (", ".join(new_entries) if new_entries else "none"))
    subprocess.run(["git", "status", "--short"], cwd=ROOT)
    pip_after = pip_packages()
    added = sorted(set(pip_after) - set(pip_before))
    print(f"packages: {len(pip_before)} before, {len(pip_after)} after"
          + (f"  ADDED: {added}" if added else "  (none added)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
