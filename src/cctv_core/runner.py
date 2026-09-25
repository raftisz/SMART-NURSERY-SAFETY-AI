"""Single-camera run loop: source -> detectors -> policies -> Event Manager.

The loop never sends notifications itself; confirmed events reach channels
only through the Event Manager and NotificationRouter.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from .detector_base import BaseDetector, EventPolicy
from .registry import build_detector, build_policy
from .schemas import DetectorOutput, Frame

# q or ESC quits. 3654 is 'ๆ', the q key on a Thai keyboard layout.
QUIT_KEYS = {ord("q"), 27, 3654}

COLORS = {
    "person": (0, 200, 0),
    "scissors": (0, 0, 255),
    "knife": (0, 0, 255),
    "smoking": (0, 140, 255),
    "cigarette": (0, 140, 255),
}
DEFAULT_COLOR = (255, 160, 0)


@dataclass
class Module:
    detector: BaseDetector
    policy: EventPolicy


def build_modules(cfg: dict[str, Any], only: set[str] | None = None) -> list[Module]:
    """Build enabled modules from config (weights are NOT loaded here)."""
    modules: list[Module] = []
    for entry in cfg.get("modules") or []:
        if entry.get("enabled", True) is False:
            continue
        module = Module(build_detector(entry["detector"]), build_policy(entry["policy"]))
        if only and module.detector.name not in only:
            continue
        modules.append(module)
    return modules


@dataclass
class RunStats:
    frames: int = 0
    seconds: float = 0.0
    infer_ms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    confirmed: int = 0

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


def draw_output(image: Any, output: DetectorOutput) -> None:
    import cv2

    for d in output.detections:
        color = COLORS.get(d.label.lower(), DEFAULT_COLOR)
        b = d.bbox
        p1, p2 = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
        cv2.rectangle(image, p1, p2, color, 2)
        text = f"{d.label} {d.confidence:.2f}"
        status = d.attributes.get("pose_status")
        if status:
            text += f" {status}"
        cv2.putText(image, text, (p1[0], max(15, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        for k in d.keypoints or []:
            if k.confidence > 0.3:
                cv2.circle(image, (int(k.x), int(k.y)), 3, color, -1)


class Runner:
    def __init__(
        self,
        source: Any,
        modules: list[Module],
        events: Any,  # factory.EventSystem
        camera_id: str = "cam0",
        display: bool = True,
        max_frames: int = 0,
        clock: Callable[[], float] = time.time,
        window_name: str = "CCTV Core",
    ):
        self.source = source
        self.modules = modules
        self.events = events
        self.camera_id = camera_id
        self.display = display
        self.max_frames = max_frames
        self.clock = clock
        self.window_name = window_name
        self.stats = RunStats()
        self._last_output: dict[str, DetectorOutput] = {}
        self._error_reported: set[str] = set()
        self._start_perf = time.perf_counter()

    def run(self) -> RunStats:
        for m in self.modules:
            m.detector.load()
        started = time.perf_counter()
        self._start_perf = started
        try:
            self._loop()
        except KeyboardInterrupt:
            print("\n[CORE] stopped (Ctrl+C)")
        finally:
            self.stats.seconds = time.perf_counter() - started
            self.source.release()
            for m in self.modules:
                m.detector.close()
            if self.display:
                import cv2
                cv2.destroyAllWindows()
            self.events.shutdown(self.clock())
        return self.stats

    def _loop(self) -> None:
        frame_index = 0
        while True:
            image = self.source.read()
            if image is None:
                print("[CORE] source ended")
                break
            now = self.clock()
            h, w = image.shape[:2] if hasattr(image, "shape") else (0, 0)
            frame = Frame(self.camera_id, frame_index, now, image, w, h)

            for m in self.modules:
                if not m.detector.should_run(frame_index):
                    continue
                self._run_module(m, frame)

            for closed in self.events.manager.tick(now):
                print(f"[EVENT] closed {closed.event_type} @ {closed.camera_id} "
                      f"(hits {closed.hit_count}, peak {closed.peak_confidence:.2f})")

            self.stats.frames += 1
            frame_index += 1
            if self.display and self._show(image):
                break
            if self.max_frames and frame_index >= self.max_frames:
                break

    def _run_module(self, m: Module, frame: Frame) -> None:
        name = m.detector.name
        try:
            output = m.detector.process(frame)
            candidates = m.policy.evaluate(output)
        except Exception as exc:  # one broken module must not stop the others
            self.stats.errors[name] += 1
            if name not in self._error_reported:
                print(f"[CORE] {name} error {type(exc).__name__}: {exc} (further errors counted only)")
                self._error_reported.add(name)
            return
        self.stats.infer_ms[name].append(output.inference_ms)
        self._last_output[name] = output
        if not candidates:
            return
        evidence = None
        if hasattr(frame.image, "copy"):
            evidence = frame.image.copy()
            draw_output(evidence, output)
        for c in candidates:
            event = self.events.manager.submit(c, image=evidence)
            if event is not None:
                self.stats.confirmed += 1
                print(f"[EVENT] {event.status.value} {event.event_type} @ {event.camera_id} "
                      f"(peak {event.peak_confidence:.2f})")

    def _show(self, image: Any) -> bool:
        """Draw the latest output of every module and a status bar. True = quit."""
        import cv2

        view = image.copy()
        for output in self._last_output.values():
            draw_output(view, output)
        elapsed = max(1e-6, time.perf_counter() - self._start_perf)
        parts = [f"{self.stats.frames / elapsed:.1f} FPS"]
        for name, times in self.stats.infer_ms.items():
            recent = times[-30:]
            parts.append(f"{name} {statistics.median(recent):.0f}ms")
        parts.append(f"open events {len(self.events.manager.open_events)}")
        cv2.rectangle(view, (0, 0), (view.shape[1], 28), (28, 28, 28), -1)
        cv2.putText(view, " | ".join(parts), (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
        cv2.imshow(self.window_name, view)
        key = cv2.waitKeyEx(1)
        return key in QUIT_KEYS
