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


class ModuleLoadError(RuntimeError):
    """A detector could not load its model (missing weights, no internet to download...)."""

    def __init__(self, name: str, cause: BaseException):
        super().__init__(f"module '{name}' failed to load: {type(cause).__name__}: {cause}")
        self.module_name = name


@dataclass
class Module:
    """One detector and the rule(s) that read its output.

    `policy` may be a single EventPolicy or a list, so one model run can feed
    several rules (e.g. hazard objects AND out-of-area from the same person boxes).
    """

    detector: BaseDetector
    policy: EventPolicy | list[EventPolicy]

    @property
    def policies(self) -> list[EventPolicy]:
        return list(self.policy) if isinstance(self.policy, (list, tuple)) else [self.policy]


def build_modules(cfg: dict[str, Any], only: set[str] | None = None) -> list[Module]:
    """Build enabled modules from config (weights are NOT loaded here)."""
    modules: list[Module] = []
    for entry in cfg.get("modules") or []:
        disabled = entry.get("enabled", True) is False
        if disabled and not only:
            continue
        specs = entry.get("policies")
        if specs is None:
            specs = [entry["policy"]]
        module = Module(build_detector(entry["detector"]), [build_policy(p) for p in specs])
        # --modules picks modules by name; naming a disabled one turns it on for this run
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
        for key in ("pose_status", "tag"):
            if d.attributes.get(key):
                text += f" {d.attributes[key]}"
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
        """Load every model, then loop. Raises ModuleLoadError (after cleaning up)."""
        started = time.perf_counter()
        try:
            for m in self.modules:
                try:
                    m.detector.load()
                except Exception as exc:
                    raise ModuleLoadError(m.detector.name, exc) from exc
            started = time.perf_counter()
            self._start_perf = started
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
            if not output.frame_width:
                output.frame_width, output.frame_height = frame.width, frame.height
            candidates = [c for p in m.policies for c in p.evaluate(output)]
        except Exception as exc:  # one broken module must not stop the others
            self.stats.errors[name] += 1
            self._last_output.pop(name, None)  # never keep drawing boxes from before the error
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
                where = f" [{event.subject}]" if event.subject else ""
                print(f"[EVENT] {event.status.value} {event.event_type}{where} @ {event.camera_id} "
                      f"(peak {event.peak_confidence:.2f})")

    def status_text(self) -> str:
        """Status bar: FPS, per-module time, module error counts, open events."""
        elapsed = max(1e-6, time.perf_counter() - self._start_perf)
        parts = [f"{self.stats.frames / elapsed:.1f} FPS"]
        for name, times in self.stats.infer_ms.items():
            recent = times[-30:]
            parts.append(f"{name} {statistics.median(recent):.0f}ms")
        for name, count in self.stats.errors.items():
            parts.append(f"{name} ERR {count}")
        parts.append(f"open events {len(self.events.manager.open_events)}")
        return " | ".join(parts)

    def _show(self, image: Any) -> bool:
        """Draw the latest output of every module and a status bar. True = quit."""
        import cv2

        view = image.copy()
        for m in self.modules:
            for p in m.policies:
                p.draw(view, self.camera_id)
        for output in self._last_output.values():
            draw_output(view, output)
        cv2.rectangle(view, (0, 0), (view.shape[1], 28), (28, 28, 28), -1)
        cv2.putText(view, self.status_text(), (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
        cv2.imshow(self.window_name, view)
        key = cv2.waitKeyEx(1)
        return key in QUIT_KEYS
