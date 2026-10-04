"""Pre-demo check: run this before every demo or rehearsal.

Checks, without starting the detection loop:
  - weights of every module that will run are on disk (no internet needed at start)
  - out_of_area has a calibrated zone for the camera
  - LINE: if enabled, the credentials are configured (values are never shown)
  - database / snapshot folders are writable and the disk has space
  - the camera opens and gives a frame; the backup clip (if any) exists and plays
  - optionally (--load-models) every model loads and runs on one frame

Usage (from the repo root):
    python tools/preflight.py
    python tools/preflight.py --modules hazard_object,climbing_pose --fallback backup.mp4 --load-models
Exit code 0 = no FAIL (WARN is allowed), 1 = at least one FAIL.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cctv_core.factory import DEFAULT_CORE_CONFIG, camera_names_from  # noqa: E402
from src.cctv_core.runner import build_modules  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

MIN_FREE_GB = 1.0


@dataclass
class Check:
    name: str
    status: str  # OK / WARN / FAIL
    detail: str


def check_weights(cfg: dict[str, Any], only: set[str] | None = None) -> list[Check]:
    checks = []
    for m in build_modules(cfg, only):
        weights = getattr(m.detector, "weights", None)
        if not weights:
            continue
        name = f"weights {m.detector.name}"
        if resolve_path(weights).is_file():
            checks.append(Check(name, "OK", str(weights)))
        else:
            checks.append(Check(name, "FAIL", f"{weights} is not on disk - Ultralytics would download "
                                              "it at start (needs internet); download it before the demo"))
    return checks


def check_zones(cfg: dict[str, Any], camera_id: str, only: set[str] | None = None) -> list[Check]:
    from src.policies.out_of_area import OutOfAreaPolicy

    checks = []
    for m in build_modules(cfg, only):
        for p in m.policies:
            if not isinstance(p, OutOfAreaPolicy):
                continue
            zones = p.zones.get(camera_id) or []
            if not zones:
                checks.append(Check("zones", "WARN", f"no zone for camera '{camera_id}' - out_of_area "
                                                     "will never alert (tools/define_zone.py)"))
                continue
            for z in zones:
                if z.child_filter and z.calibration is None:
                    checks.append(Check(f"zone {z.name}", "WARN", "not calibrated - ignored by out_of_area"))
                else:
                    kind = "child only" if z.child_filter else "any person"
                    hours = ", ".join(z.active_hours) or "always"
                    checks.append(Check(f"zone {z.name}", "OK", f"{kind}, active {hours}"))
    return checks


def check_line(cfg: dict[str, Any], line_factory: Callable[[], Any] | None = None) -> list[Check]:
    enabled = ((cfg.get("notifications") or {}).get("channels") or {}).get("line", {}).get("enabled", False)
    if not enabled:
        return [Check("LINE", "OK", "off in config (console only)")]
    if line_factory is None:
        from src.cctv_core.notifications.line_channel import LineChannel

        def line_factory():
            return LineChannel(camera_names_from(cfg))
    # only a yes/no is reported; token and user id are never printed
    if line_factory().configured:
        return [Check("LINE", "OK", "enabled, credentials present in .env")]
    return [Check("LINE", "FAIL", "enabled but .env credentials are missing (see README, LINE alerts)")]


def check_storage(cfg: dict[str, Any], min_free_gb: float = MIN_FREE_GB) -> list[Check]:
    storage = cfg.get("storage") or {}
    checks = []
    for label, rel, is_dir in (("database", storage.get("database", "data/events.db"), False),
                               ("snapshots", storage.get("snapshots_dir", "data/snapshots"), True)):
        folder = resolve_path(rel) if is_dir else resolve_path(rel).parent
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=folder):
                pass
        except OSError as exc:
            checks.append(Check(label, "FAIL", f"{folder} is not writable: {exc}"))
            continue
        free_gb = shutil.disk_usage(folder).free / 1e9
        status = "OK" if free_gb >= min_free_gb else "WARN"
        checks.append(Check(label, status, f"{folder} writable, {free_gb:.1f} GB free"))
    return checks


def check_source(value: Any, cam: dict[str, Any], open_fn: Callable[..., Any] | None = None) -> Check:
    if open_fn is None:
        from src.cctv_core.stream.camera_source import CameraSource as open_fn
    try:
        src = open_fn(value, cam.get("width"), cam.get("height"), cam.get("flip"))
    except Exception as exc:
        return Check("camera", "FAIL", str(exc))
    try:
        frame = src.read()
    finally:
        src.release()
    if frame is None:
        return Check("camera", "FAIL", f"opened {value} but got no frame")
    h, w = frame.shape[:2]
    return Check("camera", "OK", f"source {value}: {w}x{h}")


def check_fallback(path: str) -> Check:
    from src.cctv_core.stream.camera_source import CameraSource

    if not Path(path).is_file():
        return Check("backup clip", "FAIL", f"{path} not found")
    try:
        src = CameraSource(path, loop_video=True)
        frame = src.read()
        src.release()
    except Exception as exc:
        return Check("backup clip", "FAIL", f"{type(exc).__name__}: {exc}")
    if frame is None:
        return Check("backup clip", "FAIL", f"{path} gives no frame")
    return Check("backup clip", "OK", f"{path} plays ({frame.shape[1]}x{frame.shape[0]})")


def check_models_run(cfg: dict[str, Any], only: set[str] | None = None) -> list[Check]:
    """Load every model and run it once on a blank frame (catches device/weight problems)."""
    import time

    import numpy as np

    from src.cctv_core.schemas import Frame

    checks = []
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    for m in build_modules(cfg, only):
        name = f"model {m.detector.name}"
        try:
            m.detector.load()
            started = time.perf_counter()
            m.detector.process(Frame("preflight", 0, time.time(), image, 640, 480))
            checks.append(Check(name, "OK", f"loaded and ran ({(time.perf_counter() - started) * 1000:.0f} ms)"))
        except Exception as exc:
            checks.append(Check(name, "FAIL", f"{type(exc).__name__}: {exc}"))
    return checks


def exit_code(checks: list[Check]) -> int:
    return 1 if any(c.status == "FAIL" for c in checks) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-demo check for the CCTV Core")
    ap.add_argument("--config", default=DEFAULT_CORE_CONFIG)
    ap.add_argument("--camera", default=None, help="camera id (default: first)")
    ap.add_argument("--source", default=None, help="override the camera source")
    ap.add_argument("--modules", default=None, help="comma list, same as run_core.py")
    ap.add_argument("--fallback", default=None, help="backup clip that run_core.py will use")
    ap.add_argument("--no-camera", action="store_true", help="skip opening the camera")
    ap.add_argument("--load-models", action="store_true", help="also load and run every model once")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    cams = cfg.get("cameras") or [{"id": "cam0", "source": 0}]
    cam = next((c for c in cams if str(c["id"]) == args.camera), None) if args.camera else cams[0]
    if cam is None:
        print(f"camera '{args.camera}' is not in {args.config}")
        return 1
    only = {m.strip() for m in args.modules.split(",")} if args.modules else None

    checks: list[Check] = []
    modules = build_modules(cfg, only)
    checks.append(Check("modules", "OK" if modules else "FAIL",
                        ", ".join(m.detector.name for m in modules) or "none enabled"))
    checks += check_weights(cfg, only)
    checks += check_zones(cfg, str(cam["id"]), only)
    checks += check_line(cfg)
    checks += check_storage(cfg)
    if not args.no_camera:
        checks.append(check_source(args.source if args.source is not None else cam.get("source", 0), cam))
    if args.fallback:
        checks.append(check_fallback(args.fallback))
    if args.load_models:
        checks += check_models_run(cfg, only)

    print("=" * 70)
    for c in checks:
        print(f" {c.status:<4}  {c.name:<22} {c.detail}")
    print("=" * 70)
    code = exit_code(checks)
    print("PREFLIGHT " + ("PASSED" if code == 0 else "FAILED - fix every FAIL before the demo"))
    return code


if __name__ == "__main__":
    sys.exit(main())
