"""CCTV Core - run one camera through every enabled module in config/core.yaml.

Usage (from the repo root):
    python run_core.py                          # camera cam0 from config, with window
    python run_core.py --modules smoking        # only some modules
    python run_core.py --source clip.mp4        # video / image / rtsp:// instead of the camera
    python run_core.py --no-display --max-frames 300
    python run_core.py --console-only           # never send LINE, even if enabled in config
    python run_core.py --fallback backup.mp4    # demo: play this clip if the camera fails

Keys in the window: q or ESC to quit (works with a Thai keyboard too).
Alerts are produced only by the Event Manager; see docs/architecture.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cctv_core.factory import DEFAULT_CORE_CONFIG, build_event_system  # noqa: E402
from src.cctv_core.runner import ModuleLoadError, Runner, build_modules  # noqa: E402
from src.cctv_core.stream.camera_source import CameraSource  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCTV Core single-camera runner")
    p.add_argument("--config", default=DEFAULT_CORE_CONFIG)
    p.add_argument("--camera", default=None, help="camera id from config (default: first)")
    p.add_argument("--source", default=None, help="override the camera source")
    p.add_argument("--modules", default=None, help="comma list of module names to run")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--console-only", action="store_true", help="disable LINE for this run")
    p.add_argument("--fallback", default=None,
                   help="backup video played (looped, marked BACKUP CLIP) if the camera fails")
    return p.parse_args()


def open_source(value, cam: dict, fallback: str | None):
    """(source, using_fallback). Opens the camera; if that fails and a backup clip
    is given, opens the clip instead. Without a backup the error is raised."""
    try:
        return CameraSource(value, cam.get("width"), cam.get("height"), cam.get("flip")), False
    except (RuntimeError, FileNotFoundError) as exc:
        if not fallback:
            raise
        print(f"[CORE] {exc}")
        print("[CORE] using the BACKUP CLIP instead (marked on screen)")
        return open_fallback(fallback, cam), True


def open_fallback(path: str, cam: dict):
    # same mirroring as the camera, so zones drawn on the camera view still line up
    return CameraSource(path, flip=cam.get("flip"), loop_video=True)


def main() -> int:
    args = parse_args()
    cfg = load_config(resolve_path(args.config))

    cameras = cfg.get("cameras") or []
    if not cameras:
        print("[CORE] no cameras in config")
        return 1
    cam = next((c for c in cameras if str(c["id"]) == args.camera), None) if args.camera else cameras[0]
    if cam is None:
        print(f"[CORE] camera '{args.camera}' not in config")
        return 1

    if args.console_only:
        cfg.setdefault("notifications", {}).setdefault("channels", {})["line"] = {"enabled": False}

    only = {m.strip() for m in args.modules.split(",")} if args.modules else None
    modules = build_modules(cfg, only)
    if not modules:
        print("[CORE] no enabled modules - check 'modules:' in config/core.yaml or --modules")
        return 1

    source_value = args.source if args.source is not None else cam.get("source", 0)
    if args.fallback and not Path(args.fallback).is_file():
        print(f"[CORE] backup clip not found: {args.fallback}")
        return 1
    source, using_fallback = open_source(source_value, cam, args.fallback)
    events = build_event_system(cfg)

    print("=" * 60)
    print(f" CCTV Core | camera {cam['id']} ({cam.get('name', '')}) | source {source.kind}: {source_value}")
    print(f" modules : {', '.join(m.detector.name for m in modules)}")
    print(f" notify  : {', '.join(events.router.channels) or 'none'}")
    print("=" * 60)

    fallback = (lambda: open_fallback(args.fallback, cam)) if args.fallback else None
    runner = Runner(source, modules, events, camera_id=str(cam["id"]),
                    display=not args.no_display, max_frames=args.max_frames,
                    fallback=fallback, using_fallback=using_fallback)
    try:
        stats = runner.run()
    except ModuleLoadError as exc:
        print(f"[CORE] {exc}")
        print("[CORE] weights not in the repo (e.g. yolo11n-pose.pt) are downloaded on first use "
              "and need internet - download them before the demo, or leave that module off")
        return 1

    print("-" * 60)
    print(f"frames {stats.frames} in {stats.seconds:.1f}s -> {stats.fps:.1f} FPS (end-to-end)")
    for name, times in stats.infer_ms.items():
        times_sorted = sorted(times)
        med = times_sorted[len(times_sorted) // 2] if times_sorted else 0.0
        print(f"  {name:<14} runs {len(times):>5}  median {med:6.1f} ms  errors {stats.errors.get(name, 0)}")
    print(f"events confirmed/suppressed this run: {stats.confirmed}")
    print("event log: data/events.db   snapshots: data/snapshots/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
