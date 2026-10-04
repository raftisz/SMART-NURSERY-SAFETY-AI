"""Measure the whole CCTV Core on labelled video clips (before/after numbers).

Every clip runs through the same modules, policies and Event Manager as
run_core.py, but with a clock that follows the VIDEO (frame index / fps), so
"stay 3 s in the zone" means 3 s of video however fast the Mac processes it.
LINE and console alerts are always off here; evidence images go to a temp
folder unless --keep-snapshots is given.

Clip names (written by QA, see docs/eval/qa_shooting_script.md):
    <module>_<pos|neg>_<session>_<nn>[_note].mp4
    module : hazard | smoking | fight | climbing | outarea | normal (normal is always neg)
    session: s1, s2, ...   e.g. fight_pos_s2_03_push.mp4

Usage (from the repo root):
    # 1. draft a labels CSV from the file names (check it, then commit it)
    python tools/eval_clips.py --init-labels ~/qa_clips --labels docs/eval/qa_labels.csv --tune-sessions s1
    # 2. measure (writes docs/eval/<tag>_clips.csv and docs/eval/<tag>_summary.md)
    python tools/eval_clips.py --labels docs/eval/qa_labels.csv --split test --tag before

Scoring per event type T, one row per clip:
    clip expects T and T fired -> TP;  expects T, not fired -> FN
    clip does not expect T but T fired -> FP clip (+ every such event counts toward FP/hour)
Processing FPS here is NOT gate-valid (no display, file input).

Decision gates (docs/eval/decision_gates_outarea_fight.md) are judged on the
test split only, per clip subgroup = the first word after the clip number,
e.g. outarea_pos_s2_03_crouchstay_door.mp4 -> outarea / pos / crouchstay.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cctv_core.detector_base import BaseDetector  # noqa: E402
from src.cctv_core.factory import DEFAULT_CORE_CONFIG, build_event_system  # noqa: E402
from src.cctv_core.runner import ModuleLoadError, Runner, build_modules  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
MODULE_TYPES: dict[str, str | None] = {
    "hazard": "hazard_object",
    "smoking": "smoking",
    "fight": "fight",
    "climbing": "climbing",
    "outarea": "out_of_area",
    "normal": None,
}
DEFAULT_FPS = 30.0  # used only when a file reports no frame rate


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


@dataclass
class Clip:
    path: str
    split: str                      # "tune" or "test"
    expected: set[str] = field(default_factory=set)  # event types that SHOULD fire
    note: str = ""


def parse_clip_name(name: str) -> tuple[str | None, str, str]:
    """'fight_pos_s1_01_hug.mp4' -> ('fight', 'pos', 's1'). Raises ValueError."""
    parts = Path(name).stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"{name}: expected <module>_<pos|neg>_<session>_<nn>")
    module, polarity, session = parts[0].lower(), parts[1].lower(), parts[2].lower()
    if module not in MODULE_TYPES:
        raise ValueError(f"{name}: unknown module '{module}' ({', '.join(MODULE_TYPES)})")
    if polarity not in ("pos", "neg"):
        raise ValueError(f"{name}: second part must be pos or neg")
    if module == "normal" and polarity == "pos":
        raise ValueError(f"{name}: 'normal' clips are always neg")
    if not (session.startswith("s") and session[1:].isdigit()):
        raise ValueError(f"{name}: session must look like s1, s2, ...")
    return MODULE_TYPES[module], polarity, session


def init_labels(clips_dir: Path, tune_sessions: set[str]) -> list[Clip]:
    """Draft labels from file names. Files that do not follow the naming rule are reported."""
    clips: list[Clip] = []
    for path in sorted(Path(clips_dir).iterdir()):
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        try:
            event_type, polarity, session = parse_clip_name(path.name)
        except ValueError as exc:
            print(f"[LABELS] skipped {exc}")
            continue
        expected = {event_type} if polarity == "pos" and event_type else set()
        clips.append(Clip(str(path), "tune" if session in tune_sessions else "test", expected))
    return clips


def write_labels(clips: list[Clip], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["path", "split", "expected", "note"])
        for c in clips:
            w.writerow([c.path, c.split, ";".join(sorted(c.expected)) or "none", c.note])


def read_labels(path: Path) -> list[Clip]:
    """Relative clip paths are resolved against the CSV's folder."""
    path = Path(path)
    clips: list[Clip] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            clip_path = Path(row["path"]).expanduser()
            if not clip_path.is_absolute():
                clip_path = path.parent / clip_path
            expected = {t.strip() for t in row["expected"].split(";") if t.strip() and t.strip() != "none"}
            clips.append(Clip(str(clip_path), row["split"].strip(), expected, row.get("note") or ""))
    return clips


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


class VideoClock:
    """Timestamp of frame n = start + n / fps (one call per frame)."""

    def __init__(self, fps: float, start: float):
        self.fps, self.start, self._n = float(fps), float(start), 0

    def __call__(self) -> float:
        t = self.start + self._n / self.fps
        self._n += 1
        return t


class _Preloaded(BaseDetector):
    """Reuses an already loaded detector so each clip does not reload weights."""

    def __init__(self, inner: BaseDetector):
        super().__init__(name=inner.name, run_every_n_frames=inner.run_every_n_frames)
        self.inner = inner

    def load(self) -> None:
        pass

    def process(self, frame):
        return self.inner.process(frame)


def eval_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Copy of cfg with every notification channel off (never send LINE while measuring)."""
    out = copy.deepcopy(cfg)
    channels = out.setdefault("notifications", {}).setdefault("channels", {})
    channels["console"] = {"enabled": False}
    channels["line"] = {"enabled": False}
    return out


def start_timestamp(hhmm: str) -> float:
    """Today at HH:MM local time (zones with active_hours depend on the clock)."""
    h, m = (int(v) for v in hhmm.split(":"))
    return datetime.now().replace(hour=h, minute=m, second=0, microsecond=0).timestamp()


@dataclass
class ClipResult:
    clip: Clip
    fired: dict[str, int]           # event type -> confirmed/suppressed events
    duration_s: float
    frames: int
    errors: dict[str, int] = field(default_factory=dict)


class Evaluator:
    def __init__(self, cfg: dict[str, Any], camera_id: str | None = None,
                 only: set[str] | None = None, start: str = "10:00",
                 snapshots_dir: str | None = None):
        self.cfg = eval_config(cfg)
        cams = self.cfg.get("cameras") or [{"id": "cam0"}]
        cam = next((c for c in cams if str(c["id"]) == camera_id), None) if camera_id else cams[0]
        if cam is None:
            raise ValueError(f"camera '{camera_id}' is not in the config")
        self.camera = cam
        self.only = only
        self.start = start_timestamp(start)
        self._tmp = None if snapshots_dir else tempfile.TemporaryDirectory()
        self.snapshots_dir = snapshots_dir or self._tmp.name
        self._detectors: list[BaseDetector] | None = None
        self.load_count = 0

    def _modules(self):
        modules = build_modules(self.cfg, self.only)  # fresh policies (tracker state) per clip
        if self._detectors is None:
            for m in modules:
                try:
                    m.detector.load()
                except Exception as exc:
                    raise ModuleLoadError(m.detector.name, exc) from exc
            self._detectors = [m.detector for m in modules]
            self.load_count += 1
        for m, loaded in zip(modules, self._detectors):
            m.detector = _Preloaded(loaded)
        return modules

    def model_info(self) -> list[str]:
        lines = []
        for d in self._detectors or []:
            info = getattr(d, "model_info", None)
            if info is not None:
                lines.append(f"{d.name}: {info.weights} sha256 {(info.sha256 or 'n/a')[:16]}")
            else:
                lines.append(f"{d.name}: (no model info)")
        return lines

    def run_clip(self, clip: Clip) -> ClipResult:
        from src.cctv_core.stream.camera_source import CameraSource

        source = CameraSource(clip.path, flip=bool(self.camera.get("flip", False)), loop_image=False)
        fps = source.fps or DEFAULT_FPS
        events = build_event_system(self.cfg, database=":memory:", snapshots_dir=self.snapshots_dir,
                                    async_notifications=False)
        store = events.store
        # keep the store open after the run so it can be read
        events.shutdown = lambda now: (events.manager.close_all(now), events.router.close())
        runner = Runner(source, self._modules(), events, camera_id=str(self.camera["id"]),
                        display=False, clock=VideoClock(fps, self.start))
        stats = runner.run()
        fired = Counter(e["event_type"] for e in store.list_events(limit=1_000_000))
        store.close()
        return ClipResult(clip, dict(fired), stats.frames / fps, stats.frames, dict(stats.errors))

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()


# --------------------------------------------------------------------------
# Decision gates (approved 2026-10-04, before any s2 result)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateRule:
    text: str
    module: str        # clip-name module, e.g. "outarea"
    polarity: str      # "pos" / "neg"
    group: str         # clip-name subgroup
    event_type: str
    count: str         # what is counted per clip: "alerted" or "silent"
    op: str            # ">=" or "<="
    k: int
    n: int             # exact number of clips the gate is defined on


GATES: tuple[GateRule, ...] = (
    GateRule("out_of_area: crouching in zone > 3 s alerts", "outarea", "pos", "crouchstay",
             "out_of_area", "alerted", ">=", 8, 10),
    GateRule("out_of_area: adult upright in zone, false alarms", "outarea", "neg", "adultstand",
             "out_of_area", "alerted", "<=", 1, 10),
    GateRule("out_of_area: quick crouching pass < 2 s, no alert", "outarea", "neg", "crouchpass",
             "out_of_area", "silent", ">=", 9, 10),
    GateRule("fight: staged fight alerts", "fight", "pos", "staged", "fight", "alerted", ">=", 4, 6),
    GateRule("fight: everyday activity, false alarms", "fight", "neg", "daily", "fight", "alerted", "<=", 2, 8),
)


def clip_tags(path: str) -> tuple[str, str, str] | None:
    """('outarea', 'pos', 'crouchstay') from the file name, None if it has no subgroup."""
    parts = Path(path).stem.lower().split("_")
    if len(parts) < 5:
        return None
    return parts[0], parts[1], parts[4]


@dataclass
class GateResult:
    rule: GateRule
    clips: int
    counted: int       # clips matching rule.count
    ran: bool = True   # False: no module in this run produces rule.event_type

    @property
    def status(self) -> str:
        if not self.ran:
            return "NOT RUN"
        if self.clips < self.rule.n:
            return "INCOMPLETE"
        if self.clips > self.rule.n:
            return "CHECK"     # the gate is defined on exactly n clips
        ok = self.counted >= self.rule.k if self.rule.op == ">=" else self.counted <= self.rule.k
        return "PASS" if ok else "FAIL"


def evaluate_gates(results: list[ClipResult], available: set[str] | None = None) -> list[GateResult]:
    """`available` = event types the run's policies can produce (None = assume all)."""
    out = []
    for rule in GATES:
        clips = [r for r in results if r.clip.split == "test"
                 and clip_tags(r.clip.path) == (rule.module, rule.polarity, rule.group)]
        alerted = sum(1 for r in clips if r.fired.get(rule.event_type, 0) > 0)
        counted = alerted if rule.count == "alerted" else len(clips) - alerted
        ran = available is None or rule.event_type in available
        out.append(GateResult(rule, len(clips), counted, ran))
    return out


# --------------------------------------------------------------------------
# Scoring and reports
# --------------------------------------------------------------------------


@dataclass
class TypeScore:
    tp: int = 0
    fn: int = 0
    fp_clips: int = 0
    neg_clips: int = 0
    fp_events: int = 0
    neg_seconds: float = 0.0

    @property
    def fp_per_hour(self) -> float:
        return self.fp_events / (self.neg_seconds / 3600.0) if self.neg_seconds > 0 else 0.0


def score(results: list[ClipResult]) -> dict[str, TypeScore]:
    types = sorted({t for r in results for t in r.clip.expected} | {t for r in results for t in r.fired})
    scores = {t: TypeScore() for t in types}
    for r in results:
        for t, s in scores.items():
            fired = r.fired.get(t, 0)
            if t in r.clip.expected:
                if fired:
                    s.tp += 1
                else:
                    s.fn += 1
            else:
                s.neg_clips += 1
                s.neg_seconds += r.duration_s
                if fired:
                    s.fp_clips += 1
                    s.fp_events += fired
    return scores


def write_report(results: list[ClipResult], scores: dict[str, TypeScore], out_dir: Path,
                 tag: str, meta: list[str], gates: list[GateResult] | None = None) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_csv = out_dir / f"{tag}_clips.csv"
    with open(clips_csv, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["clip", "split", "expected", "fired", "duration_s", "frames", "errors"])
        for r in results:
            w.writerow([Path(r.clip.path).name, r.clip.split,
                        ";".join(sorted(r.clip.expected)) or "none",
                        ";".join(f"{t}:{n}" for t, n in sorted(r.fired.items())) or "none",
                        f"{r.duration_s:.1f}", r.frames,
                        ";".join(f"{k}:{v}" for k, v in sorted(r.errors.items()))])

    summary = out_dir / f"{tag}_summary.md"
    lines = [f"# Clip evaluation: {tag}", ""] + [f"- {m}" for m in meta] + [
        f"- clips: {len(results)}, video time {sum(r.duration_s for r in results) / 60:.1f} min",
        "",
        "| event type | detected (TP / positive clips) | false alarm clips (FP / negative clips) "
        "| false alarm events | FP per hour of negative video |",
        "|---|---|---|---|---|",
    ]
    for t, s in scores.items():
        pos = s.tp + s.fn
        lines.append(f"| {t} | {s.tp}/{pos} | {s.fp_clips}/{s.neg_clips} | {s.fp_events} | "
                     f"{s.fp_per_hour:.1f} (on {s.neg_seconds / 60:.1f} min) |")
    if gates:
        lines += ["", "## Decision gates (docs/eval/decision_gates_outarea_fight.md, test split only)", "",
                  "| criterion | clips | counted | needed | result |", "|---|---|---|---|---|"]
        for g in gates:
            sign = "≥" if g.rule.op == ">=" else "≤"
            lines.append(f"| {g.rule.text} | {g.clips}/{g.rule.n} | {g.counted}/{g.clips} {g.rule.count} | "
                         f"{sign} {g.rule.k}/{g.rule.n} | **{g.status}** |")
        lines += ["", "INCOMPLETE = fewer clips than the gate needs (not a pass); "
                  "CHECK = more clips than the gate is defined on; "
                  "NOT RUN = the module for this event type was off in this run."]
    lines += ["", "Counts are clips, not percentages: small numbers, read them as such.",
              "Processing speed in this run is not gate-valid (file input, no display).", ""]
    summary.write_text("\n".join(lines), encoding="utf-8")
    return clips_csv, summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure the CCTV Core on labelled clips")
    ap.add_argument("--labels", required=True, help="labels CSV (written by --init-labels)")
    ap.add_argument("--init-labels", metavar="CLIPS_DIR", help="draft the labels CSV from file names")
    ap.add_argument("--tune-sessions", default="s1", help="sessions used for tuning (comma list)")
    ap.add_argument("--split", default="test", choices=["test", "tune", "all"])
    ap.add_argument("--config", default=DEFAULT_CORE_CONFIG)
    ap.add_argument("--modules", default=None, help="comma list of module names (like run_core.py)")
    ap.add_argument("--camera", default="qa", help="camera id in the config whose zones/flip apply")
    ap.add_argument("--start", default="10:00", help="clock time the clips are treated as starting at")
    ap.add_argument("--tag", default="eval", help="name of the result files")
    ap.add_argument("--out", default="docs/eval")
    ap.add_argument("--keep-snapshots", default=None, help="folder to keep evidence images in")
    args = ap.parse_args()

    labels = resolve_path(args.labels)
    if args.init_labels:
        if labels.exists():
            print(f"[LABELS] {labels} already exists - not overwritten")
            return 1
        tune = {s.strip().lower() for s in args.tune_sessions.split(",") if s.strip()}
        clips = init_labels(Path(args.init_labels).expanduser(), tune)
        write_labels(clips, labels)
        print(f"[LABELS] {len(clips)} clips -> {labels} (check it before measuring)")
        return 0

    clips = [c for c in read_labels(labels) if args.split == "all" or c.split == args.split]
    if not clips:
        print(f"[EVAL] no '{args.split}' clips in {labels}")
        return 1
    cfg = load_config(resolve_path(args.config))
    only = {m.strip() for m in args.modules.split(",")} if args.modules else None
    try:
        ev = Evaluator(cfg, camera_id=args.camera, only=only, start=args.start,
                       snapshots_dir=args.keep_snapshots)
    except ValueError as exc:
        print(f"[EVAL] {exc}")
        return 1

    results: list[ClipResult] = []
    try:
        for i, clip in enumerate(clips, 1):
            print(f"[EVAL] {i}/{len(clips)} {Path(clip.path).name}")
            results.append(ev.run_clip(clip))
    except ModuleLoadError as exc:
        print(f"[EVAL] {exc}")
        return 1
    finally:
        ev.close()

    meta = [f"date: {datetime.now():%Y-%m-%d %H:%M}", f"config: {args.config}",
            f"labels: {args.labels} (split {args.split})", f"camera: {args.camera}"] + ev.model_info()
    produced = {getattr(p, "event_type", None) for m in build_modules(ev.cfg, only) for p in m.policies}
    gates = evaluate_gates(results, available=produced) if args.split == "test" else None
    clips_csv, summary = write_report(results, score(results), resolve_path(args.out), args.tag, meta,
                                      gates=gates)
    print(summary.read_text(encoding="utf-8"))
    print(f"[EVAL] wrote {clips_csv} and {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
