"""Smoking verifier (yolo11n-cls) on the Mendeley Smoker Detection dataset.

The verifier is a second check: an alert needs the smoking DETECTOR to fire
(the same rule as config/core.yaml) AND the classifier to say "smoking" with
probability >= threshold. Mendeley images are whole-person photos, so the
classifier sees the whole image here.

Splits are used strictly as follows:
    Training   -> training only (on Colab, notebooks/smoking_cls_colab.ipynb)
    Validation -> `select`: choose the threshold (rule fixed below, before any result)
    Testing    -> `report`: the before/after numbers; the threshold is read from the
                  file `select` wrote and can not be given by hand

Threshold rule (fixed in advance): among thresholds 0.05..0.95, keep those whose
recall on Validation is at most `--max-recall-drop` (default 0.05) below the
detector alone; pick the one with the fewest false alarms (ties -> the lower
threshold). Adopt only if it has fewer false alarms than the detector alone.

Usage (from the repo root; DATASET = the folder holding Training/Validation/Testing):
    python tools/eval_smoking_cls.py dupes  --dataset DATASET
    python tools/eval_smoking_cls.py select --dataset DATASET --classifier models/smoking_cls/smoking_cls_v1.pt
    python tools/eval_smoking_cls.py report --dataset DATASET --classifier models/smoking_cls/smoking_cls_v1.pt
Results go to docs/eval/smoking_cls_*.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import resolve_path  # noqa: E402

OUT_DIR = "docs/eval"
THRESHOLD_FILE = "docs/eval/smoking_cls_threshold.json"
CANDIDATES = [round(0.05 * i, 2) for i in range(1, 20)]


# --------------------------------------------------------------------------
# Pure helpers (unit tested)
# --------------------------------------------------------------------------


def label_from_name(name: str) -> int:
    """Mendeley file names: smoking_XXXX.jpg = 1, notsmoking_XXXX.jpg = 0."""
    stem = Path(name).name.lower()
    if stem.startswith("notsmoking_"):
        return 0
    if stem.startswith("smoking_"):
        return 1
    raise ValueError(f"cannot tell the label of {name}")


@dataclass
class Counts:
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0

    @property
    def pos(self) -> int:
        return self.tp + self.fn

    @property
    def neg(self) -> int:
        return self.fp + self.tn

    @property
    def recall(self) -> float:
        return self.tp / self.pos if self.pos else 0.0


def counts(fired: list[bool], labels: list[int]) -> Counts:
    c = Counts()
    for f, y in zip(fired, labels):
        if y == 1:
            c.tp += f
            c.fn += not f
        else:
            c.fp += f
            c.tn += not f
    return c


def verified(det_fired: list[bool], cls_prob: list[float], threshold: float) -> list[bool]:
    return [bool(d and p >= threshold) for d, p in zip(det_fired, cls_prob)]


def is_better(base: Counts, new: Counts, max_recall_drop: float) -> bool:
    return new.fp < base.fp and new.recall >= base.recall - max_recall_drop - 1e-9


@dataclass
class Selection:
    threshold: float | None
    baseline: Counts
    chosen: Counts | None
    table: list[tuple[float, Counts]]

    @property
    def adopted(self) -> bool:
        return self.chosen is not None and self.chosen.fp < self.baseline.fp


def select_threshold(det_fired: list[bool], cls_prob: list[float], labels: list[int],
                     max_recall_drop: float = 0.05) -> Selection:
    base = counts(det_fired, labels)
    table = [(t, counts(verified(det_fired, cls_prob, t), labels)) for t in CANDIDATES]
    ok = [(t, c) for t, c in table if c.recall >= base.recall - max_recall_drop - 1e-9]
    if not ok:
        return Selection(None, base, None, table)
    t, c = min(ok, key=lambda tc: (tc[1].fp, tc[0]))
    return Selection(t, base, c, table)


def dhash(image: Any) -> int:
    """64-bit difference hash: survives resizing, brightness and recompression."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicates(reference: dict[str, int], others: dict[str, int],
                    max_bits: int = 6) -> list[tuple[str, str, int]]:
    """(other, closest reference, differing bits) for every near-duplicate pair."""
    out = []
    for name, h in sorted(others.items()):
        best = min(((bin(h ^ r).count("1"), rname) for rname, r in reference.items()), default=None)
        if best is not None and best[0] <= max_bits:
            out.append((name, best[1], best[0]))
    return out


def save_threshold(path: Path, threshold: float, meta: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"threshold": threshold, "split": "Validation", **meta},
                                     indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_threshold(path: Path) -> float:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("split") != "Validation":
        raise ValueError("the threshold must come from the Validation split")
    return float(data["threshold"])


# --------------------------------------------------------------------------
# Model runs
# --------------------------------------------------------------------------


def split_images(dataset: Path, split: str) -> list[Path]:
    folder = Path(dataset).expanduser() / split
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def run_detector(images: list[Path]) -> tuple[list[bool], str]:
    """Fires exactly like run_core.py: the smoking module and its policy from config/core.yaml."""
    import cv2

    from src.cctv_core.runner import build_modules
    from src.cctv_core.schemas import Frame
    from src.utils.config import load_config

    (module,) = build_modules(load_config(resolve_path("config/core.yaml")), only={"smoking"})
    module.detector.load()
    fired = []
    for i, path in enumerate(images):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        out = module.detector.process(Frame("mendeley", i, 0.0, img, w, h))
        fired.append(any(p.evaluate(out) for p in module.policies))
    info = module.detector.model_info
    return fired, f"{info.weights} sha256 {(info.sha256 or 'n/a')[:16]}"


def run_classifier(images: list[Path], weights: str) -> tuple[list[float], str]:
    import cv2
    from ultralytics import YOLO

    from src.detection.yolo_detector import select_device
    from src.detectors.yolo_label import sha256_of

    path = resolve_path(weights)
    model = YOLO(str(path))
    names = {int(k): v for k, v in model.names.items()}
    if "smoking" not in names.values():
        raise ValueError(f"{weights} has no 'smoking' class: {names}")
    idx = next(k for k, v in names.items() if v == "smoking")
    device = select_device("auto")
    probs = []
    for p in images:
        res = model.predict(cv2.imread(str(p)), imgsz=224, device=device, verbose=False)[0]
        probs.append(float(res.probs.data[idx]))
    return probs, f"{weights} sha256 {sha256_of(path)[:16]}"


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def _row(name: str, c: Counts) -> str:
    return f"| {name} | {c.tp}/{c.pos} ({c.recall:.0%}) | {c.fp}/{c.neg} ({c.fp / c.neg if c.neg else 0:.0%}) |"


def cmd_dupes(args) -> int:
    import cv2

    def hashes(split):
        return {p.name: dhash(cv2.imread(str(p))) for p in split_images(args.dataset, split)}

    train = hashes("Training")
    lines = ["# Mendeley near-duplicate check", "",
             f"- date: {datetime.now():%Y-%m-%d %H:%M}",
             f"- dHash, at most {args.max_bits} of 64 bits different = near-duplicate", ""]
    for split in ("Validation", "Testing"):
        dupes = near_duplicates(train, hashes(split), args.max_bits)
        lines.append(f"## {split}: {len(dupes)} images look like a Training image")
        lines += [f"- {o} ~ {t} ({b} bits)" for o, t, b in dupes[:50]]
        if len(dupes) > 50:
            lines.append(f"- ... {len(dupes) - 50} more")
        lines.append("")
    out = resolve_path(OUT_DIR) / "smoking_cls_dupes.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4] + [line for line in lines if line.startswith("## ")]))
    print(f"wrote {out}")
    return 0


def cmd_select(args) -> int:
    images = split_images(args.dataset, "Validation")
    labels = [label_from_name(p.name) for p in images]
    det, det_info = run_detector(images)
    prob, cls_info = run_classifier(images, args.classifier)
    sel = select_threshold(det, prob, labels, args.max_recall_drop)

    lines = ["# Smoking verifier: threshold selection on Mendeley Validation", "",
             f"- date: {datetime.now():%Y-%m-%d %H:%M}", f"- detector: {det_info}",
             f"- classifier: {cls_info}", f"- images: {len(images)} ({sum(labels)} smoking)",
             f"- rule: fewest false alarms with recall at most {args.max_recall_drop:.0%} below the detector alone",
             "", "| threshold | detected (TP/positives) | false alarms (FP/negatives) |", "|---|---|---|",
             _row("detector alone", sel.baseline)]
    lines += [_row(f"+ verifier >= {t:.2f}", c) for t, c in sel.table]
    if sel.adopted:
        lines += ["", f"**Chosen threshold: {sel.threshold:.2f}** (then measure on Testing with `report`)"]
        save_threshold(resolve_path(THRESHOLD_FILE), sel.threshold,
                       {"classifier": cls_info, "detector": det_info,
                        "max_recall_drop": args.max_recall_drop, "date": f"{datetime.now():%Y-%m-%d %H:%M}"})
    else:
        lines += ["", "**No threshold reduces false alarms within the recall budget: verifier NOT adopted.**"]
    out = resolve_path(OUT_DIR) / "smoking_cls_validation.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out}")
    return 0 if sel.adopted else 1


def cmd_report(args) -> int:
    threshold = load_threshold(resolve_path(THRESHOLD_FILE))
    images = split_images(args.dataset, "Testing")
    labels = [label_from_name(p.name) for p in images]
    det, det_info = run_detector(images)
    prob, cls_info = run_classifier(images, args.classifier)
    saved = json.loads(resolve_path(THRESHOLD_FILE).read_text(encoding="utf-8"))
    if saved.get("classifier") != cls_info:
        print(f"[REPORT] threshold was chosen for {saved.get('classifier')}, not {cls_info} - run select again")
        return 1
    base = counts(det, labels)
    new = counts(verified(det, prob, threshold), labels)
    better = is_better(base, new, args.max_recall_drop)
    lines = ["# Smoking verifier: before / after on Mendeley Testing", "",
             f"- date: {datetime.now():%Y-%m-%d %H:%M}", f"- detector: {det_info}",
             f"- classifier: {cls_info}", f"- threshold: {threshold:.2f} (chosen on Validation)",
             f"- images: {len(images)} ({sum(labels)} smoking); Testing was not used for training or selection",
             "", "| | detected (TP/positives) | false alarms (FP/negatives) |", "|---|---|---|",
             _row("before: detector alone", base), _row("after: detector + verifier", new), "",
             f"Decision: **{'ADOPT' if better else 'DO NOT ADOPT'}** (needs fewer false alarms and recall "
             f"at most {args.max_recall_drop:.0%} lower)",
             "", "Mendeley is one source of whole-person photos; QA clips are reported separately "
             "(tools/eval_clips.py) and decide whether this works on the camera."]
    out = resolve_path(OUT_DIR) / "smoking_cls_mendeley_test.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoking verifier on the Mendeley dataset")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dupes", "select", "report"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, type=Path,
                       help="folder with Training/Validation/Testing")
        if name == "dupes":
            p.add_argument("--max-bits", type=int, default=6)
        else:
            p.add_argument("--classifier", required=True)
            p.add_argument("--max-recall-drop", type=float, default=0.05)
    args = ap.parse_args()
    return {"dupes": cmd_dupes, "select": cmd_select, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
