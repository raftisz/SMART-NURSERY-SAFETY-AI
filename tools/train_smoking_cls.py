"""Train the smoking verifier (yolo11n-cls) on this Mac, the same way as the Colab notebook.

Uses ONLY the Mendeley `Training` folder (716 images). It is split per class
85/15 with seed 0 (exactly as notebooks/smoking_cls_colab.ipynb) so Ultralytics
can pick the best epoch; Validation and Testing are never read here. Those are
used afterwards by tools/eval_smoking_cls.py (select on Validation, report on
Testing).

Writes a NEW file models/smoking_cls/smoking_cls_v1_mac.pt plus train_info.json
next to it; it refuses to overwrite and never writes into models/smoking/.

Usage (from the repo root):
    python tools/train_smoking_cls.py --dataset ~/Downloads/"Smoker Detection Dataset/Smoker Detection" \\
        --work /tmp/smoking_cls_work
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import resolve_path  # noqa: E402

SEED = 0
VAL_FRACTION = 0.15
CLASSES = ("smoking", "notsmoking")  # same order as the notebook (it affects the shuffle)
TRAIN_ARGS = {"epochs": 30, "imgsz": 224, "batch": 32, "patience": 10}
BASE_WEIGHTS = "yolo11n-cls.pt"
DEFAULT_OUT = "models/smoking_cls/smoking_cls_v1_mac.pt"
PROTECTED = "models/smoking"


def label_of(name: str) -> str:
    n = name.lower()
    if n.startswith("notsmoking_"):
        return "notsmoking"
    if n.startswith("smoking_"):
        return "smoking"
    raise ValueError(f"cannot tell the class of {name}")


def training_files(dataset_root: Path) -> dict[str, list[Path]]:
    """Images of <dataset_root>/Training only, by class."""
    folder = Path(dataset_root).expanduser() / "Training"
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found")
    by_class: dict[str, list[Path]] = {c: [] for c in CLASSES}
    for p in sorted(folder.glob("*.jpg")):
        by_class[label_of(p.name)].append(p)
    return by_class


def split_training(by_class: dict[str, list[Path]]) -> dict[str, list[Path]]:
    """{'train/smoking': [...], 'val/smoking': [...], ...} - the notebook's split, seed 0."""
    rng = random.Random(SEED)
    out: dict[str, list[Path]] = {}
    for cls in CLASSES:
        files = list(by_class[cls])
        rng.shuffle(files)
        n_val = round(len(files) * VAL_FRACTION)
        out[f"val/{cls}"] = files[:n_val]
        out[f"train/{cls}"] = files[n_val:]
    return out


def check_output(path: Path) -> Path:
    path = Path(path)
    resolved = resolve_path(path).resolve() if not path.is_absolute() else path.resolve()
    if resolve_path(PROTECTED).resolve() in resolved.parents:
        raise ValueError(f"never write into {PROTECTED}/ (the original smoking model)")
    if resolved.exists():
        raise FileExistsError(f"{resolved} exists - choose a new name, weights are never overwritten")
    return resolved


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_info(weights: Path, counts: dict[str, int], args: dict[str, Any], device: str) -> dict[str, Any]:
    import platform

    info: dict[str, Any] = {
        "file": Path(weights).name, "sha256": sha256_file(weights),
        "date": datetime.datetime.now().isoformat(timespec="minutes"),
        "base": BASE_WEIGHTS,
        "data": f"Mendeley Smoker Detection - Training only, split {1 - VAL_FRACTION:.0%}/{VAL_FRACTION:.0%} "
                f"per class with seed {SEED} (same as notebooks/smoking_cls_colab.ipynb)",
        "split_counts": counts, "train_args": args, "device": device,
        "machine": f"{platform.machine()} {platform.platform()}",
    }
    try:
        import torch
        import ultralytics

        info.update(ultralytics=ultralytics.__version__, torch=torch.__version__)
    except ImportError:
        pass
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the smoking verifier on the Mac")
    ap.add_argument("--dataset", required=True, type=Path, help="folder holding Training/")
    ap.add_argument("--work", required=True, type=Path, help="scratch folder for the split and the run")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    out = check_output(Path(args.out))
    split = split_training(training_files(args.dataset))
    counts = {k: len(v) for k, v in split.items()}
    print("split:", counts, "total", sum(counts.values()))

    ds = Path(args.work) / "ds"
    if ds.exists():
        shutil.rmtree(ds)
    for key, files in split.items():
        dest = ds / key
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / f.name)

    from ultralytics import YOLO

    model = YOLO(str(resolve_path(BASE_WEIGHTS)))
    run_args = dict(TRAIN_ARGS, seed=SEED, deterministic=True)
    model.train(data=str(ds), device=args.device, plots=False, project=str(Path(args.work) / "runs"),
                name="smoking_cls_v1_mac", exist_ok=True, **run_args)

    best = Path(args.work) / "runs" / "smoking_cls_v1_mac" / "weights" / "best.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    info = train_info(out, counts, run_args, args.device)
    results_csv = best.parent.parent / "results.csv"
    if results_csv.is_file():
        info["results_csv_last_line"] = results_csv.read_text().strip().splitlines()[-1]
    out.with_name("train_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
