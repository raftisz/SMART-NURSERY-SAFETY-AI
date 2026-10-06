"""Train the custom smoke detector (yolo11n) on this Mac, outside the main pipeline.

Dataset: ~/smoke_data/dataset built by tools/build_smoke_dataset.py
         (ScPuteri "Cigarette Vape Smoke" v11, CC BY 4.0, smoke boxes only, plus
          ITLP Campus Indoor CC BY 4.0 corridor photos as no-smoke images)
Fixed settings: yolo11n.pt, imgsz 640, seed 0, device mps (no CPU fallback), plots off.
Runs go to ~/smoke_data/runs/<name>; an existing run is never overwritten.
YOLO_AUTOINSTALL is forced off so nothing is installed while training.

Usage (from the repo root):
    python tools/train_smoke_custom.py --epochs 1  --name timing_1epoch      # time one epoch
    python tools/train_smoke_custom.py --epochs 50 --name smoke_custom_yolo11n_v1
    python tools/train_smoke_custom.py --finalize ~/smoke_data/runs/smoke_custom_yolo11n_v1
"""

from __future__ import annotations

import os

os.environ["YOLO_AUTOINSTALL"] = "False"  # read by ultralytics at import: never auto-install

import argparse
import datetime
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "yolo11n.pt"
IMGSZ, SEED, DEVICE, BATCH = 640, 0, "mps", 16
DATA = Path("~/smoke_data/dataset/data.yaml").expanduser()
PROJECT = Path("~/smoke_data/runs").expanduser()
OUT_MODEL = ROOT / "models/smoke/smoke_custom_yolo11n_v1.pt"
FORBIDDEN = "smoke_eval_clips"
SOURCES = [
    "Roboflow 'Cigarette Vape Smoke' v11 by ScPuteri (universe.roboflow.com/scputeri/cigarette-vape-smoke-kcg4j), "
    "CC BY 4.0 - smoke boxes (class 2) only",
    "ITLP Campus Indoor, OPR-Project (huggingface.co/datasets/OPR-Project/ITLP-Campus-Indoor, commit f0908283), "
    "CC BY 4.0 - 400 corridor photos as no-smoke images",
]


def train_args(data: Path, epochs: int, name: str, project: Path, workers: int = 8,
               batch: int = BATCH, imgsz: int = IMGSZ, cache: bool = False) -> dict:
    return dict(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, seed=SEED, deterministic=True,
                device=DEVICE, project=str(project), name=name, exist_ok=False, plots=False,
                patience=max(epochs, 1), workers=workers, cache=cache)


def check_paths(data: Path, project: Path) -> None:
    for p in (data, project):
        if FORBIDDEN in str(p):
            raise ValueError(f"{p}: test clips must never be used for training")


def check_output(path: Path) -> Path:
    if Path(path).exists():
        raise FileExistsError(f"{path} exists - weights are never overwritten")
    return Path(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_info(weights: Path, build_info: dict, args: dict, val_metrics: dict, seconds: float) -> dict:
    info = {
        "file": Path(weights).name, "sha256": sha256_file(weights),
        "date": datetime.datetime.now().isoformat(timespec="minutes"), "base": BASE,
        "classes": {0: "smoke"}, "data_sources": SOURCES, "license_note": "data CC BY 4.0; weights AGPL-3.0 (Ultralytics)",
        "dataset": build_info, "train_args": args, "val_split_metrics": val_metrics,
        "train_seconds": round(seconds), "device": DEVICE,
        "note": "val metrics are on the dataset split (close-up web photos + corridor photos), not on our camera; "
                "measure with tools/smoke_live_test.py before any use",
    }
    try:
        import torch
        import ultralytics

        info.update(ultralytics=ultralytics.__version__, torch=torch.__version__)
    except ImportError:
        pass
    return info


def run_training(epochs: int, name: str, workers: int = 8, reason: str = "",
                 batch: int = BATCH, imgsz: int = IMGSZ, cache: bool = False) -> int:
    import torch
    from ultralytics import YOLO

    check_paths(DATA, PROJECT)
    if not torch.backends.mps.is_available():
        print("mps is not available - stop (no CPU fallback)")
        return 2
    if (PROJECT / name).exists():
        print(f"{PROJECT / name} exists - choose a new --name")
        return 2
    args = train_args(DATA, epochs, name, PROJECT, workers, batch, imgsz, cache)
    print("train args:", args)
    started = time.perf_counter()
    try:
        YOLO(str(ROOT / BASE)).train(**args)
    except Exception as exc:  # report and stop; never retry on another device
        print(f"TRAINING STOPPED: {type(exc).__name__}: {exc}")
        return 3
    seconds = time.perf_counter() - started
    (PROJECT / name / "wall_time.json").write_text(json.dumps(
        {"epochs": epochs, "workers": workers, "batch": batch, "imgsz": imgsz, "cache": cache,
         "seconds": round(seconds, 1), "reason": reason}))
    print(f"wall time for {epochs} epoch(s): {seconds:.0f} s")
    return 0


def finalize(run_dir: Path) -> int:
    from ultralytics import YOLO

    run_dir = Path(run_dir).expanduser()
    best = run_dir / "weights" / "best.pt"
    out = check_output(OUT_MODEL)
    wall = json.loads((run_dir / "wall_time.json").read_text()) if (run_dir / "wall_time.json").is_file() else {}
    imgsz = wall.get("imgsz", IMGSZ)
    val = YOLO(str(best)).val(data=str(DATA), split="val", imgsz=imgsz, device=DEVICE, plots=False)
    metrics = {"mAP50": float(val.box.map50), "mAP50-95": float(val.box.map),
               "precision": float(val.box.mp), "recall": float(val.box.mr)}
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    build_info = json.loads((DATA.parent / "build_info.json").read_text())
    args = json.loads(json.dumps(train_args(DATA, wall.get("epochs", 0), run_dir.name, run_dir.parent,
                                            wall.get("workers", 8), wall.get("batch", BATCH), imgsz,
                                            wall.get("cache", False))))
    info = train_info(out, build_info, args, metrics, wall.get("seconds", 0.0))
    info["epochs_decision"] = wall.get("reason", "")
    out.with_name("smoke_custom_yolo11n_v1.train_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"sha256": info["sha256"], "val": metrics}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the custom smoke detector on the Mac (mps)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--name", default="smoke_custom_yolo11n_v1")
    ap.add_argument("--workers", type=int, default=8, help="dataloader workers (0 = load in the main process)")
    ap.add_argument("--reason", default="", help="why this number of epochs (stored in train_info.json)")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--cache", action="store_true", help="cache images in RAM (default off)")
    ap.add_argument("--finalize", default=None, help="run folder: copy best.pt to models/smoke and write train_info")
    args = ap.parse_args()
    if args.finalize:
        return finalize(Path(args.finalize))
    return run_training(args.epochs, args.name, args.workers, args.reason, args.batch, args.imgsz, args.cache)


if __name__ == "__main__":
    sys.exit(main())
