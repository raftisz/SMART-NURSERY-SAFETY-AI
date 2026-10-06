"""Fine-tune the hazard_object model (COCO yolo11 -> + web knife data).

Starts from the pretrained COCO weights and keeps all 80 classes (see
tools/build_hazard_dataset.py for why). Defaults are chosen for a FINE-TUNE,
not training from scratch:
  - lr0 0.002 with SGD (5x below the Ultralytics default): the COCO features
    are already good; a large step erases them (catastrophic forgetting).
  - freeze 10 = the backbone stays as trained on COCO; only neck + head adapt.
    Halves the backward pass on CPU and limits forgetting. Use --freeze 0 on a
    GPU with the full dataset.
  - mosaic on, closed for the last 20% of epochs, fliplr, HSV jitter: standard.
  - scale 0.9: objects far from the camera are small; the stock 0.5 rarely
    shows the model a knife below ~1/4 of its photo size.
  - seed 0, deterministic, so a re-run gives the same weights.

Writes models/hazard/<name>.pt + <name>.train_info.json; refuses to overwrite.

Usage
    python tools/train_hazard_model.py --data <hazard_v1>/data.yaml --epochs 12 --name hazard_v1
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--base", default="yolo11n.pt")
    ap.add_argument("--name", default="hazard_v1")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--freeze", type=int, default=10)
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--scale", type=float, default=0.9,
                    help="random zoom range +-scale. 0.9 (vs Ultralytics 0.5) shrinks objects down to 10%% "
                         "of their size, so the model sees many SMALL knives/scissors (far from the camera)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--project", type=Path, default=Path("runs/hazard"))
    ap.add_argument("--fraction", type=float, default=1.0, help="use part of the train split (smoke test)")
    args = ap.parse_args()

    out_dir = ROOT / "models" / "hazard"
    out_pt = out_dir / f"{args.name}.pt"
    if out_pt.exists():
        print(f"{out_pt} exists - pick another --name (models are never overwritten)")
        return 1

    import torch
    import ultralytics
    from ultralytics import YOLO

    train_args = dict(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, seed=0, deterministic=True,
        optimizer="SGD", lr0=args.lr0, lrf=0.1, momentum=0.937, weight_decay=5e-4,
        warmup_epochs=1.0, cos_lr=True, freeze=args.freeze,
        mosaic=1.0, close_mosaic=max(1, round(args.epochs * 0.2)), mixup=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, scale=args.scale, translate=0.1,
        patience=max(5, args.epochs // 3), fraction=args.fraction,
        project=str(args.project.resolve()), name=args.name, exist_ok=True, plots=True, verbose=True,
    )
    model = YOLO(args.base)
    model.train(**train_args)

    run = args.project.resolve() / args.name
    best = run / "weights" / "best.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out_pt)
    digest = hashlib.sha256(out_pt.read_bytes()).hexdigest()
    info = {
        "file": out_pt.name, "sha256": digest,
        "date": datetime.datetime.now().isoformat(timespec="minutes"),
        "base": args.base, "classes": "COCO-80 (unchanged order)",
        "data": str(args.data), "train_args": {k: v for k, v in train_args.items() if k != "project"},
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "device": args.device,
    }
    stats = args.data.parent / "stats.json"
    if stats.exists():
        info["dataset_stats"] = json.loads(stats.read_text())
    (out_dir / f"{args.name}.train_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
