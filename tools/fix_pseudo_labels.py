"""Re-apply the GT-conflict rule to a dataset built before the rule existed.

tools/build_hazard_dataset.py now drops teacher boxes that sit on a ground-truth
box of another class (e.g. teacher 'scissors' on a GT knife). Datasets built
earlier (the first CPU build, the first Colab run) still have them. This
removes them in place without re-running the teacher: every non-GT box in a
label file came from the teacher, and the GT can be re-read from the source.

Usage
    python tools/fix_pseudo_labels.py --data <hazard dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_hazard_dataset import (HOD_KNIFE, KNIFE, SOHAS_TO_COCO,  # noqa: E402
                                        conflicts_with_gt, format_yolo, iou_xywh, parse_yolo)


def source_gt(src: Path, source: str) -> list:
    if source == "hod":
        lab = src.parent.parent / "txt" / (src.stem + ".txt")
        return [(KNIFE, *b[1:]) for b in parse_yolo(lab.read_text()) if b[0] == HOD_KNIFE]
    lab = Path(str(src).replace("/images/", "/labels/")).with_suffix(".txt")
    raw = parse_yolo(lab.read_text()) if lab.exists() else []
    return [(SOHAS_TO_COCO[b[0]], *b[1:]) for b in raw if b[0] in SOHAS_TO_COCO]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    removed = files = 0
    for m in json.loads((args.data / "manifest.json").read_text()):
        if m["source"] == "coco" or m["split"] == "test":
            continue
        path = args.data / "labels" / m["split"] / (m["file"] + ".txt")
        gt = source_gt(Path(m["src"]), m["source"])
        labels = parse_yolo(path.read_text())
        # label files are rounded to 6 decimals, so a GT box is recognised by IoU, not equality
        keep = [b for b in labels
                if any(g[0] == b[0] and iou_xywh(g, b) > 0.99 for g in gt) or not conflicts_with_gt(b, gt)]
        if len(keep) != len(labels):
            removed += len(labels) - len(keep)
            files += 1
            path.write_text(format_yolo(keep))
    # Ultralytics caches labels; a stale cache would hide the fix
    for cache in (args.data / "labels").glob("*.cache"):
        cache.unlink()
    print(f"removed {removed} conflicting teacher boxes from {files} label files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
