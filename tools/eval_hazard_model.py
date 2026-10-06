"""Compare hazard_object weights on held-out images, with confidence intervals.

Two kinds of numbers, because they answer different questions:

* box level  - AP50 per class (COCO 101-point, IoU 0.5) and precision/recall at
               a threshold. "Does the model find the object and put the box in
               the right place?"
* image level - what the alert actually depends on. An image "has a hazard"
               when it has a knife or scissors box in the ground truth.
                 hazard recall      = hazard images with >= 1 knife/scissors
                                      detection at >= threshold
                 false-alarm rate   = hazard-FREE images with >= 1 knife/scissors
                                      detection at >= threshold
               (the label does not have to match: a knife reported as scissors
               still raises the same hazard_object alert)

Every number has a 95% bootstrap interval (resampling images, 1000 reps, seed 0).
Differences between two models use a PAIRED bootstrap (same resampled images
for both), so "B is better than A" is only claimed when the interval of the
difference excludes 0.

Thresholds are picked on the VALIDATION split (largest F2 = recall counts
twice as much as precision, because a missed knife is worse than a false
alarm) and then applied unchanged to the TEST split. Never the other way round.

Usage
    python tools/eval_hazard_model.py --data <hazard_v1 dir> \
        --model base=yolo11n.pt@640 --model base960=yolo11n.pt@960 \
        --model v1=models/hazard/hazard_v1.pt@640 --out docs/eval/hazard_v1_eval.md
Predictions are cached next to the dataset (preds_<name>_<split>.json).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

KNIFE, SCISSORS, PERSON = 43, 76, 0
HAZARD = (KNIFE, SCISSORS)
CLASSES = {PERSON: "person", KNIFE: "knife", SCISSORS: "scissors"}
IOU = 0.5
REPS = 1000
SEED = 0


# --------------------------------------------------------------------------
# metrics (pure, unit-tested)
# --------------------------------------------------------------------------

def iou_xyxy(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_image(gts: list, preds: list, iou_thr: float = IOU) -> list[tuple[float, bool]]:
    """Greedy matching for ONE class in ONE image. preds: [(conf, box)]. -> [(conf, is_tp)]."""
    used = [False] * len(gts)
    out = []
    for conf, box in sorted(preds, key=lambda p: -p[0]):
        best, best_j = iou_thr, -1
        for j, g in enumerate(gts):
            if used[j]:
                continue
            v = iou_xyxy(box, g)
            if v >= best:
                best, best_j = v, j
        if best_j >= 0:
            used[best_j] = True
        out.append((conf, best_j >= 0))
    return out


def average_precision(scored: list[tuple[float, bool]], n_gt: int) -> float:
    """COCO-style 101-point interpolated AP from (conf, is_tp) over all images."""
    if n_gt == 0:
        return float("nan")
    if not scored:
        return 0.0
    scored = sorted(scored, key=lambda s: -s[0])
    tp = np.cumsum([s[1] for s in scored])
    fp = np.cumsum([not s[1] for s in scored])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    # precision envelope
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(recall, r, side="left")
        ap += precision[idx] if idx < len(precision) else 0.0
    return ap / 101


#: relative object size = sqrt(box area / image area). In a 1280x720 frame,
#: 0.03 ~ a 29 px object (scissors held 3-4 m away), 0.08 ~ 77 px.
SIZE_BUCKETS = {"small (<3%)": (0.0, 0.03), "medium (3-8%)": (0.03, 0.08), "large (>8%)": (0.08, 9.0)}


class PerImage:
    """Pre-matched results of one model on one image, so bootstrap reps are cheap."""

    def __init__(self, gt: dict[int, list], preds: dict[int, list], wh: tuple[int, int] = (1, 1)):
        self.n_gt = {c: len(gt.get(c, [])) for c in CLASSES}
        area = max(1.0, float(wh[0] * wh[1]))
        # label-agnostic hazard boxes, for recall by object size
        self.hazard_gt = [(b, (((b[2] - b[0]) * (b[3] - b[1])) / area) ** 0.5) for c in HAZARD for b in gt.get(c, [])]
        self.hazard_pred = sorted([p for c in HAZARD for p in preds.get(c, [])], key=lambda p: -p[0])
        self.scored = {c: match_image(gt.get(c, []), preds.get(c, [])) for c in CLASSES}
        self.has_hazard = any(gt.get(c) for c in HAZARD)
        # best hazard confidence in the image (any hazard label)
        self.hazard_conf = max([p[0] for c in HAZARD for p in preds.get(c, [])], default=0.0)


def class_ap(images: list[PerImage], c: int) -> float:
    scored = [s for im in images for s in im.scored[c]]
    return average_precision(scored, sum(im.n_gt[c] for im in images))


def pr_at(images: list[PerImage], c: int, thr: float) -> tuple[float, float]:
    tp = sum(s[1] for im in images for s in im.scored[c] if s[0] >= thr)
    npred = sum(1 for im in images for s in im.scored[c] if s[0] >= thr)
    ngt = sum(im.n_gt[c] for im in images)
    return (tp / npred if npred else float("nan")), (tp / ngt if ngt else float("nan"))


def recall_by_size(images: list[PerImage], thr: float, lo: float, hi: float) -> float:
    """Box recall of knife+scissors (either label) whose relative size is in [lo, hi)."""
    hit = total = 0
    for im in images:
        want = [k for k, (_, size) in enumerate(im.hazard_gt) if lo <= size < hi]
        if not want:
            continue
        total += len(want)
        used = set()
        for conf, box in im.hazard_pred:
            if conf < thr:
                break
            best, best_k = IOU, -1
            for k, (g, _) in enumerate(im.hazard_gt):
                if k in used:
                    continue
                v = iou_xyxy(box, g)
                if v >= best:
                    best, best_k = v, k
            if best_k >= 0:
                used.add(best_k)
        hit += len(used & set(want))
    return hit / total if total else float("nan")


def image_rates(images: list[PerImage], thr: float) -> tuple[float, float]:
    """(hazard recall, false-alarm rate) at image level."""
    pos = [im for im in images if im.has_hazard]
    neg = [im for im in images if not im.has_hazard]
    rec = sum(im.hazard_conf >= thr for im in pos) / len(pos) if pos else float("nan")
    far = sum(im.hazard_conf >= thr for im in neg) / len(neg) if neg else float("nan")
    return rec, far


def best_f2_threshold(images: list[PerImage], c: int, grid=None) -> float:
    grid = grid if grid is not None else [round(x, 2) for x in np.arange(0.05, 0.91, 0.05)]
    best, best_t = -1.0, grid[0]
    for t in grid:
        p, r = pr_at(images, c, t)
        if p != p or r != r or p + r == 0:
            continue
        f2 = 5 * p * r / (4 * p + r)
        if f2 > best:
            best, best_t = f2, t
    return best_t


def bootstrap(fn, images: list[PerImage], reps: int = REPS, seed: int = SEED):
    """(point, lo, hi) of fn(images) with a 95% percentile interval."""
    point = fn(images)
    rng = random.Random(seed)
    n = len(images)
    vals = []
    for _ in range(reps):
        sample = [images[rng.randrange(n)] for _ in range(n)]
        v = fn(sample)
        if v == v:
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    vals.sort()
    return point, vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def paired_bootstrap(fn, a: list[PerImage], b: list[PerImage], reps: int = REPS, seed: int = SEED):
    """(point, lo, hi) of fn(b) - fn(a) on the same resampled images."""
    assert len(a) == len(b)
    point = fn(b) - fn(a)
    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        v = fn([b[i] for i in idx]) - fn([a[i] for i in idx])
        if v == v:
            vals.append(v)
    vals.sort()
    if not vals:
        return point, float("nan"), float("nan")
    return point, vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def load_gt(label_path: Path, w: int, h: int) -> dict[int, list]:
    gt: dict[int, list] = defaultdict(list)
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(p[0])
            if c not in CLASSES:
                continue
            x, y, bw, bh = map(float, p[1:5])
            gt[c].append(((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
    return gt


def predict_cached(weights: str, imgsz: int, images: list[Path], cache: Path, device: str,
                   zoom: bool = False) -> dict:
    """conf 0.01 so every threshold can be evaluated afterwards. zoom=True adds the
    person-guided zoom pass of src/detectors/hazard_zoom.py (default ZoomConfig)."""
    if cache.exists():
        return json.loads(cache.read_text())
    from ultralytics import YOLO
    model = YOLO(weights)
    names = {v: k for k, v in model.names.items()}
    want = {names[n]: c for c, n in CLASSES.items() if n in names}
    hazard_ids = sorted(i for i, c in want.items() if c in HAZARD)
    if zoom:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.detectors.hazard_zoom import ZoomConfig, zoom_detect
        zcfg = ZoomConfig(enabled=True)

        def predict_crops(crops):
            rs = model.predict(crops, imgsz=zcfg.crop_imgsz, conf=0.01, classes=hazard_ids,
                               device=device, verbose=False)
            return [[(str(want[int(c)]), float(p), tuple(b))
                     for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
                    for r in rs]
    out = {}
    for k in range(0, len(images), 16):
        batch = images[k:k + 16]
        for path, r in zip(batch, model.predict([str(p) for p in batch], imgsz=imgsz, conf=0.01,
                                                classes=sorted(want), device=device, verbose=False)):
            h, w = r.orig_shape
            boxes = [[want[int(c)], float(p), *map(float, b)]
                     for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            if zoom:
                full = [(str(c), p, tuple(b)) for c, p, *b in boxes]
                full_for_zoom = [("person" if c == str(PERSON) else c, p, b) for c, p, b in full]
                merged, _ = zoom_detect(r.orig_img, full_for_zoom, "person", zcfg, predict_crops)
                boxes = [[PERSON if c == "person" else int(c), p, *b] for c, p, b in merged]
            out[path.name] = {"wh": [w, h], "boxes": boxes}
        if (k // 16) % 25 == 0:
            print(f"  [{cache.stem}] {k + len(batch)}/{len(images)}", flush=True)
    cache.write_text(json.dumps(out))
    return out


def per_image(data: Path, split: str, preds: dict, files: list[Path]) -> list[PerImage]:
    res = []
    for f in files:
        rec = preds[f.name]
        w, h = rec["wh"]
        gt = load_gt(data / "labels" / split / (f.stem + ".txt"), w, h)
        pr: dict[int, list] = defaultdict(list)
        for c, conf, *box in rec["boxes"]:
            pr[int(c)].append((conf, box))
        res.append(PerImage(gt, pr, (w, h)))
    return res


def sha(path: str) -> str:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else "(stock download)"


def fmt(v, pct=True) -> str:
    point, lo, hi = v
    if point != point:
        return "n/a"
    if pct:
        return f"{100 * point:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, help="name=weights@imgsz")
    ap.add_argument("--baseline", help="name of the model differences are measured against (default: first)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fixed-threshold", type=float, default=0.25,
                    help="the threshold config/core.yaml uses today; reported next to the tuned one")
    ap.add_argument("--max-coco-negatives", type=int, default=1000,
                    help="cap on hazard-free COCO test images (CPU time)")
    args = ap.parse_args()

    models = []
    for spec in args.model:
        name, rest = spec.split("=", 1)
        weights, size = rest.rsplit("@", 1)
        zoom = size.endswith("+zoom")
        models.append((name, weights, int(size.replace("+zoom", "")), zoom))
    base_name = args.baseline or models[0][0]

    manifest = {m["file"]: m for m in json.loads((args.data / "manifest.json").read_text())}
    files: dict[str, list[Path]] = {}
    for split in ("val", "test"):
        fs = sorted((args.data / "images" / split).glob("*.jpg"))
        if split == "test":
            rng = random.Random(SEED)
            coco_neg = [f for f in fs if manifest[f.stem]["source"] == "coco"
                        and not any(c in HAZARD for c in load_gt(args.data / "labels" / split / (f.stem + ".txt"), 1, 1))]
            rng.shuffle(coco_neg)
            drop = set(coco_neg[args.max_coco_negatives:])
            fs = [f for f in fs if f not in drop]
        files[split] = fs

    results: dict[str, dict[str, list[PerImage]]] = {}
    for name, weights, imgsz, zoom in models:
        results[name] = {}
        for split in ("val", "test"):
            cache = args.data / f"preds_{name}_{split}.json"
            preds = predict_cached(weights, imgsz, files[split], cache, args.device, zoom)
            results[name][split] = per_image(args.data, split, preds, files[split])

    def subset(name: str, split: str, source: str | None) -> list[PerImage]:
        ims = results[name][split]
        if source is None:
            return ims
        return [im for im, f in zip(ims, files[split]) if manifest[f.stem]["source"] == source]

    lines = [f"# hazard_object: model comparison on held-out images",
             "", f"- date: {datetime.datetime.now():%Y-%m-%d %H:%M}",
             f"- dataset: `{args.data.name}` (built by tools/build_hazard_dataset.py; split by source, "
             "near-duplicates grouped)",
             f"- test images: {len(files['test'])} (COCO hazard-free images capped at {args.max_coco_negatives})",
             "- 95% bootstrap intervals in brackets (images resampled, 1000 reps)",
             "- **These are web/COCO photos, not the nursery camera.** They rank models; they do not "
             "replace the QA clips (tools/eval_clips.py) or the Phase 2 Decision Gate.", "",
             "| model | weights | sha256 | imgsz | person-guided zoom |", "|---|---|---|---|---|"]
    for name, weights, imgsz, zoom in models:
        lines.append(f"| {name} | `{weights}` | `{sha(weights)}` | {imgsz} | {'yes' if zoom else 'no'} |")

    thresholds: dict[str, dict[int, float]] = {}
    for name, *_ in models:
        val = results[name]["val"]
        thresholds[name] = {c: best_f2_threshold(val, c) for c in HAZARD}
        # image-level threshold: the lower of the two (an alert needs either label)
    lines += ["", "## 1. Box level, test split (AP50 %)", "",
              "| model | source | person | knife | scissors |", "|---|---|---|---|---|"]
    for name, *_ in models:
        for source in ("coco", "hod", "sohas"):
            ims = subset(name, "test", source)
            row = []
            for c in (PERSON, KNIFE, SCISSORS):
                if sum(im.n_gt[c] for im in ims) == 0:
                    row.append("-")
                else:
                    row.append(fmt(bootstrap(lambda s, c=c: class_ap(s, c), ims)))
            lines.append(f"| {name} | {source} | " + " | ".join(row) + " |")

    lines += ["", "## 2. Image level (what raises the alert), test split", "",
              f"Threshold `tuned` = min of the per-class F2-best thresholds on the validation split; "
              f"`fixed` = {args.fixed_threshold} (today's config).", "",
              "| model | threshold | source | hazard recall % | false-alarm rate % |", "|---|---|---|---|---|"]
    for name, *_ in models:
        tuned = min(thresholds[name].values())
        for label, thr in (("fixed", args.fixed_threshold), ("tuned", tuned)):
            for source in (None, "coco", "hod", "sohas"):
                ims = subset(name, "test", source)
                rec = bootstrap(lambda s, t=thr: image_rates(s, t)[0], ims)
                far = bootstrap(lambda s, t=thr: image_rates(s, t)[1], ims)
                lines.append(f"| {name} | {label} {thr:.2f} | {source or 'all'} | {fmt(rec)} | {fmt(far)} |")

    lines += ["", f"## 2b. Recall by object size (far objects are small), test split, threshold {args.fixed_threshold}", "",
              "Box recall of knife + scissors (either label counts). Size = sqrt(box area / image area).", "",
              "| model | " + " | ".join(SIZE_BUCKETS) + " |", "|---|" + "---|" * len(SIZE_BUCKETS)]
    for name, *_ in models:
        ims = results[name]["test"]
        cells = [fmt(bootstrap(lambda s, lo=lo, hi=hi: recall_by_size(s, args.fixed_threshold, lo, hi), ims))
                 for lo, hi in SIZE_BUCKETS.values()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    counts = [sum(1 for im in results[models[0][0]]["test"] for _, sz in im.hazard_gt if lo <= sz < hi)
              for lo, hi in SIZE_BUCKETS.values()]
    lines.append("| (objects) | " + " | ".join(str(c) for c in counts) + " |")

    lines += ["", "## 3. Per-class thresholds chosen on validation (F2)", "",
              "| model | knife | scissors |", "|---|---|---|"]
    for name, *_ in models:
        lines.append(f"| {name} | {thresholds[name][KNIFE]:.2f} | {thresholds[name][SCISSORS]:.2f} |")

    lines += ["", f"## 4. Paired differences vs `{base_name}` (test split, points; + = better)", "",
              "| model | knife AP50 (all) | scissors AP50 (coco) | person AP50 (coco) | "
              f"hazard recall @ {args.fixed_threshold} | small-object recall @ {args.fixed_threshold} | "
              f"false-alarm rate @ {args.fixed_threshold} (− = better) |",
              "|---|---|---|---|---|---|---|"]
    for name, *_ in models:
        if name == base_name:
            continue
        a_all, b_all = results[base_name]["test"], results[name]["test"]
        a_coco, b_coco = subset(base_name, "test", "coco"), subset(name, "test", "coco")
        t = args.fixed_threshold
        cells = [
            fmt(paired_bootstrap(lambda s: class_ap(s, KNIFE), a_all, b_all)),
            fmt(paired_bootstrap(lambda s: class_ap(s, SCISSORS), a_coco, b_coco)),
            fmt(paired_bootstrap(lambda s: class_ap(s, PERSON), a_coco, b_coco)),
            fmt(paired_bootstrap(lambda s: image_rates(s, t)[0], a_all, b_all)),
            fmt(paired_bootstrap(lambda s: recall_by_size(s, t, *SIZE_BUCKETS["small (<3%)"]), a_all, b_all)),
            fmt(paired_bootstrap(lambda s: image_rates(s, t)[1], a_all, b_all)),
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    n_sc = sum(im.n_gt[SCISSORS] for im in subset(models[0][0], "test", "coco"))
    lines += ["", "## Notes", "",
              f"- scissors test boxes: {n_sc} (COCO only). Small n: read the interval, not the point.",
              "- Web knife labels are incomplete (checked by eye); test labels were NOT completed by a model, "
              "so a correct box on an unlabelled knife counts as a false positive for every model alike.",
              "- Sohas hard negatives (smartphone, purse, bill, card held in the hand) are in the "
              "false-alarm rate of the `sohas` rows."]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)
        (args.out.with_suffix(".json")).write_text(json.dumps(
            {n: {str(k): v for k, v in t.items()} for n, t in thresholds.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
