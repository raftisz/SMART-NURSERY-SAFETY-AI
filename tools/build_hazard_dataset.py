"""Build the hazard_object fine-tuning dataset (YOLO format, all 80 COCO classes).

Why 80 classes and not just person/scissors/knife
-------------------------------------------------
The new web data has ground truth for `knife` (and `smartphone`) only. If we
trained a 3-class head, Ultralytics would re-initialise the classifier and the
model would have to re-learn `scissors` from the handful of COCO scissors
images we can replay - it would get worse at the primary gate class. Keeping
the COCO head and filling every other class with *teacher pseudo-labels*
(knowledge distillation from a larger COCO model) lets the new knife data
change the knife decision without erasing what the base model knows.

Sources (all public; licences in docs/eval/hazard_v1_dataset.md)
- COCO val2017 (Ultralytics mirror, GT for 80 classes). It is NOT in the
  training set of the pretrained yolo11 weights, so its held-out part is a fair
  test of the baseline. Split A = replay for training, C = validation,
  B = test (never trained on, never used for selection).
- HOD knife (normal + hard cases), Ha et al., WACV-W 2024. GT class 5 = knife.
- Sohas weapon detection (DaSCI, CC BY-SA 4.0): knife held in hand + objects
  handled the same way (smartphone, purse, bill, card) = hard negatives.
  GT knife -> COCO knife (43), GT smartphone -> COCO cell phone (67).
  Pistol-only images are dropped (not a nursery object).

Leakage control
- HOD: near-duplicate images (dHash <= 6 of 64 bits, same rule as
  docs/eval/smoking_cls_dupes.md) are grouped and a group never straddles splits.
- Sohas: the authors' own train/test split is kept; video frames of one clip
  (same filename prefix) stay in one split.

Label rules for non-COCO images (merge_labels)
- GT boxes are kept as-is.
- Teacher boxes of a class that has GT in this source (knife, cell phone) are
  only added when they do not overlap any GT box of that class (IoU < 0.3) AND
  have conf >= complete_conf: the web labels miss knives (checked by eye:
  two knives on a table, one labelled), and an unlabelled knife would be
  taught as background.
- Teacher boxes of every other class are added at conf >= pseudo_conf,
  EXCEPT when they overlap a GT box of another class (IoU >= 0.3): the GT wins.
  Measured on the first build: 363 of 413 teacher "scissors" boxes in the knife
  photos sat on a GT knife - the teacher's own knife/scissors confusion, which
  would otherwise be taught to the student as truth.
- Hard negatives: when a per-source cap is used, at least `neg_fraction` of the
  kept images are images WITHOUT a knife (phones, purses, bills, cards in a hand).
- Test splits keep GT only (no teacher), so the test is not biased toward
  models that behave like the teacher.

Usage
    python tools/build_hazard_dataset.py --coco <coco dir> --hod <HOD repo> \
        --sohas <Sohas_weapon-Detection-YOLOv5 dir> --out <work dir> \
        [--teacher yolo11m.pt] [--max-train-per-source 1200]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SEED = 0
KNIFE, SCISSORS, PERSON, CELL_PHONE = 43, 76, 0, 67
HOD_KNIFE = 5
SOHAS_TO_COCO = {2: KNIFE, 1: CELL_PHONE}  # 0 pistol, 3 purse, 4 bill, 5 card -> no COCO class
SOHAS_PISTOL = 0
DHASH_BITS = 6
MAX_SIDE = 960  # images are stored resized; training/eval use imgsz <= 960


# --------------------------------------------------------------------------
# pure helpers (unit-tested in tests/test_build_hazard_dataset.py)
# --------------------------------------------------------------------------

Box = tuple[int, float, float, float, float]  # cls, xc, yc, w, h (normalised)


def parse_yolo(text: str) -> list[Box]:
    """YOLO txt -> boxes. Polygon rows (COCO segments) become their bounding box."""
    out: list[Box] = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        v = [float(x) for x in p[1:]]
        if len(v) == 4:
            out.append((c, *v))
        else:
            xs, ys = v[0::2], v[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            out.append((c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return out


def format_yolo(boxes: list[Box]) -> str:
    return "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes)


def iou_xywh(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def conflicts_with_gt(box: Box, gt: list[Box], overlap_iou: float = 0.3) -> bool:
    """A teacher box sitting on a GT box of ANOTHER class (e.g. 'scissors' on a GT knife)."""
    return any(g[0] != box[0] and iou_xywh(g, box) >= overlap_iou for g in gt)


def merge_labels(gt: list[Box], teacher: list[tuple[Box, float]], gt_classes: set[int],
                 pseudo_conf: float = 0.4, complete_conf: float = 0.6,
                 overlap_iou: float = 0.3) -> list[Box]:
    """GT + teacher pseudo-labels (see module docstring for the rules)."""
    out = list(gt)
    for box, conf in teacher:
        c = box[0]
        if conflicts_with_gt(box, gt, overlap_iou):
            continue
        if c in gt_classes:
            if conf < complete_conf:
                continue
            if any(g[0] == c and iou_xywh(g, box) >= overlap_iou for g in out):
                continue
        elif conf < pseudo_conf:
            continue
        elif any(g[0] == c and iou_xywh(g, box) >= 0.5 for g in out):
            continue
        out.append(box)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def group_near_duplicates(hashes: dict[str, int], max_bits: int = DHASH_BITS) -> dict[str, str]:
    """Union-find over items whose dHash differs by <= max_bits. Returns item -> group root."""
    parent = {k: k for k in hashes}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    keys = sorted(hashes)
    # bucket by 4 x 16-bit bands: two hashes within 6 bits share at least one band? not
    # guaranteed, so compare all pairs; n is a few thousand -> a few million cheap ops.
    vals = [hashes[k] for k in keys]
    for i in range(len(keys)):
        hi = vals[i]
        for j in range(i + 1, len(keys)):
            if hamming(hi, vals[j]) <= max_bits:
                ri, rj = find(keys[i]), find(keys[j])
                if ri != rj:
                    parent[rj] = ri
    return {k: find(k) for k in keys}


def split_groups(groups: dict[str, str], fractions: dict[str, float], seed: int = SEED) -> dict[str, str]:
    """Assign whole groups to splits so each split gets ~fraction of ITEMS. item -> split."""
    members: dict[str, list[str]] = defaultdict(list)
    for item, g in groups.items():
        members[g].append(item)
    order = sorted(members)
    random.Random(seed).shuffle(order)
    total = len(groups)
    names = list(fractions)
    target = {n: fractions[n] * total for n in names}
    filled = {n: 0 for n in names}
    out: dict[str, str] = {}
    for g in order:
        # the split furthest below its target takes the next group
        n = max(names, key=lambda s: (target[s] - filled[s]) / max(target[s], 1e-9))
        for item in members[g]:
            out[item] = n
        filled[n] += len(members[g])
    return out


def sohas_group(name: str) -> str:
    """Clip/source prefix of a Sohas file: 'KravMagaTraining21141.jpg' -> 'KravMagaTraining'."""
    stem = Path(name).stem
    return stem.rstrip("0123456789").rstrip("_") or stem


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def dhash(path: Path) -> int:
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (im[:, 1:] > im[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


@dataclass
class Item:
    src: Path
    source: str          # coco | hod | sohas
    split: str           # train | val | test
    gt: list[Box]
    gt_classes: set[int] = field(default_factory=set)


def collect_coco(coco_dir: Path, n_train: int, n_val: int) -> list[Item]:
    img_dir, lab_dir = coco_dir / "images" / "val2017", coco_dir / "labels" / "val2017"
    files = sorted(img_dir.glob("*.jpg"))
    rng = random.Random(SEED)
    labels = {f: parse_yolo((lab_dir / (f.stem + ".txt")).read_text())
              if (lab_dir / (f.stem + ".txt")).exists() else [] for f in files}
    hazard = [f for f in files if any(b[0] in (KNIFE, SCISSORS) for b in labels[f])]
    other = [f for f in files if f not in set(hazard)]
    rng.shuffle(hazard)
    rng.shuffle(other)
    # hazard images: 40% train / 10% val / 50% test, so the test keeps half the scissors
    nh = len(hazard)
    h_tr, h_va = hazard[: int(nh * 0.4)], hazard[int(nh * 0.4): int(nh * 0.5)]
    h_te = hazard[int(nh * 0.5):]
    o_tr, o_va, o_te = other[:n_train], other[n_train:n_train + n_val], other[n_train + n_val:]
    items = []
    for split, fs in (("train", h_tr + o_tr), ("val", h_va + o_va), ("test", h_te + o_te)):
        items += [Item(f, "coco", split, labels[f], set(range(80))) for f in fs]
    return items


def collect_hod(hod_dir: Path) -> list[Item]:
    files = sorted((hod_dir / "dataset" / "class" / "knife").glob("*_cases/jpg/*.jpg"))
    print(f"[hod] hashing {len(files)} images for near-duplicate groups ...")
    hashes = {str(f): dhash(f) for f in files}
    groups = group_near_duplicates(hashes)
    split = split_groups(groups, {"train": 0.7, "val": 0.1, "test": 0.2})
    n_groups = len(set(groups.values()))
    print(f"[hod] {len(files)} images in {n_groups} groups")
    items = []
    for f in files:
        lab = f.parent.parent / "txt" / (f.stem + ".txt")
        gt = [(KNIFE, *b[1:]) for b in parse_yolo(lab.read_text()) if b[0] == HOD_KNIFE]
        items.append(Item(f, "hod", split[str(f)], gt, {KNIFE}))
    return items


def collect_sohas(sohas_dir: Path) -> list[Item]:
    base = sohas_dir / "obj_train_data"
    items = []
    train_files = sorted((base / "images" / "train").glob("*"))
    groups = {str(f): sohas_group(f.name) for f in train_files}
    # big sources (knife_, pistol_, ...) are web photos, not clips: group them per file
    sizes: dict[str, int] = defaultdict(int)
    for g in groups.values():
        sizes[g] += 1
    groups = {k: (g if sizes[g] < 300 else k) for k, g in groups.items()}
    tv = split_groups(groups, {"train": 0.9, "val": 0.1})
    for split_dir in ("train", "test"):
        for f in sorted((base / "images" / split_dir).glob("*")):
            lab = base / "labels" / split_dir / (f.stem + ".txt")
            raw = parse_yolo(lab.read_text()) if lab.exists() else []
            if raw and all(b[0] == SOHAS_PISTOL for b in raw):
                continue
            gt = [(SOHAS_TO_COCO[b[0]], *b[1:]) for b in raw if b[0] in SOHAS_TO_COCO]
            split = "test" if split_dir == "test" else tv[str(f)]
            items.append(Item(f, "sohas", split, gt, {KNIFE, CELL_PHONE}))
    return items


def write_image(src: Path, dst: Path) -> None:
    import cv2
    im = cv2.imread(str(src))
    if im is None:
        raise ValueError(f"unreadable image {src}")
    h, w = im.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", type=Path, required=True, help="dir with images/val2017 and labels/val2017")
    ap.add_argument("--hod", type=Path, required=True, help="HOD-Benchmark-Dataset repo")
    ap.add_argument("--sohas", type=Path, required=True, help="Sohas_weapon-Detection-YOLOv5 dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--teacher", default="yolo11m.pt", help="COCO model for pseudo-labels")
    ap.add_argument("--pseudo-conf", type=float, default=0.4)
    ap.add_argument("--complete-conf", type=float, default=0.6)
    ap.add_argument("--coco-train", type=int, default=1500)
    ap.add_argument("--coco-val", type=int, default=300)
    ap.add_argument("--max-train-per-source", type=int, default=0,
                    help="cap train images per web source (0 = all). For CPU-only runs.")
    ap.add_argument("--max-val-per-source", type=int, default=0)
    ap.add_argument("--neg-fraction", type=float, default=0.3,
                    help="with a cap: share of kept web images that have NO knife (hard negatives)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    items = collect_coco(args.coco, args.coco_train, args.coco_val) + collect_hod(args.hod) \
        + collect_sohas(args.sohas)
    rng = random.Random(SEED)
    for cap, split in ((args.max_train_per_source, "train"), (args.max_val_per_source, "val")):
        if not cap:
            continue
        kept = []
        for src in ("hod", "sohas"):
            pool = [i for i in items if i.source == src and i.split == split]
            rng.shuffle(pool)
            pos = [i for i in pool if any(b[0] == KNIFE for b in i.gt)]
            neg = [i for i in pool if not any(b[0] == KNIFE for b in i.gt)]
            n_neg = min(len(neg), round(cap * args.neg_fraction))
            kept += pos[:cap - n_neg] + neg[:n_neg]
        items = [i for i in items if i.source == "coco" or i.split != split] + kept

    out = args.out
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    need_teacher = [i for i in items if i.source != "coco" and i.split != "test"]
    teacher_boxes: dict[str, list[tuple[Box, float]]] = {}
    if need_teacher:
        from ultralytics import YOLO
        model = YOLO(args.teacher)
        print(f"[teacher] {args.teacher} on {len(need_teacher)} images ...")
        for k in range(0, len(need_teacher), 16):
            batch = need_teacher[k:k + 16]
            results = model.predict([str(i.src) for i in batch], imgsz=640, conf=min(args.pseudo_conf, 0.25),
                                    device=args.device, verbose=False)
            for it, r in zip(batch, results):
                tb = []
                for xywhn, c, p in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    tb.append(((int(c), *xywhn), float(p)))
                teacher_boxes[str(it.src)] = tb
            if (k // 16) % 20 == 0:
                print(f"[teacher] {k + len(batch)}/{len(need_teacher)}", flush=True)

    manifest = []
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for n, it in enumerate(items):
        name = f"{it.source}_{n:05d}_{it.src.stem}"[:120]
        labels = it.gt
        if str(it.src) in teacher_boxes:
            labels = merge_labels(it.gt, teacher_boxes[str(it.src)], it.gt_classes,
                                  args.pseudo_conf, args.complete_conf)
        write_image(it.src, out / "images" / it.split / f"{name}.jpg")
        (out / "labels" / it.split / f"{name}.txt").write_text(format_yolo(labels))
        key = f"{it.split}/{it.source}"
        stats[key]["images"] += 1
        stats[key]["knife_boxes"] += sum(b[0] == KNIFE for b in labels)
        stats[key]["scissors_boxes"] += sum(b[0] == SCISSORS for b in labels)
        stats[key]["person_boxes"] += sum(b[0] == PERSON for b in labels)
        stats[key]["added_by_teacher"] += len(labels) - len(it.gt)
        manifest.append({"file": name, "src": str(it.src), "source": it.source, "split": it.split})

    import yaml
    names = coco_names(args.teacher)
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
         "names": names}, sort_keys=False, allow_unicode=True))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


def coco_names(weights: str) -> dict[int, str]:
    """The 80 COCO class names in Ultralytics order, read from the pretrained weights."""
    from ultralytics import YOLO
    names = dict(YOLO(weights).names)
    if len(names) != 80 or names[KNIFE] != "knife" or names[SCISSORS] != "scissors":
        raise ValueError(f"{weights} is not a COCO-80 model")
    return names


if __name__ == "__main__":
    sys.exit(main())
