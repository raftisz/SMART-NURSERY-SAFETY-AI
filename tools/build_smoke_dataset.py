"""Build a one-class (smoke) YOLO dataset OUTSIDE the repo, for training on Colab.

Source: the Roboflow export "Cigarette Vape Smoke" v11 by ScPuteri (CC BY 4.0),
classes ['cigarette', 'person', 'smoke', 'vape'] -> only class 2 (smoke) is kept,
as class 0. Images without smoke are kept as negatives (empty label file).

Train/val split (about 85/15, seed 0) is by SOURCE GROUP so near-identical images
(frames of one video, re-uploads) never sit on both sides:
    group = file name before '_jpg.rf.' (or _jpeg/_png), minus a leftover '_jpeg'
            token and the whole trailing run of digits/separators
            ('S3-N1206MF_000123' -> 'S3-N1206MF', frames of one video stay together)
    exception (approved 2026-10-06, option b): scraper-numbered names ('gambar-NNNN-')
            and all-digit names are one image per group
Groups with smoke and groups without smoke are split separately so val gets both.
Then any val group holding an image that looks like a train image (dHash <= 4 of
64 bits, e.g. the same web picture under another name) is moved to train, repeated
until none is left. The build asserts that no group and no near-duplicate pair sits
on both sides.

Extra negatives (optional, e.g. indoor corridor photos) are grouped by their source
folder so frames from one camera run stay on one side.

Never reads the test clips (any path containing 'smoke_eval_clips' is refused) and
never overwrites an existing output folder.

Usage:
    python tools/build_smoke_dataset.py \\
        --source ~/Downloads/"Cigarette Vape Smoke.v11i.yolov11" --out ~/smoke_data/dataset
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

SMOKE_CLASS = 2              # 'smoke' in the ScPuteri export
VAL_FRACTION = 0.15
SEED = 0
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
FORBIDDEN = "smoke_eval_clips"


def source_group(name: str) -> str:
    base = re.split(r"_(?:jpg|jpeg|png)\.rf\.", name, maxsplit=1, flags=re.I)[0]
    base = re.sub(r"[_\-](?:jpe?g|png|webp)$", "", base, flags=re.I)
    return re.sub(r"[\s_\-()\d]+$", "", base) or "<numeric-only>"


PER_IMAGE_GROUPS = {"<numeric-only>", "gambar"}


def split_key(name: str) -> str:
    """Group used for the train/val split (see the module docstring)."""
    g = source_group(name)
    if g in PER_IMAGE_GROUPS:
        return "id:" + re.split(r"_(?:jpg|jpeg|png)\.rf\.", name, maxsplit=1, flags=re.I)[0]
    return g


def smoke_only(lines: list[str]) -> list[str]:
    """Keep smoke boxes (class SMOKE_CLASS) as class 0; drop every other class."""
    out = []
    for line in lines:
        t = line.split()
        if len(t) == 5 and int(t[0]) == SMOKE_CLASS:
            out.append(" ".join(["0"] + t[1:]))
    return out


def split_groups(items: dict[str, bool], group_of=source_group) -> dict[str, list[str]]:
    """items: {image name: has smoke}. Whole groups go to train or val, ~VAL_FRACTION of images to val."""
    groups: dict[str, list[str]] = {}
    for name in sorted(items):
        groups.setdefault(group_of(name), []).append(name)
    split = {"train": [], "val": []}
    for has_smoke in (True, False):
        stratum = sorted(g for g, names in groups.items() if any(items[n] for n in names) == has_smoke)
        random.Random(SEED).shuffle(stratum)
        total = sum(len(groups[g]) for g in stratum)
        target, val = round(total * VAL_FRACTION), 0
        for g in stratum:
            size = len(groups[g])
            if val < target and val + size <= target * 1.2 + 1:
                split["val"] += groups[g]
                val += size
            else:
                split["train"] += groups[g]
    for k in split:
        split[k].sort()
    shared = {group_of(n) for n in split["train"]} & {group_of(n) for n in split["val"]}
    assert not shared, f"groups on both sides: {sorted(shared)[:5]}"
    return split


def _dhash(path: Path) -> int | None:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    return int("".join("1" if b else "0" for b in (small[:, 1:] > small[:, :-1]).flatten()), 2)


def near_duplicates(train: list[Path], val: list[Path], max_bits: int = 4) -> list[tuple[str, str]]:
    th = {p.name: _dhash(p) for p in train}
    out = []
    for p in val:
        h = _dhash(p)
        if h is None:
            continue
        for name, t in th.items():
            if t is not None and bin(h ^ t).count("1") <= max_bits:
                out.append((p.name, name))
                break
    return out


DUP_BITS = 4


def move_near_duplicates_to_train(split: dict[str, list[str]], paths: dict[str, Path],
                                  group_of: dict[str, str]) -> int:
    """Move every val group that has an image near-identical to a train image into train.
    Repeats until no such pair is left. Returns the number of images moved."""
    hashes = {n: _dhash(paths[n]) for n in paths}
    moved = 0
    while True:
        train_h = [hashes[n] for n in split["train"] if hashes[n] is not None]
        bad = {group_of[v] for v in split["val"] if hashes[v] is not None
               and any(bin(hashes[v] ^ t).count("1") <= DUP_BITS for t in train_h)}
        if not bad:
            return moved
        move = [v for v in split["val"] if group_of[v] in bad]
        split["val"] = [v for v in split["val"] if group_of[v] not in bad]
        split["train"] = sorted(split["train"] + move)
        moved += len(move)


def build(source: Path, out: Path, extra_negatives: Path | None = None) -> dict:
    source, out = Path(source).expanduser(), Path(out).expanduser()
    for p in (source, out, extra_negatives):
        if p is not None and FORBIDDEN in str(p):
            raise ValueError(f"{p}: test clips must never be used for training")
    if out.exists():
        raise FileExistsError(f"{out} exists - choose a new folder, nothing is overwritten")

    # source images + smoke-only labels
    images: dict[str, Path] = {}
    labels: dict[str, list[str]] = {}
    for split_dir in ("train", "valid", "test"):
        img_dir = source / split_dir / "images"
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            assert p.name not in images, f"duplicate image name {p.name}"
            lab = source / split_dir / "labels" / (p.stem + ".txt")
            images[p.name] = p
            labels[p.name] = smoke_only(lab.read_text().splitlines() if lab.is_file() else [])
    group_of = {n: split_key(n) for n in images}

    if extra_negatives is not None:
        root = Path(extra_negatives).expanduser()
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in IMAGE_SUFFIXES:
                name = "neg__" + "__".join(p.relative_to(root).parts)
                assert name not in images
                images[name], labels[name] = p, []
                group_of[name] = "neg:" + str(p.parent.relative_to(root))

    split = split_groups({n: bool(labels[n]) for n in images}, group_of=lambda n: group_of[n])
    moved = move_near_duplicates_to_train(split, images, group_of)
    shared = {group_of[n] for n in split["train"]} & {group_of[n] for n in split["val"]}
    assert not shared, f"groups on both sides: {sorted(shared)[:5]}"

    report: dict = {}
    for part, names in split.items():
        (out / part / "images").mkdir(parents=True)
        (out / part / "labels").mkdir(parents=True)
        for n in names:
            dest_name = Path(n).stem + images[n].suffix.lower()
            shutil.copy2(images[n], out / part / "images" / dest_name)
            text = "\n".join(labels[n])
            (out / part / "labels" / (Path(n).stem + ".txt")).write_text(text + ("\n" if text else ""))
        report[part] = {
            "images": len(names),
            "smoke_images": sum(1 for n in names if labels[n]),
            "negative_images": sum(1 for n in names if not labels[n]),
            "extra_negative_images": sum(1 for n in names if n.startswith("neg__")),
            "smoke_boxes": sum(len(labels[n]) for n in names),
            "groups": len({group_of[n] for n in names}),
        }
    (out / "data.yaml").write_text("train: train/images\nval: val/images\nnames:\n  0: smoke\n")
    dupes = near_duplicates([out / "train" / "images" / p.name for p in (out / "train" / "images").iterdir()],
                            [out / "val" / "images" / p.name for p in (out / "val" / "images").iterdir()],
                            max_bits=DUP_BITS)
    assert not dupes, f"near-duplicate images on both sides: {dupes[:5]}"
    report["near_duplicates_val_vs_train"] = len(dupes)
    report["groups"] = len(set(group_of.values()))
    report["images_moved_val_to_train_for_near_duplicates"] = moved
    info = {"built": datetime.now().isoformat(timespec="minutes"), "source": str(source),
            "source_license": "CC BY 4.0 (Roboflow 'Cigarette Vape Smoke' v11 by ScPuteri)",
            "extra_negatives": str(extra_negatives) if extra_negatives else None,
            "kept_class": f"{SMOKE_CLASS} (smoke) -> 0", "val_fraction": VAL_FRACTION, "seed": SEED,
            "group_rule": "name before _jpg.rf., minus _jpeg token and trailing digits/separators; "
                          "; gambar-NNNN and all-digit names one image per group (option b); "
                          f"val groups with a near-duplicate (dHash <= {DUP_BITS} bits) in train moved to train",
            "report": report}
    (out / "build_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the one-class smoke dataset (outside the repo)")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="~/smoke_data/dataset")
    ap.add_argument("--extra-negatives", default=None, help="folder of no-smoke images (approved sources only)")
    args = ap.parse_args()
    report = build(Path(args.source), Path(args.out),
                   Path(args.extra_negatives) if args.extra_negatives else None)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
