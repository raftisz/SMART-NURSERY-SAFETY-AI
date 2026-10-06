"""Build dataset v2 (one class: smoke) OUTSIDE the repo = dataset v1 + Indoor Fire Smoke (Zenodo).

v1  : ~/smoke_data/dataset (ScPuteri smoke boxes + ITLP no-smoke photos), copied with its
      train/val split unchanged.
Zenodo: "Indoor Fire Smoke Dataset", Putra, A.K. (Binus University), DOI 10.5281/zenodo.15826133,
      CC BY 4.0 (from Roboflow indoor-fire-smoke). Its data.yaml names the classes '0'/'1';
      class 1 = smoke, class 0 = fire (identified with two fire/smoke models, see report).
      Rules (approved 2026-10-06):
        - only class 1 boxes, remapped to 0; images without a smoke box are not used at all
        - only its train and valid splits (its test split is never read)
        - groups = file name before '.rf.'; at most CAP images, chosen by whole groups, seed 0
        - a Zenodo group with a picture near-identical (dHash <= 4 bits) to a v1 picture goes to
          that v1 picture's side; other groups are split ~85/15 (seed 0)
        - then every val group with a picture near-identical to a train picture moves to train,
          until none is left; the build asserts 0 near-duplicates between train and val
Never reads or writes anything containing 'smoke_eval_clips'; never overwrites.

Usage:
    python tools/build_smoke_dataset_v2.py --v1 ~/smoke_data/dataset \\
        --zenodo ~/smoke_data/zenodo_indoor_fire_smoke/"Indoor Fire Smoke.zip" \\
        --out ~/smoke_data/dataset_v2 --zip ~/smoke_data/dataset_v2.zip
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_smoke_dataset as v1b  # noqa: E402  (reused: split_groups, split_key, DUP_BITS)

SEED = 0
CAP = 1500
ZEN_ROOT = "Indoor Fire Smoke"
ZEN_SPLITS = ("train", "valid")          # never 'test'
ZEN_SMOKE = 1
FORBIDDEN = "smoke_eval_clips"
DUP_BITS = v1b.DUP_BITS
CREDITS = {
    "ScPuteri": "Roboflow 'Cigarette Vape Smoke' v11 by ScPuteri, CC BY 4.0",
    "ITLP": "ITLP Campus Indoor, OPR-Project (Hugging Face, commit f0908283), CC BY 4.0",
    "Zenodo": "Putra, A.K. (Binus University), Indoor Fire Smoke Dataset, Zenodo DOI 10.5281/zenodo.15826133, "
              "CC BY 4.0 (from Roboflow indoor-fire-smoke)",
}


def zenodo_smoke(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        t = line.split()
        if len(t) == 5 and int(t[0]) == ZEN_SMOKE:
            out.append(" ".join(["0"] + t[1:]))
    return out


def zenodo_group(name: str) -> str:
    return "zen:" + re.split(r"_(?:jpg|jpeg|png)\.rf\.", Path(name).name, maxsplit=1, flags=re.I)[0]


def v1_group(name: str) -> str:
    if name.startswith("neg__"):                     # ITLP: neg__session__floor__cam__file
        return "neg:" + "/".join(name.split("__")[1:-1])
    return v1b.split_key(name)


def select_groups(groups: dict[str, list[str]], cap: int = CAP) -> list[str]:
    order = sorted(groups)
    random.Random(SEED).shuffle(order)
    chosen, total = [], 0
    for g in order:
        if total + len(groups[g]) <= cap:
            chosen.append(g)
            total += len(groups[g])
    return sorted(chosen)


def dhash(img) -> int:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    return int("".join("1" if b else "0" for b in (s[:, 1:] > s[:, :-1]).flatten()), 2)


def _bits(a: np.ndarray) -> np.ndarray:
    return np.unpackbits(a.astype(np.uint64).view(np.uint8)).reshape(-1, 64).sum(1)


def near(h: int, others: np.ndarray) -> np.ndarray:
    if others.size == 0:
        return np.zeros(0, dtype=bool)
    return _bits(np.uint64(h) ^ others) <= DUP_BITS


def build(v1_dir: Path, zen_zip: Path, out: Path, cap: int = CAP) -> dict:
    v1_dir, zen_zip, out = Path(v1_dir).expanduser(), Path(zen_zip).expanduser(), Path(out).expanduser()
    for p in (v1_dir, zen_zip, out):
        if FORBIDDEN in str(p):
            raise ValueError(f"{p}: test clips must never be used for training")
    if out.exists():
        raise FileExistsError(f"{out} exists - nothing is overwritten")

    # ---- v1, split kept as it is ----
    items: dict[str, dict] = {}
    for part in ("train", "val"):
        for p in sorted((v1_dir / part / "images").iterdir()):
            lab = v1_dir / part / "labels" / (p.stem + ".txt")
            lines = [l for l in lab.read_text().splitlines() if l.strip()] if lab.is_file() else []
            src = "ITLP" if p.name.startswith("neg__") else "ScPuteri"
            items[p.name] = {"src": src, "side": part, "group": v1_group(p.name), "labels": lines,
                             "hash": dhash(cv2.imread(str(p))), "path": p}

    # ---- Zenodo candidates: train+valid, smoke images only ----
    z = zipfile.ZipFile(zen_zip)
    stats = {"zenodo_fire_only_skipped": 0, "zenodo_test_split_ignored": 0}
    zen: dict[str, dict] = {}
    for n in sorted(z.namelist()):
        parts = n.split("/")
        if len(parts) != 4 or parts[0] != ZEN_ROOT or parts[2] != "images" or not n.lower().endswith(".jpg"):
            continue
        if parts[1] == "test":
            stats["zenodo_test_split_ignored"] += 1
            continue
        if parts[1] not in ZEN_SPLITS:
            continue
        lab = f"{ZEN_ROOT}/{parts[1]}/labels/{Path(n).stem}.txt"
        lines = zenodo_smoke(z.read(lab).decode().splitlines()) if lab in z.NameToInfo else []
        if not lines:
            stats["zenodo_fire_only_skipped"] += 1
            continue
        zen["zen__" + parts[3]] = {"src": "Zenodo", "zip_name": n, "group": zenodo_group(parts[3]), "labels": lines}
    groups: dict[str, list[str]] = {}
    for name, it in zen.items():
        groups.setdefault(it["group"], []).append(name)
    chosen = select_groups(groups, cap)
    stats.update(zenodo_candidates=len(zen), zenodo_candidate_groups=len(groups), zenodo_selected_groups=len(chosen))
    for g in chosen:
        for name in groups[g]:
            it = zen[name]
            it["bytes"] = z.read(it["zip_name"])
            it["hash"] = dhash(cv2.imdecode(np.frombuffer(it["bytes"], np.uint8), cv2.IMREAD_COLOR))
            items[name] = it

    # ---- Zenodo sides: follow v1 for near-identical pictures, else ~85/15 ----
    v1_names = [n for n, it in items.items() if it["src"] != "Zenodo"]
    v1_h = np.array([items[n]["hash"] for n in v1_names], dtype=np.uint64)
    forced: dict[str, str] = {}
    forced_images = 0
    for g in chosen:
        sides = set()
        for name in groups[g]:
            hit = near(items[name]["hash"], v1_h)
            if hit.any():
                forced_images += 1
                sides |= {items[v1_names[i]]["side"] for i in np.nonzero(hit)[0]}
        if sides:
            forced[g] = "train" if "train" in sides else "val"
    free = {name: True for g in chosen if g not in forced for name in groups[g]}
    split = v1b.split_groups(free, group_of=lambda n: items[n]["group"])
    for side in ("train", "val"):
        for name in split[side]:
            items[name]["side"] = side
    for g, side in forced.items():
        for name in groups[g]:
            items[name]["side"] = side
    stats.update(zenodo_images_near_v1=forced_images, zenodo_groups_placed_with_v1=len(forced))

    # ---- move val groups with a near-identical train picture to train, until none ----
    moved = 0
    while True:
        names = sorted(items)
        train_h = np.array([items[n]["hash"] for n in names if items[n]["side"] == "train"], dtype=np.uint64)
        bad = {items[n]["group"] for n in names if items[n]["side"] == "val" and near(items[n]["hash"], train_h).any()}
        if not bad:
            break
        for n in names:
            if items[n]["side"] == "val" and items[n]["group"] in bad:
                items[n]["side"] = "train"
                moved += 1
    shared = ({it["group"] for it in items.values() if it["side"] == "train"}
              & {it["group"] for it in items.values() if it["side"] == "val"})
    assert not shared, f"groups on both sides: {sorted(shared)[:5]}"
    train_h = np.array([it["hash"] for it in items.values() if it["side"] == "train"], dtype=np.uint64)
    dupes = sum(1 for it in items.values() if it["side"] == "val" and near(it["hash"], train_h).any())
    assert dupes == 0, f"{dupes} val pictures still near-identical to train"

    # ---- write ----
    for side in ("train", "val"):
        (out / side / "images").mkdir(parents=True)
        (out / side / "labels").mkdir(parents=True)
    for name, it in sorted(items.items()):
        img_dest = out / it["side"] / "images" / name
        if it["src"] == "Zenodo":
            img_dest.write_bytes(it["bytes"])
        else:
            shutil.copy2(it["path"], img_dest)
        text = "\n".join(it["labels"])
        (out / it["side"] / "labels" / (Path(name).stem + ".txt")).write_text(text + ("\n" if text else ""))
    (out / "data.yaml").write_text("train: train/images\nval: val/images\nnames:\n  0: smoke\n")

    report: dict = {}
    for side in ("train", "val"):
        rows = [it for it in items.values() if it["side"] == side]
        report[side] = {"images": len(rows), "smoke_images": sum(1 for r in rows if r["labels"]),
                        "negative_images": sum(1 for r in rows if not r["labels"]),
                        "smoke_boxes": sum(len(r["labels"]) for r in rows),
                        "groups": len({r["group"] for r in rows}),
                        "by_source": {s: sum(1 for r in rows if r["src"] == s) for s in ("ScPuteri", "ITLP", "Zenodo")}}
    report.update(stats, images_moved_val_to_train_for_near_duplicates=moved, near_duplicates_val_vs_train=dupes)
    info = {"built": datetime.now().isoformat(timespec="minutes"), "seed": SEED, "zenodo_cap": cap,
            "v1_dataset": str(v1_dir), "zenodo_zip": str(zen_zip),
            "zenodo_class_mapping": "class 1 = smoke -> 0, class 0 = fire dropped (identified with rabahdev "
                                    "and luminous0219 fire/smoke models; data.yaml names were '0','1')",
            "rules": __doc__.split("Rules (approved 2026-10-06):")[1].split("Never reads")[0].strip(),
            "credits": CREDITS, "report": report}
    (out / "build_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    return report


def make_zip(folder: Path, zip_path: Path) -> int:
    folder, zip_path = Path(folder).expanduser(), Path(zip_path).expanduser()
    if zip_path.exists():
        raise FileExistsError(f"{zip_path} exists")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                arc = str(Path(folder.name) / p.relative_to(folder))
                assert FORBIDDEN not in arc
                z.write(p, arc)
    with zipfile.ZipFile(zip_path) as z:
        bad = [n for n in z.namelist() if FORBIDDEN in n]
        assert not bad, bad
        return len(z.namelist())


def main() -> int:
    ap = argparse.ArgumentParser(description="Build smoke dataset v2 (v1 + Zenodo Indoor Fire Smoke)")
    ap.add_argument("--v1", default="~/smoke_data/dataset")
    ap.add_argument("--zenodo", default="~/smoke_data/zenodo_indoor_fire_smoke/Indoor Fire Smoke.zip")
    ap.add_argument("--out", default="~/smoke_data/dataset_v2")
    ap.add_argument("--zip", default="~/smoke_data/dataset_v2.zip")
    ap.add_argument("--cap", type=int, default=CAP)
    args = ap.parse_args()
    report = build(Path(args.v1), Path(args.zenodo), Path(args.out), args.cap)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    n = make_zip(Path(args.out), Path(args.zip))
    print(f"zip {args.zip}: {n} entries, no '{FORBIDDEN}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
