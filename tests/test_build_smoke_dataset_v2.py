import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path("tools").resolve()))
import build_smoke_dataset_v2 as b2  # noqa: E402

RNG = np.random.default_rng(0)


def img_bytes(img):
    return cv2.imencode(".jpg", img)[1].tobytes()


def fake_v1(root: Path, n_train=20, n_val=5):
    pics = {}
    for part, n in (("train", n_train), ("val", n_val)):
        (root / part / "images").mkdir(parents=True)
        (root / part / "labels").mkdir(parents=True)
        for i in range(n):
            name = f"v1{part}{i:02d}_jpg.rf.x.jpg"
            img = RNG.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            pics[name] = (part, img)
            cv2.imwrite(str(root / part / "images" / name), img)
            (root / part / "labels" / (Path(name).stem + ".txt")).write_text("0 0.5 0.5 0.2 0.2\n" if i % 2 else "")
    return pics


def fake_zenodo(path: Path, items):
    """items: [(split, file name, label lines, image)]"""
    with zipfile.ZipFile(path, "w") as z:
        for split, name, lines, img in items:
            z.writestr(f"Indoor Fire Smoke/{split}/images/{name}", img_bytes(img))
            z.writestr(f"Indoor Fire Smoke/{split}/labels/{Path(name).stem}.txt", "\n".join(lines))


class LabelTests(unittest.TestCase):
    def test_smoke_class_1_kept_as_0_and_fire_dropped(self):
        self.assertEqual(b2.zenodo_smoke(["0 0.1 0.1 0.2 0.2", "1 0.5 0.5 0.3 0.3"]), ["0 0.5 0.5 0.3 0.3"])
        self.assertEqual(b2.zenodo_smoke(["0 0.1 0.1 0.2 0.2"]), [])          # fire only -> not used at all

    def test_group_is_name_before_rf(self):
        self.assertEqual(b2.zenodo_group("FireVid36_mp4-48_jpg.rf.7277.jpg"), "zen:FireVid36_mp4-48")


class SelectTests(unittest.TestCase):
    def test_cap_is_applied_by_whole_groups_and_deterministic(self):
        groups = {f"zen:g{i}": [f"g{i}_{k}" for k in range(1 + i % 3)] for i in range(100)}
        a = b2.select_groups(groups, cap=50)
        self.assertLessEqual(sum(len(groups[g]) for g in a), 50)
        self.assertEqual(a, b2.select_groups(groups, cap=50))
        for g in a:
            self.assertIn(g, groups)                                   # whole groups only


class BuildTests(unittest.TestCase):
    def test_build_v2_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            v1 = fake_v1(tmp / "v1")
            items = []
            for i in range(40):                                        # zenodo smoke images, unique pictures
                items.append(("train" if i < 30 else "valid", f"z{i:02d}_jpg.rf.a.jpg",
                              ["1 0.5 0.5 0.3 0.3", "0 0.2 0.2 0.1 0.1"], RNG.integers(0, 255, (64, 64, 3), dtype=np.uint8)))
            items.append(("train", "fireonly_jpg.rf.b.jpg", ["0 0.5 0.5 0.3 0.3"], RNG.integers(0, 255, (64, 64, 3), dtype=np.uint8)))
            items.append(("test", "t0_jpg.rf.c.jpg", ["1 0.5 0.5 0.3 0.3"], RNG.integers(0, 255, (64, 64, 3), dtype=np.uint8)))
            # a zenodo copy of a v1 VAL picture -> must end on the v1 picture's side
            v1_val_name = next(n for n, (p, _) in v1.items() if p == "val")
            items.append(("train", "copyofv1_jpg.rf.d.jpg", ["1 0.5 0.5 0.3 0.3"], v1[v1_val_name][1]))
            fake_zenodo(tmp / "zen.zip", items)

            report = b2.build(tmp / "v1", tmp / "zen.zip", tmp / "out", cap=1500)
            sides = {p.name: part for part in ("train", "val") for p in (tmp / "out" / part / "images").iterdir()}
            self.assertEqual(report["near_duplicates_val_vs_train"], 0)
            self.assertNotIn("zen__fireonly_jpg.rf.b.jpg", sides)                # fire-only not used
            self.assertNotIn("zen__t0_jpg.rf.c.jpg", sides)                      # zenodo test split not used
            self.assertEqual(sides["zen__copyofv1_jpg.rf.d.jpg"], sides[v1_val_name])
            for n, (part, _) in v1.items():
                self.assertIn(n, sides)
            lab = (tmp / "out" / sides["zen__z00_jpg.rf.a.jpg"] / "labels" / "zen__z00_jpg.rf.a.txt").read_text().split()
            self.assertEqual(lab[0], "0")                                        # remapped smoke
            self.assertEqual(len(lab), 5)                                        # fire box dropped
            self.assertIn("0: smoke", (tmp / "out" / "data.yaml").read_text())
            with self.assertRaises(FileExistsError):
                b2.build(tmp / "v1", tmp / "zen.zip", tmp / "out", cap=1500)
            with self.assertRaises(ValueError):
                b2.build(tmp / "v1", tmp / "zen.zip", tmp / "smoke_eval_clips" / "x", cap=1500)


if __name__ == "__main__":
    unittest.main()
