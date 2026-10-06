import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("tools").resolve()))
import build_smoke_dataset as bsd  # noqa: E402


def fake_source(root: Path, items):
    """items: [(file name, [label lines])] in a Roboflow 'valid/' layout."""
    (root / "valid" / "images").mkdir(parents=True)
    (root / "valid" / "labels").mkdir(parents=True)
    for name, lines in items:
        (root / "valid" / "images" / name).write_bytes(b"jpg")
        (root / "valid" / "labels" / (Path(name).stem + ".txt")).write_text("\n".join(lines))
    return root


class GroupTests(unittest.TestCase):
    def test_group_from_source_name(self):
        g = bsd.source_group
        self.assertEqual(g("S3-N1206MF_000123_jpg.rf.abc.jpg"), "S3-N1206MF")
        self.assertEqual(g("smoke_a12_jpg.rf.def.jpg"), "smoke_a")
        self.assertEqual(g("000044_jpeg_jpg.rf.22cc.jpg"), "<numeric-only>")
        self.assertEqual(g("photo-13_png.rf.4d0a.jpg"), "photo")
        self.assertEqual(g("vaping42_jpg.rf.x.jpg"), "vaping")
        self.assertEqual(g("gambar-3280-_jpg.rf.b739.jpg"), "gambar")          # trailing '-' after the number
        self.assertEqual(g("photo_2024-03-18_08-52-44_jpg.rf.9e12.jpg"), "photo")
        self.assertEqual(g("OIP-11-_jpeg_jpg.rf.5afc.jpg"), "OIP")


class SplitKeyTests(unittest.TestCase):
    def test_scraper_numbered_and_numeric_names_are_one_image_per_group(self):
        k = bsd.split_key
        self.assertEqual(k("gambar-3280-_jpg.rf.b739.jpg"), "id:gambar-3280-")
        self.assertNotEqual(k("gambar-3280-_jpg.rf.b739.jpg"), k("gambar-3281-_jpg.rf.c.jpg"))
        self.assertEqual(k("000044_jpeg_jpg.rf.22cc.jpg"), "id:000044_jpeg")
        self.assertEqual(k("S3-N1206MF_000123_jpg.rf.abc.jpg"), "S3-N1206MF")   # video frames stay together


class LabelTests(unittest.TestCase):
    def test_only_smoke_boxes_kept_as_class_0(self):
        out = bsd.smoke_only(["0 0.1 0.1 0.2 0.2", "2 0.5 0.5 0.4 0.4", "3 0.3 0.3 0.1 0.1", "2 0.7 0.7 0.1 0.1"])
        self.assertEqual(out, ["0 0.5 0.5 0.4 0.4", "0 0.7 0.7 0.1 0.1"])

    def test_no_smoke_gives_empty_label(self):
        self.assertEqual(bsd.smoke_only(["1 0.5 0.5 0.9 0.9", "3 0.2 0.2 0.1 0.1"]), [])


class SplitTests(unittest.TestCase):
    def test_groups_never_cross_and_val_is_about_15_percent(self):
        items = {}
        for g in range(60):
            for k in range(1 + g % 4):
                items[f"src{chr(65 + g % 26)}{chr(65 + g // 26)}_{k}_jpg.rf.h.jpg"] = g % 3 == 0
        split = bsd.split_groups(items)
        groups = {s: {bsd.source_group(n) for n in names} for s, names in split.items()}
        self.assertFalse(groups["train"] & groups["val"])
        frac = len(split["val"]) / len(items)
        self.assertTrue(0.10 <= frac <= 0.22, frac)
        self.assertTrue(any(items[n] for n in split["val"]))        # val gets smoke images too
        self.assertEqual(split, bsd.split_groups(items))              # deterministic


class BuildTests(unittest.TestCase):
    def test_build_writes_dataset_and_refuses_test_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = fake_source(Path(tmp) / "src", [
                (f"grp{i}_{k}_jpg.rf.h.jpg", ["2 0.5 0.5 0.4 0.4"] if i % 2 else ["1 0.5 0.5 0.9 0.9"])
                for i in range(30) for k in range(2)])
            out = Path(tmp) / "dataset"
            report = bsd.build(src, out)
            self.assertEqual(report["train"]["images"] + report["val"]["images"], 60)
            self.assertTrue((out / "data.yaml").read_text().count("0: smoke") == 1)
            labels = list((out / "train" / "labels").glob("*.txt")) + list((out / "val" / "labels").glob("*.txt"))
            self.assertEqual(len(labels), 60)
            with self.assertRaises(ValueError):
                bsd.build(Path(tmp) / "smoke_eval_clips", Path(tmp) / "x")
            with self.assertRaises(FileExistsError):
                bsd.build(src, out)                                   # never overwrites


class NearDuplicateTests(unittest.TestCase):
    def test_same_picture_under_different_names_stays_on_one_side(self):
        import cv2
        import numpy as np

        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            items = []
            for i in range(40):
                items.append((f"src{i:02d}x_jpg.rf.h.jpg", ["2 0.5 0.5 0.4 0.4"] if i % 2 else []))
            src = fake_source(Path(tmp) / "src", items)
            pics = {}
            for name, _ in items:
                img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
                pics[name] = img
                cv2.imwrite(str(src / "valid" / "images" / name), img)
            # the same picture again under unrelated names (a web re-upload)
            dupes = {f"OIP-{k}-_jpg.rf.d{k}.jpg": items[k][0] for k in range(10)}
            for other, original in dupes.items():
                cv2.imwrite(str(src / "valid" / "images" / other), pics[original])
                (src / "valid" / "labels" / (Path(other).stem + ".txt")).write_text("")
            report = bsd.build(src, Path(tmp) / "out")
            self.assertEqual(report["near_duplicates_val_vs_train"], 0)
            self.assertGreater(report["val"]["images"], 0)          # no chain collapse into one group
            sides = {p.name: part for part in ("train", "val")
                     for p in (Path(tmp) / "out" / part / "images").iterdir()}
            for other, original in dupes.items():
                self.assertEqual(sides[other], sides[original], other)


if __name__ == "__main__":
    unittest.main()
