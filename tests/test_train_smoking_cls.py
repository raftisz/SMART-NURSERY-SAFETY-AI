import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("tools").resolve()))
import train_smoking_cls as tsc  # noqa: E402


def fake_training(folder: Path, n_smoking=358, n_not=358) -> Path:
    train = folder / "Training"
    train.mkdir(parents=True)
    for i in range(n_smoking):
        (train / f"smoking_{i:04d}.jpg").write_bytes(b"x")
    for i in range(n_not):
        (train / f"notsmoking_{i:04d}.jpg").write_bytes(b"x")
    return folder


class SplitTests(unittest.TestCase):
    def test_split_is_per_class_85_15_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = tsc.training_files(fake_training(Path(tmp)))
            split = tsc.split_training(files)
        self.assertEqual({k: len(v) for k, v in split.items()},
                         {"train/smoking": 304, "val/smoking": 54,
                          "train/notsmoking": 304, "val/notsmoking": 54})
        names = [p.name for v in split.values() for p in v]
        self.assertEqual(len(names), len(set(names)))          # no image in two places
        for key, paths in split.items():
            self.assertTrue(all(p.name.startswith(key.split("/")[1] + "_") for p in paths), key)

    def test_split_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = tsc.training_files(fake_training(Path(tmp)))
            a, b = tsc.split_training(files), tsc.split_training(files)
        self.assertEqual({k: [p.name for p in v] for k, v in a.items()},
                         {k: [p.name for p in v] for k, v in b.items()})

    def test_only_the_training_folder_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_training(Path(tmp), 3, 3)
            (root / "Testing").mkdir()
            (root / "Testing" / "smoking_9999.jpg").write_bytes(b"x")
            names = {p.name for files in tsc.training_files(root).values() for p in files}
        self.assertNotIn("smoking_9999.jpg", names)
        self.assertEqual(len(names), 6)

    def test_unknown_file_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_training(Path(tmp), 1, 1)
            (root / "Training" / "other.jpg").write_bytes(b"x")
            with self.assertRaises(ValueError):
                tsc.training_files(root)


class OutputTests(unittest.TestCase):
    def test_refuses_to_overwrite_existing_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "smoking_cls_v1_mac.pt"
            out.write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                tsc.check_output(out)

    def test_never_writes_into_the_original_smoking_model_folder(self):
        with self.assertRaises(ValueError):
            tsc.check_output(Path("models/smoking/new.pt"))

    def test_train_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp) / "m.pt"
            w.write_bytes(b"weights")
            info = tsc.train_info(w, {"train/smoking": 1}, {"epochs": 30}, "mps")
            json.dumps(info)
        self.assertEqual(len(info["sha256"]), 64)
        self.assertIn("Training only", info["data"])
        self.assertEqual(info["device"], "mps")


class SameAsNotebookTests(unittest.TestCase):
    def test_settings_match_the_colab_notebook(self):
        nb = json.loads(Path("notebooks/smoking_cls_colab.ipynb").read_text(encoding="utf-8"))
        code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        self.assertIn(f"SEED = {tsc.SEED}", code)
        self.assertIn(f"len(files) * {tsc.VAL_FRACTION}", code)
        for key in ("epochs", "imgsz", "batch", "patience"):
            self.assertIn(f"{key}={tsc.TRAIN_ARGS[key]}", code, key)


class RegistryTests(unittest.TestCase):
    WEIGHTS = Path("models/smoking_cls/smoking_cls_v1_mac.pt")

    def test_trained_weights_are_registered_with_their_hash(self):
        if not self.WEIGHTS.is_file():
            self.skipTest("verifier weights not trained on this machine")
        sha = tsc.sha256_file(self.WEIGHTS)
        info = json.loads(self.WEIGHTS.with_name("train_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["sha256"], sha)
        self.assertIn(sha, Path("models/README.md").read_text(encoding="utf-8"))

    def test_rejected_verifier_is_not_used_by_the_core(self):
        # Mendeley Testing said DO NOT ADOPT (docs/eval/smoking_cls_mendeley_test.md)
        self.assertNotIn("smoking_cls", Path("config/core.yaml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
