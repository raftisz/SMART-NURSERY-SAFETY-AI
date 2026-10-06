import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("tools").resolve()))
import train_smoke_custom as tsc  # noqa: E402


class SettingsTests(unittest.TestCase):
    def test_fixed_training_settings(self):
        self.assertEqual(tsc.BASE, "yolo11n.pt")
        self.assertEqual((tsc.IMGSZ, tsc.SEED, tsc.DEVICE), (640, 0, "mps"))

    def test_train_args_use_mps_and_never_plot_or_overwrite(self):
        a = tsc.train_args(Path("/d/data.yaml"), epochs=1, name="timing", project=Path("/runs"))
        self.assertEqual((a["device"], a["imgsz"], a["seed"], a["epochs"]), ("mps", 640, 0, 1))
        self.assertFalse(a["plots"])
        self.assertFalse(a["exist_ok"])
        self.assertTrue(a["deterministic"])

    def test_low_memory_settings_are_passed_through(self):
        a = tsc.train_args(Path("/d/data.yaml"), epochs=1, name="t", project=Path("/runs"),
                           workers=0, batch=4, imgsz=480, cache=False)
        self.assertEqual((a["batch"], a["imgsz"], a["cache"], a["workers"]), (4, 480, False, 0))
        self.assertEqual((a["device"], a["seed"]), ("mps", 0))

    def test_workers_is_passed_through(self):
        a = tsc.train_args(Path("/d/data.yaml"), epochs=1, name="t", project=Path("/runs"), workers=0)
        self.assertEqual(a["workers"], 0)


class PathGuardTests(unittest.TestCase):
    def test_test_clips_are_refused(self):
        with self.assertRaises(ValueError):
            tsc.check_paths(Path("/x/smoke_eval_clips/data.yaml"), Path("/runs"))

    def test_existing_output_model_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "m.pt"
            out.write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                tsc.check_output(out)


class InfoTests(unittest.TestCase):
    def test_train_info_records_hash_sources_counts_and_val(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp) / "m.pt"
            w.write_bytes(b"weights")
            build = {"source_license": "CC BY 4.0", "report": {"train": {"images": 10}, "val": {"images": 2}}}
            info = tsc.train_info(w, build, {"epochs": 50}, {"mAP50": 0.5}, 123.0)
            json.dumps(info)
        self.assertEqual(len(info["sha256"]), 64)
        self.assertEqual(info["dataset"]["report"]["val"]["images"], 2)
        self.assertEqual(info["val_split_metrics"]["mAP50"], 0.5)
        self.assertIn("ITLP", info["data_sources"][1])
        self.assertIn("not", info["note"])


if __name__ == "__main__":
    unittest.main()
