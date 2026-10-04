import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("tools").resolve()))
import eval_smoking_cls as esc  # noqa: E402


class LabelTests(unittest.TestCase):
    def test_label_from_name(self):
        self.assertEqual(esc.label_from_name("smoking_0020.jpg"), 1)
        self.assertEqual(esc.label_from_name("notsmoking_0004.jpg"), 0)
        with self.assertRaises(ValueError):
            esc.label_from_name("other_0001.jpg")


class CountTests(unittest.TestCase):
    def test_counts(self):
        c = esc.counts(fired=[True, False, True, False], labels=[1, 1, 0, 0])
        self.assertEqual((c.tp, c.fn, c.fp, c.tn), (1, 1, 1, 1))
        self.assertEqual(c.pos, 2)
        self.assertEqual(c.neg, 2)

    def test_verified_needs_both(self):
        fired = esc.verified([True, True, False], [0.9, 0.2, 0.99], threshold=0.5)
        self.assertEqual(fired, [True, False, False])   # classifier alone never fires


class SelectTests(unittest.TestCase):
    # 10 positives the detector catches (classifier fairly sure), 10 negatives it
    # wrongly fires on (classifier mostly unsure)
    LABELS = [1] * 10 + [0] * 10
    DET = [True] * 20
    PROB = [0.95, 0.9, 0.9, 0.85, 0.8, 0.8, 0.7, 0.6, 0.4, 0.2] + \
           [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.5, 0.6, 0.7, 0.9]

    def test_picks_fewest_false_alarms_within_recall_budget(self):
        sel = esc.select_threshold(self.DET, self.PROB, self.LABELS, max_recall_drop=0.10)
        # baseline recall 10/10; may lose at most 1 positive. 0.35 and 0.40 both keep
        # 9 positives and 4 false alarms -> the tie goes to the lower threshold
        self.assertAlmostEqual(sel.threshold, 0.35)
        self.assertEqual((sel.chosen.tp, sel.chosen.fp), (9, 4))
        self.assertTrue(sel.adopted)

    def test_not_adopted_when_no_threshold_helps(self):
        prob = [0.5] * 20                              # classifier cannot tell them apart
        sel = esc.select_threshold(self.DET, prob, self.LABELS, max_recall_drop=0.05)
        self.assertFalse(sel.adopted)

    def test_decision_rule(self):
        base = esc.Counts(tp=105, fn=7, fp=41, tn=71)
        better = esc.Counts(tp=101, fn=11, fp=12, tn=100)
        worse_recall = esc.Counts(tp=90, fn=22, fp=5, tn=107)
        self.assertTrue(esc.is_better(base, better, max_recall_drop=0.05))
        self.assertFalse(esc.is_better(base, worse_recall, max_recall_drop=0.05))
        self.assertFalse(esc.is_better(base, base, max_recall_drop=0.05))  # no FP gain


class DuplicateTests(unittest.TestCase):
    def test_near_duplicates_found_across_splits(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        b = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        a_bright = np.clip(a.astype(int) + 10, 0, 255).astype(np.uint8)   # same photo, brighter
        hashes_train = {"t1.jpg": esc.dhash(a)}
        hashes_test = {"x1.jpg": esc.dhash(a_bright), "x2.jpg": esc.dhash(b)}
        dupes = esc.near_duplicates(hashes_train, hashes_test, max_bits=6)
        self.assertEqual([(d[0], d[1]) for d in dupes], [("x1.jpg", "t1.jpg")])


class ReportTests(unittest.TestCase):
    def test_threshold_file_round_trip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            esc.save_threshold(path, 0.4, {"classifier": "x.pt"})
            data = json.loads(path.read_text())
            self.assertEqual(data["threshold"], 0.4)
            self.assertEqual(esc.load_threshold(path), 0.4)


class NotebookTests(unittest.TestCase):
    NB = Path("notebooks/smoking_cls_colab.ipynb")

    def code(self):
        nb = json.loads(self.NB.read_text(encoding="utf-8"))
        return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    def test_code_cells_are_valid_python(self):
        import ast

        for src in self.code():
            ast.parse("\n".join(line for line in src.splitlines() if not line.lstrip().startswith("!")))

    def test_pins_the_same_ultralytics_as_this_machine(self):
        try:
            import ultralytics
        except ImportError:
            self.skipTest("ultralytics is not installed on this machine")

        self.assertIn(f"ultralytics=={ultralytics.__version__}", "\n".join(self.code()))

    def test_pin_check_is_skipped_without_ultralytics(self):
        from unittest import mock

        check = NotebookTests("test_pins_the_same_ultralytics_as_this_machine")
        with mock.patch.dict(sys.modules, {"ultralytics": None}):   # import now fails
            with self.assertRaises(unittest.SkipTest):
                check.test_pins_the_same_ultralytics_as_this_machine()

    def test_reads_training_images_only(self):
        import re

        globs = re.findall(r"(\w+)\.glob\(", "\n".join(self.code()))
        self.assertTrue(globs)
        self.assertEqual(set(globs), {"TRAIN_DIR"})


if __name__ == "__main__":
    unittest.main()
