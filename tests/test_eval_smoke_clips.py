import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path("tools").resolve()))
import eval_smoke_clips as esc  # noqa: E402


def write_video(path, frames, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(frames):
        # frame index as whole-frame brightness (survives lossy mp4 compression)
        w.write(np.full((48, 64, 3), i * 30, dtype=np.uint8))
    w.release()


class LockedCriteriaTests(unittest.TestCase):
    def test_criteria_are_the_approved_values(self):
        self.assertEqual(esc.CONF, 0.25)
        self.assertEqual((esc.POS_CLIPS, esc.POS_MIN_DETECTED), (5, 4))
        self.assertEqual(esc.NEG_MAX_FRAMES_PER_CLIP, 2)


class ScoringTests(unittest.TestCase):
    def clip(self, name, detected_frames, frames=100, best=0.5):
        return esc.ClipResult(name, "pos" if name.startswith("smoke_") else "neg", frames,
                              detected_frames, best if detected_frames else 0.0)

    def test_pass(self):
        res = [self.clip(f"smoke_{i}.mp4", 10) for i in range(4)] + [self.clip("smoke_4.mp4", 0)] + \
              [self.clip("none_0.mp4", 2), self.clip("none_1.mp4", 0)]
        v = esc.verdict(res)
        self.assertEqual((v.pos_detected, v.pos_clips, v.status), (4, 5, "PASS"))

    def test_too_few_smoke_clips_detected_fails(self):
        res = [self.clip(f"smoke_{i}.mp4", 1 if i < 3 else 0) for i in range(5)] + [self.clip("none_0.mp4", 0)]
        self.assertEqual(esc.verdict(res).status, "FAIL")

    def test_one_negative_clip_over_two_frames_fails(self):
        res = [self.clip(f"smoke_{i}.mp4", 5) for i in range(5)] + \
              [self.clip("none_0.mp4", 0), self.clip("none_1.mp4", 3)]
        v = esc.verdict(res)
        self.assertEqual(v.status, "FAIL")
        self.assertEqual(v.neg_failed, ["none_1.mp4"])

    def test_wrong_number_of_smoke_clips_is_incomplete(self):
        res = [self.clip(f"smoke_{i}.mp4", 5) for i in range(4)] + [self.clip("none_0.mp4", 0)]
        self.assertEqual(esc.verdict(res).status, "INCOMPLETE")

    def test_no_negative_clips_is_incomplete(self):
        res = [self.clip(f"smoke_{i}.mp4", 5) for i in range(5)]
        self.assertEqual(esc.verdict(res).status, "INCOMPLETE")


class ClipListTests(unittest.TestCase):
    def test_only_smoke_and_none_clips_are_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            for n in ("smoke_a.mp4", "none_b.mp4", "other.mp4", "smoke_c.MOV", "notes.txt"):
                (Path(tmp) / n).write_bytes(b"")
            names = [p.name for p in esc.list_clips(Path(tmp))]
        self.assertEqual(names, ["none_b.mp4", "smoke_a.mp4"])


class EvaluateClipTests(unittest.TestCase):
    def test_frames_count_once_and_use_the_smoke_class_only(self):
        # fake model: frames 2 and 3 have two smoke boxes, frame 5 has smoke below CONF,
        # frame 6 has only a fire box
        def predict(image):
            i = int(round(float(image.mean()) / 30))
            return {2: [("smoke", 0.6), ("smoke", 0.4)], 3: [("smoke", 0.3)],
                    5: [("smoke", 0.2)], 6: [("fire", 0.9)]}.get(i, [])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smoke_x.mp4"
            write_video(path, 8)
            r = esc.evaluate_clip(path, predict)
        self.assertEqual((r.frames, r.detected_frames), (8, 2))
        self.assertAlmostEqual(r.best_conf, 0.6, places=1)

    def test_model_without_smoke_class_is_rejected(self):
        with self.assertRaises(ValueError):
            esc.smoke_class_id({0: "fire", 1: "person"})
        self.assertEqual(esc.smoke_class_id({0: "fire", 1: "smoke"}), 1)
        self.assertEqual(esc.smoke_class_id({0: "Smoke"}), 0)


if __name__ == "__main__":
    unittest.main()
