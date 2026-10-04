import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.cctv_core.stream.camera_source import CameraSource

sys.path.insert(0, str(Path("tools").resolve()))
import eval_clips  # noqa: E402
from eval_clips import Clip, ClipResult  # noqa: E402


def write_video(path, frames, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for _ in range(frames):
        w.write(np.zeros((48, 64, 3), dtype=np.uint8))
    w.release()


# A config whose only module "sees" scissors in every frame; confirming needs 3 s.
FAKE_CFG = {
    "cameras": [{"id": "cam0", "source": 0, "flip": True},
                {"id": "qa", "source": "", "flip": False}],
    "modules": [{
        "detector": {"class": "tests.fakes:FakeDetector"},
        "policy": {"class": "src.cctv_core.events.policies:LabelMatchPolicy",
                   "params": {"event_type": "hazard_object", "labels": ["scissors"]}},
    }],
    "events": {"types": {"hazard_object": {"min_duration_s": 3.0}}},
    "notifications": {"channels": {"console": {"enabled": True}, "line": {"enabled": True}},
                      "routes": [{"channels": ["console", "line"], "event_types": "*"}]},
}


class NamingTests(unittest.TestCase):
    def test_parse_clip_name(self):
        self.assertEqual(eval_clips.parse_clip_name("fight_pos_s1_01_hug.mp4"),
                         ("fight", "pos", "s1"))
        self.assertEqual(eval_clips.parse_clip_name("outarea_neg_s2_03.MOV"),
                         ("out_of_area", "neg", "s2"))
        self.assertEqual(eval_clips.parse_clip_name("normal_neg_s1_01.mp4"),
                         (None, "neg", "s1"))

    def test_bad_names_are_rejected(self):
        for bad in ("fight_s1_01.mp4", "dance_pos_s1_01.mp4", "normal_pos_s1_01.mp4",
                    "fight_pos_x1_01.mp4"):
            with self.assertRaises(ValueError, msg=bad):
                eval_clips.parse_clip_name(bad)

    def test_init_labels_splits_by_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("fight_pos_s1_01.mp4", "fight_neg_s2_01.mp4",
                         "hazard_pos_s2_01.mov", "notes.txt"):
                (Path(tmp) / name).write_bytes(b"")
            clips = eval_clips.init_labels(Path(tmp), tune_sessions={"s1"})
            by_name = {Path(c.path).name: c for c in clips}
            self.assertEqual(set(by_name), {"fight_pos_s1_01.mp4", "fight_neg_s2_01.mp4",
                                            "hazard_pos_s2_01.mov"})
            self.assertEqual(by_name["fight_pos_s1_01.mp4"].split, "tune")
            self.assertEqual(by_name["fight_pos_s1_01.mp4"].expected, {"fight"})
            self.assertEqual(by_name["fight_neg_s2_01.mp4"].split, "test")
            self.assertEqual(by_name["fight_neg_s2_01.mp4"].expected, set())

    def test_labels_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            clips = [Clip("a.mp4", "test", {"fight", "climbing"}, "two kids"),
                     Clip("b.mp4", "tune", set())]
            path = Path(tmp) / "labels.csv"
            eval_clips.write_labels(clips, path)
            back = eval_clips.read_labels(path)
            self.assertEqual([(c.split, c.expected, c.note) for c in back],
                             [(c.split, c.expected, c.note) for c in clips])
            # relative clip paths are resolved against the CSV's folder
            self.assertEqual(back[0].path, str(Path(tmp) / "a.mp4"))


class ShootingScriptTests(unittest.TestCase):
    """docs/eval/qa_shooting_script.md must stay in step with the naming rule."""

    def test_every_example_name_parses_and_covers_every_module(self):
        import re

        text = Path("docs/eval/qa_shooting_script.md").read_text(encoding="utf-8")
        names = re.findall(r"`([a-z]+_(?:pos|neg)_s\d+_\d+[a-z0-9_]*\.mp4)`", text)
        self.assertGreaterEqual(len(names), 10)
        types = set()
        for name in names:
            event_type, _, _ = eval_clips.parse_clip_name(name)
            types.add(event_type)
        self.assertEqual(types, set(eval_clips.MODULE_TYPES.values()))


class ShootingScriptS2Tests(unittest.TestCase):
    """Round s2 must have exactly the clips the out_of_area / fight gates need."""

    EXPECTED = {("outarea", "pos", "crouchstay"): 10, ("outarea", "neg", "adultstand"): 10,
                ("outarea", "neg", "crouchpass"): 10, ("fight", "pos", "staged"): 6,
                ("fight", "neg", "daily"): 8}

    def test_s2_gate_groups_have_the_required_clip_counts(self):
        import re

        text = Path("docs/eval/qa_shooting_script.md").read_text(encoding="utf-8")
        rows = re.findall(r"`((outarea|fight)_(pos|neg)_s2_(\d+)_([a-z]+)_[a-z0-9_]*\.mp4)` ถึง "
                          r"`((outarea|fight)_(pos|neg)_s2_(\d+)_([a-z]+)_[a-z0-9_]*\.mp4)` \| (\d+) \|", text)
        self.assertTrue(rows)
        counts = {}
        for first, mod, pol, a, grp, last, mod2, pol2, b, grp2, stated in rows:
            for name in (first, last):
                eval_clips.parse_clip_name(name)            # both ends follow the naming rule
            self.assertEqual((mod, pol, grp), (mod2, pol2, grp2), first)
            n = int(b) - int(a) + 1
            self.assertEqual(n, int(stated), first)         # the "จำนวน" column matches the range
            counts[(mod, pol, grp)] = counts.get((mod, pol, grp), 0) + n
        self.assertEqual(counts, self.EXPECTED)


class GateDocTests(unittest.TestCase):
    DOC = Path("docs/eval/decision_gates_outarea_fight.md")

    def test_gates_recorded_with_date_and_numbers(self):
        text = self.DOC.read_text(encoding="utf-8")
        self.assertIn("2026-10-04", text)
        self.assertIn("ก่อนเห็นผล", text)
        for figure in ("≥ 8/10", "≤ 1/10", "≥ 9/10", "≥ 4/6", "≤ 2/8"):
            self.assertIn(figure, text)


class ClockTests(unittest.TestCase):
    def test_video_clock_follows_frames_not_wall_time(self):
        clock = eval_clips.VideoClock(fps=10, start=100.0)
        self.assertEqual([round(clock(), 3) for _ in range(3)], [100.0, 100.1, 100.2])

    def test_camera_source_reports_video_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            write_video(path, 3, fps=12)
            src = CameraSource(str(path))
            self.assertAlmostEqual(src.fps, 12.0)
            src.release()


class ScoreTests(unittest.TestCase):
    def test_score_counts_per_type(self):
        r = [
            ClipResult(Clip("p1", "test", {"fight"}), {"fight": 1}, 10.0, 100),
            ClipResult(Clip("p2", "test", {"fight"}), {}, 10.0, 100),
            ClipResult(Clip("n1", "test", set()), {"fight": 2, "smoking": 1}, 1800.0, 100),
            ClipResult(Clip("n2", "test", set()), {}, 1800.0, 100),
        ]
        s = eval_clips.score(r)
        self.assertEqual((s["fight"].tp, s["fight"].fn), (1, 1))
        self.assertEqual((s["fight"].fp_clips, s["fight"].neg_clips), (1, 2))
        self.assertEqual(s["fight"].fp_events, 2)
        self.assertAlmostEqual(s["fight"].fp_per_hour, 2 / 1.0)     # 2 events in 1 hour of negatives
        self.assertEqual((s["smoking"].tp, s["smoking"].fn, s["smoking"].fp_clips), (0, 0, 1))
        self.assertEqual(s["smoking"].neg_clips, 4)                 # nobody expected smoking


def gate_results(group, n, fired, module="outarea", polarity="pos", event="out_of_area"):
    out = []
    for i in range(1, n + 1):
        name = f"{module}_{polarity}_s2_{i:02d}_{group}_x.mp4"
        expected = {event} if polarity == "pos" else set()
        out.append(ClipResult(Clip(f"/clips/{name}", "test", expected),
                              {event: 1} if i <= fired else {}, 5.0, 50))
    return out


class GateTests(unittest.TestCase):
    def results(self):
        return (gate_results("crouchstay", 10, 8)                               # 8/10 alert -> PASS
                + gate_results("adultstand", 10, 2, polarity="neg")             # 2/10 false -> FAIL
                + gate_results("crouchpass", 9, 0, polarity="neg")              # only 9 clips -> INCOMPLETE
                + gate_results("staged", 6, 4, module="fight", event="fight")   # 4/6 -> PASS
                + gate_results("daily", 8, 2, module="fight", polarity="neg", event="fight"))  # 2/8 -> PASS

    def test_each_gate_status(self):
        status = {g.rule.group: g.status for g in eval_clips.evaluate_gates(self.results())}
        self.assertEqual(status, {"crouchstay": "PASS", "adultstand": "FAIL", "crouchpass": "INCOMPLETE",
                                  "staged": "PASS", "daily": "PASS"})

    def test_gate_whose_event_type_was_not_run_is_not_judged(self):
        rows = eval_clips.evaluate_gates(self.results(), available={"out_of_area"})
        status = {g.rule.group: g.status for g in rows}
        self.assertEqual(status["staged"], "NOT RUN")       # fight needs the pose module
        self.assertEqual(status["crouchstay"], "PASS")

    def test_extra_clips_need_checking(self):
        rows = eval_clips.evaluate_gates(gate_results("staged", 7, 7, module="fight", event="fight"))
        self.assertEqual(next(g for g in rows if g.rule.group == "staged").status, "CHECK")

    def test_tune_clips_never_count(self):
        tune = gate_results("staged", 6, 6, module="fight", event="fight")
        for r in tune:
            r.clip.split = "tune"
        rows = eval_clips.evaluate_gates(tune)
        self.assertEqual(next(g for g in rows if g.rule.group == "staged").status, "INCOMPLETE")

    def test_report_has_a_gate_section(self):
        import tempfile

        res = self.results()
        with tempfile.TemporaryDirectory() as tmp:
            _, summary = eval_clips.write_report(res, eval_clips.score(res), Path(tmp), "t", [],
                                                 gates=eval_clips.evaluate_gates(res))
            text = summary.read_text(encoding="utf-8")
        self.assertIn("Decision gates", text)
        self.assertIn("FAIL", text)
        self.assertIn("2/10", text)

    def test_gate_numbers_match_the_approved_document(self):
        doc = Path("docs/eval/decision_gates_outarea_fight.md").read_text(encoding="utf-8")
        for rule in eval_clips.GATES:
            sign = "≥" if rule.op == ">=" else "≤"
            self.assertIn(f"{sign} {rule.k}/{rule.n}", doc, rule.text)


class RunClipTests(unittest.TestCase):
    def test_eval_config_never_sends_line_and_leaves_input_alone(self):
        cfg = eval_clips.eval_config(FAKE_CFG)
        self.assertFalse(cfg["notifications"]["channels"]["line"]["enabled"])
        self.assertFalse(cfg["notifications"]["channels"]["console"]["enabled"])
        self.assertTrue(FAKE_CFG["notifications"]["channels"]["line"]["enabled"])

    def test_durations_use_video_time(self):
        """min_duration_s 3 s: a 2 s clip must not fire, a 4 s clip must, however
        fast the machine processes the frames."""
        with tempfile.TemporaryDirectory() as tmp:
            short, long_ = Path(tmp) / "short.mp4", Path(tmp) / "long.mp4"
            write_video(short, 20, fps=10)
            write_video(long_, 40, fps=10)
            ev = eval_clips.Evaluator(FAKE_CFG, camera_id="qa")
            r_short = ev.run_clip(Clip(str(short), "test", {"hazard_object"}))
            r_long = ev.run_clip(Clip(str(long_), "test", {"hazard_object"}))
            ev.close()
        self.assertEqual(r_short.fired, {})
        self.assertEqual(r_long.fired, {"hazard_object": 1})
        self.assertAlmostEqual(r_long.duration_s, 4.0)
        self.assertEqual(r_long.frames, 40)

    def test_models_load_once_for_many_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.mp4"
            write_video(path, 5)
            ev = eval_clips.Evaluator(FAKE_CFG, camera_id="qa")
            for _ in range(3):
                ev.run_clip(Clip(str(path), "test", set()))
            self.assertEqual(ev.load_count, 1)
            ev.close()


if __name__ == "__main__":
    unittest.main()
