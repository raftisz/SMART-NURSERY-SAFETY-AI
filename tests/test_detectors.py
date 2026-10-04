import unittest
from types import SimpleNamespace

from src.cctv_core.schemas import BBox, Detection, DetectorOutput, Frame, Keypoint
from src.detectors.climbing_pose import ClimbingPosePolicy, classify_pose
from src.detectors.hazard_object import HazardObjectDetector
from src.detectors.smoking import SmokingDetector
from src.detectors.yolo_label import resolve_weights

SMOKING_SHA = "fded2de373f09f73628930ea4b7249efba061fadac73ca796dc82ffdf63841e2"


def pose(nose_y=100, wrist_y=200, hip_y=300, ankle_y=450, conf=0.9):
    """17 COCO keypoints; only nose, wrists, hips and ankles matter to the rule."""
    k = [Keypoint(0, 250, conf) for _ in range(17)]
    k[0] = Keypoint(0, nose_y, conf)
    k[9] = k[10] = Keypoint(0, wrist_y, conf)
    k[11] = k[12] = Keypoint(0, hip_y, conf)
    k[15] = k[16] = Keypoint(0, ankle_y, conf)
    return k


class ClimbingRuleTests(unittest.TestCase):
    def test_standing_is_safe(self):
        self.assertEqual(classify_pose(pose()), "SAFE")

    def test_hands_up_only_is_warning(self):
        self.assertEqual(classify_pose(pose(wrist_y=50)), "WARNING")

    def test_foot_raised_only_is_warning(self):
        self.assertEqual(classify_pose(pose(ankle_y=250)), "WARNING")

    def test_both_is_danger(self):
        self.assertEqual(classify_pose(pose(wrist_y=50, ankle_y=250)), "DANGER")

    def test_low_confidence_keypoints_ignored(self):
        self.assertEqual(classify_pose(pose(wrist_y=50, ankle_y=250, conf=0.1)), "SAFE")

    def test_too_few_keypoints(self):
        self.assertEqual(classify_pose(pose()[:10]), "NORMAL")

    def test_lying_down_is_not_climbing(self):
        # lying on a mat, head left, feet right, arms stretched past the head:
        # wrist "above" the nose and ankle "above" the hips only by a few pixels
        k = [Keypoint(0, 0, 0.9) for _ in range(17)]
        k[0] = Keypoint(100, 300, 0.9)                              # nose
        k[5], k[6] = Keypoint(150, 300, 0.9), Keypoint(150, 305, 0.9)  # shoulders
        k[9], k[10] = Keypoint(60, 295, 0.9), Keypoint(60, 290, 0.9)   # wrists
        k[11], k[12] = Keypoint(300, 305, 0.9), Keypoint(300, 300, 0.9)  # hips
        k[15], k[16] = Keypoint(500, 298, 0.9), Keypoint(500, 302, 0.9)  # ankles
        self.assertNotIn(classify_pose(k), ("DANGER", "WARNING"))

    def test_hidden_hips_do_not_count_as_raised_feet(self):
        k = pose()
        k[11] = k[12] = Keypoint(0, 500, 0.05)  # hips occluded: position is a guess
        self.assertEqual(classify_pose(k), "SAFE")

    def _output(self, kpts):
        d = Detection("person", 0.8, BBox(0, 0, 10, 10), keypoints=kpts)
        return DetectorOutput("climbing_pose", "cam0", 0, 1.0, [d])

    def test_policy_danger_only_by_default(self):
        p = ClimbingPosePolicy()
        self.assertEqual(p.evaluate(self._output(pose(wrist_y=50))), [])
        c = p.evaluate(self._output(pose(wrist_y=50, ankle_y=250)))
        self.assertEqual(c[0].event_type, "climbing")
        self.assertEqual(c[0].extra["pose_status"], "DANGER")

    def test_policy_warning_when_allowed(self):
        c = ClimbingPosePolicy(require_both=False).evaluate(self._output(pose(wrist_y=50)))
        self.assertEqual(c[0].extra["pose_status"], "WARNING")


class YoloAdapterTests(unittest.TestCase):
    def test_process_converts_legacy_detections(self):
        det = HazardObjectDetector()
        legacy = [SimpleNamespace(class_name="scissors", confidence=0.66, x1=1, y1=2, x2=3, y2=4)]
        det._yolo = SimpleNamespace(detect=lambda img, conf, imgsz, nms_iou: (legacy, 5.0, 1))
        out = det.process(Frame("cam0", 7, 12.0, image=None))
        self.assertEqual(out.detector, "hazard_object")
        self.assertEqual((out.frame_index, out.timestamp), (7, 12.0))
        self.assertEqual(out.detections[0].label, "scissors")

    def test_process_before_load_fails_clearly(self):
        with self.assertRaises(RuntimeError):
            SmokingDetector().process(Frame("cam0", 0, 0.0))

    def test_defaults(self):
        s = SmokingDetector()
        self.assertEqual((s.weights, s.conf, s.target_classes),
                         ("models/smoking/best.pt", 0.5, ["cigarette", "smoking"]))
        self.assertEqual(HazardObjectDetector().target_classes, ["person", "scissors", "knife"])

    def test_project_weights_resolved_with_hash(self):
        path, digest = resolve_weights("models/smoking/best.pt")
        self.assertTrue(path.endswith("models/smoking/best.pt"))
        self.assertEqual(digest, SMOKING_SHA)  # weight file unchanged

    def test_stock_weights_passed_through(self):
        # a name with no file in the repo is passed through for Ultralytics to download
        # (not "yolo11n.pt": that file exists once it has been downloaded on a machine)
        name = "stock-weights-not-on-disk.pt"
        self.assertEqual(resolve_weights(name), (name, None))


if __name__ == "__main__":
    unittest.main()
