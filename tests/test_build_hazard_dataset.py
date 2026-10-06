"""Pure helpers of tools/build_hazard_dataset.py (no images, no model)."""

import unittest

from tools.build_hazard_dataset import (CELL_PHONE, KNIFE, PERSON, SCISSORS, format_yolo,
                                        group_near_duplicates, iou_xywh, merge_labels,
                                        parse_yolo, sohas_group, split_groups)


class ParseTests(unittest.TestCase):
    def test_box_rows(self):
        self.assertEqual(parse_yolo("43 0.5 0.5 0.2 0.4\n\n"), [(43, 0.5, 0.5, 0.2, 0.4)])

    def test_polygon_rows_become_boxes(self):
        (c, x, y, w, h), = parse_yolo("76 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6")
        self.assertEqual(c, 76)
        for got, want in zip((x, y, w, h), (0.2, 0.4, 0.2, 0.4)):
            self.assertAlmostEqual(got, want)

    def test_round_trip(self):
        boxes = [(0, 0.5, 0.5, 0.25, 0.75)]
        self.assertEqual(parse_yolo(format_yolo(boxes)), boxes)


class MergeTests(unittest.TestCase):
    gt = [(KNIFE, 0.5, 0.5, 0.1, 0.3)]

    def test_teacher_person_added_above_pseudo_conf(self):
        out = merge_labels(self.gt, [((PERSON, 0.3, 0.5, 0.2, 0.8), 0.7)], {KNIFE})
        self.assertIn((PERSON, 0.3, 0.5, 0.2, 0.8), out)

    def test_teacher_low_conf_dropped(self):
        self.assertEqual(merge_labels(self.gt, [((PERSON, 0.3, 0.5, 0.2, 0.8), 0.3)], {KNIFE}), self.gt)

    def test_teacher_duplicate_of_gt_class_dropped(self):
        same = ((KNIFE, 0.51, 0.5, 0.1, 0.3), 0.95)
        self.assertEqual(merge_labels(self.gt, [same], {KNIFE}), self.gt)

    def test_missing_knife_completed_only_when_confident(self):
        other = (KNIFE, 0.1, 0.1, 0.05, 0.1)
        self.assertEqual(merge_labels(self.gt, [(other, 0.5)], {KNIFE}), self.gt)
        self.assertIn(other, merge_labels(self.gt, [(other, 0.8)], {KNIFE}))

    def test_scissors_from_teacher_kept(self):
        s = (SCISSORS, 0.8, 0.8, 0.1, 0.1)
        self.assertIn(s, merge_labels(self.gt, [(s, 0.5)], {KNIFE, CELL_PHONE}))

    def test_teacher_scissors_on_gt_knife_dropped(self):
        on_knife = (SCISSORS, 0.5, 0.5, 0.1, 0.28)
        self.assertEqual(merge_labels(self.gt, [(on_knife, 0.9)], {KNIFE}), self.gt)

    def test_teacher_person_around_knife_kept(self):
        holder = (PERSON, 0.5, 0.5, 0.6, 0.9)   # big box around the small knife: low IoU
        self.assertIn(holder, merge_labels(self.gt, [(holder, 0.9)], {KNIFE}))

    def test_iou(self):
        self.assertAlmostEqual(iou_xywh((0, 0.5, 0.5, 0.2, 0.2), (0, 0.5, 0.5, 0.2, 0.2)), 1.0)
        self.assertEqual(iou_xywh((0, 0.1, 0.1, 0.1, 0.1), (0, 0.9, 0.9, 0.1, 0.1)), 0.0)


class SplitTests(unittest.TestCase):
    def test_near_duplicates_share_a_group(self):
        g = group_near_duplicates({"a": 0b1111, "b": 0b1110, "c": 0xFFFF_0000_FFFF_0000})
        self.assertEqual(g["a"], g["b"])
        self.assertNotEqual(g["a"], g["c"])

    def test_groups_never_straddle_splits(self):
        groups = {f"i{k}": f"g{k // 3}" for k in range(300)}
        split = split_groups(groups, {"train": 0.7, "val": 0.1, "test": 0.2})
        for gid in set(groups.values()):
            self.assertEqual(len({split[i] for i, g in groups.items() if g == gid}), 1)
        n_test = sum(s == "test" for s in split.values())
        self.assertTrue(50 <= n_test <= 70, n_test)

    def test_split_is_deterministic(self):
        groups = {f"i{k}": f"g{k}" for k in range(50)}
        self.assertEqual(split_groups(groups, {"a": 0.5, "b": 0.5}),
                         split_groups(groups, {"a": 0.5, "b": 0.5}))

    def test_sohas_group(self):
        self.assertEqual(sohas_group("KravMagaTraining21141.jpg"), "KravMagaTraining")
        self.assertEqual(sohas_group("knife_1241.jpg"), "knife")


if __name__ == "__main__":
    unittest.main()
