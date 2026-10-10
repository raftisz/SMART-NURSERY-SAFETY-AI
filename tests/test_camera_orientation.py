"""Video files with a rotation tag (phone clips) must come out the right way up."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from src.cctv_core.stream.camera_source import CameraSource


def marked_frame():
    """4x6 frame, only the top-left pixel is white, so every rotation is distinguishable."""
    f = np.zeros((4, 6, 3), dtype=np.uint8)
    f[0, 0] = 255
    return f


class TaggedCapture:
    """cv2.VideoCapture stand-in for a file with a rotation tag.

    `honours_auto_off`: whether the backend accepts CAP_PROP_ORIENTATION_AUTO = 0
    (if not, OpenCV keeps rotating the frames itself and they arrive upright)."""

    def __init__(self, meta, auto=1.0, honours_auto_off=True, frame=None):
        self.props = {cv2.CAP_PROP_ORIENTATION_META: float(meta), cv2.CAP_PROP_ORIENTATION_AUTO: float(auto)}
        self.honours_auto_off = honours_auto_off
        self.frame = marked_frame() if frame is None else frame

    def isOpened(self):
        return True

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_ORIENTATION_AUTO and not self.honours_auto_off:
            return False
        self.props[prop] = float(value)
        return True

    def read(self):
        return True, self.frame.copy()

    def release(self):
        pass


class OrientationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.clip = Path(tmp.name) / "clip.mp4"
        self.clip.write_bytes(b"")

    def open(self, cap, **kw):
        with mock.patch("cv2.VideoCapture", return_value=cap):
            return CameraSource(str(self.clip), **kw)

    def test_180_tag_is_applied(self):
        src = self.open(TaggedCapture(180))
        self.assertEqual(src.rotation, 180)
        np.testing.assert_array_equal(src.read(), cv2.rotate(marked_frame(), cv2.ROTATE_180))

    def test_90_and_270_tags_match_opencv_auto_rotation(self):
        src = self.open(TaggedCapture(90))
        np.testing.assert_array_equal(src.read(), cv2.rotate(marked_frame(), cv2.ROTATE_90_CLOCKWISE))
        src = self.open(TaggedCapture(270))
        np.testing.assert_array_equal(src.read(), cv2.rotate(marked_frame(), cv2.ROTATE_90_COUNTERCLOCKWISE))

    def test_no_tag_leaves_frames_alone(self):
        src = self.open(TaggedCapture(0))
        self.assertEqual(src.rotation, 0)
        np.testing.assert_array_equal(src.read(), marked_frame())

    def test_never_rotates_twice_when_opencv_keeps_rotating(self):
        upright = cv2.rotate(marked_frame(), cv2.ROTATE_180)   # what OpenCV would already deliver
        src = self.open(TaggedCapture(180, honours_auto_off=False, frame=upright))
        self.assertEqual(src.rotation, 0)
        np.testing.assert_array_equal(src.read(), upright)

    def test_rotation_happens_before_the_mirror_flip(self):
        src = self.open(TaggedCapture(180), flip=True)
        expected = cv2.flip(cv2.rotate(marked_frame(), cv2.ROTATE_180), 1)
        np.testing.assert_array_equal(src.read(), expected)

    def test_unusual_tag_value_is_ignored(self):
        src = self.open(TaggedCapture(45))
        self.assertEqual(src.rotation, 0)
        np.testing.assert_array_equal(src.read(), marked_frame())

    def test_still_images_and_webcams_are_not_rotated(self):
        img = sorted(Path("data/test_frames").glob("*.jpg"))[0]
        self.assertEqual(CameraSource(str(img), loop_image=False).rotation, 0)


if __name__ == "__main__":
    unittest.main()
