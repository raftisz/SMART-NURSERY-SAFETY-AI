"""Smoking: the project's own model models/smoking/best.pt (cigarette/face/smoking).

conf 0.50 is the default of the original Streamlit app (legacy/streamlit_smoking_app.py).
"""

from __future__ import annotations

from .yolo_label import YoloLabelDetector


class SmokingDetector(YoloLabelDetector):
    name = "smoking"

    def __init__(self, weights: str = "models/smoking/best.pt",
                 target_classes: list[str] | None = None, conf: float = 0.5, **kwargs):
        super().__init__(weights, target_classes or ["cigarette", "smoking"], conf=conf, **kwargs)
