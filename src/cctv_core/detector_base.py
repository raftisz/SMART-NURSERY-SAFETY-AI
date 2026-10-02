"""Interfaces every detection module implements.

To add a new module:
  1. subclass BaseDetector (what the model sees)
  2. subclass EventPolicy, or reuse one from events/policies.py (what counts as an event)
  3. add both to config/core.yaml - no change to the Core is needed.

Detectors must NOT send notifications, save files or apply cooldowns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import DetectorOutput, EventCandidate, Frame


class BaseDetector(ABC):
    """Wraps one model. Loaded once, then called per frame."""

    #: unique module name, e.g. "hazard_object", "smoking"
    name: str = "unnamed"

    def __init__(self, name: str | None = None, run_every_n_frames: int = 1, **_: object):
        if name:
            self.name = name
        if run_every_n_frames < 1:
            raise ValueError("run_every_n_frames must be >= 1")
        self.run_every_n_frames = run_every_n_frames

    def should_run(self, frame_index: int) -> bool:
        """Lets heavy detectors skip frames so several can share one Mac."""
        return frame_index % self.run_every_n_frames == 0

    @abstractmethod
    def load(self) -> None:
        """Load weights. Called once before the first frame."""

    @abstractmethod
    def process(self, frame: Frame) -> DetectorOutput:
        """Run the model on one frame and report what was seen."""

    def close(self) -> None:
        """Release resources. Optional."""


class EventPolicy(ABC):
    """Turns what a detector saw into zero or more event candidates."""

    @abstractmethod
    def evaluate(self, output: DetectorOutput) -> list[EventCandidate]:
        ...

    def draw(self, image, camera_id: str) -> None:
        """Optional: draw rule geometry (e.g. zones) on the live view. Default: nothing."""
