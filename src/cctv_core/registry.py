"""Build detectors and policies from config entries.

A config entry looks like:
    class: "src.detectors.smoking:SmokingDetector"
    params: {weights: "models/smoking/best.pt"}

Classes are imported by path, so a new module needs a new file and a config
entry - the Core itself does not change.
"""

from __future__ import annotations

import importlib
from typing import Any

from .detector_base import BaseDetector, EventPolicy


def import_object(path: str) -> Any:
    """Import "package.module:Name"."""
    if ":" not in path:
        raise ValueError(f"expected 'module:Name', got {path!r}")
    module_name, attr = path.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"{attr!r} not found in {module_name!r}") from exc


def _build(spec: dict[str, Any], base: type) -> Any:
    if "class" not in spec:
        raise ValueError(f"config entry has no 'class': {spec!r}")
    cls = import_object(spec["class"])
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise TypeError(f"{spec['class']} is not a {base.__name__}")
    return cls(**(spec.get("params") or {}))


def build_detector(spec: dict[str, Any]) -> BaseDetector:
    return _build(spec, BaseDetector)


def build_policy(spec: dict[str, Any]) -> EventPolicy:
    return _build(spec, EventPolicy)
