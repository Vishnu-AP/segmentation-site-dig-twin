from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Type

import numpy as np
from PIL import Image


@dataclass
class SegmentationResult:
    """Result for a single image.

    per_class_masks: class_name -> bool array of shape (H, W). A pixel may belong
    to multiple class masks if backends produce overlapping detections; the
    output writer resolves overlaps using class priority (lower class_id wins).
    """
    image_size: tuple                                  # (W, H) per PIL convention
    per_class_masks: Dict[str, np.ndarray] = field(default_factory=dict)


_REGISTRY: Dict[str, Type["BaseSegmenter"]] = {}


def register_backend(name: str):
    def deco(cls: Type["BaseSegmenter"]):
        if name in _REGISTRY:
            raise ValueError(f"Backend '{name}' already registered")
        _REGISTRY[name] = cls
        cls.backend_name = name
        return cls
    return deco


def get_backend(name: str) -> Type["BaseSegmenter"]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown backend '{name}'. Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_backends() -> List[str]:
    return sorted(_REGISTRY.keys())


class BaseSegmenter(ABC):
    """Backend interface.

    Subclasses register themselves with @register_backend("name") and implement
    `predict`. Each backend is given the class mapping at construction time and
    decides for itself how to use it (open-vocab prompt, label filter, etc.).
    """
    backend_name: str = "base"

    def __init__(self, classes: Dict[str, int], device: str = "cpu", **kwargs):
        self.classes = classes
        self.device = device

    @classmethod
    def add_cli_args(cls, parser):
        """Backends may add their own argparse args. Default: nothing."""
        return

    @abstractmethod
    def predict(self, images: List[Image.Image]) -> List[SegmentationResult]:
        ...
