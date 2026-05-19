"""Importing this module registers all built-in backends.

To add a new backend:
    1. Create seg/backends/your_backend.py
    2. Subclass BaseSegmenter and decorate the class with @register_backend("name")
    3. Import it below.
"""
from seg.backends import gdino_sam2  # noqa: F401
from seg.backends import yolo        # noqa: F401
from seg.backends import yoloe       # noqa: F401
