from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def load_classes(path: str) -> Dict[str, int]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"classes file must be a non-empty {{name: id}} object: {path}")
    if any(v == 0 for v in data.values()):
        raise ValueError("class id 0 is reserved for 'unsegmented'. Use ids >= 1.")
    return data


def discover_images(input_path: str) -> List[str]:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(input_path)
    if p.is_file():
        return [str(p)]
    paths = sorted(
        str(f) for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {input_path}")
    return paths


def build_semantic_mask(
    per_class_masks: Dict[str, np.ndarray],
    classes: Dict[str, int],
    image_size: tuple,
) -> np.ndarray:
    """Compose a single (H, W) uint16 class-index mask.

    Pixel value 0 means 'unsegmented'. Overlaps are resolved by class priority:
    lower class id wins (i.e. classes listed first in classes.json).
    """
    W, H = image_size
    mask = np.zeros((H, W), dtype=np.uint16)
    for cls_name in sorted(per_class_masks.keys(), key=lambda c: -classes[c]):
        # Assign in descending priority so lowest-id is written last and "wins".
        m = per_class_masks[cls_name]
        mask[m] = classes[cls_name]
    return mask


def save_outputs(
    output_dir: str,
    image_name: str,
    semantic_mask: np.ndarray,
    per_class_masks: Dict[str, np.ndarray],
    overlay_rgb: np.ndarray,
    classes: Dict[str, int],
):
    out = Path(output_dir)
    (out / "semantic_mask").mkdir(parents=True, exist_ok=True)
    (out / "overlay").mkdir(parents=True, exist_ok=True)
    per_class_root = out / "per_class"
    per_class_root.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        (per_class_root / cls).mkdir(parents=True, exist_ok=True)

    stem = Path(image_name).stem

    # 1. Single semantic mask: PNG (uint8 if possible) + .npy (full precision)
    if semantic_mask.max() <= 255:
        Image.fromarray(semantic_mask.astype(np.uint8), mode="L").save(
            out / "semantic_mask" / f"{stem}.png"
        )
    else:
        Image.fromarray(semantic_mask).save(out / "semantic_mask" / f"{stem}.png")
    np.save(out / "semantic_mask" / f"{stem}.npy", semantic_mask)

    # 2. Per-class binary masks (always written, empty mask if class wasn't detected)
    H, W = semantic_mask.shape
    empty = np.zeros((H, W), dtype=np.uint8)
    for cls in classes:
        m = per_class_masks.get(cls)
        arr = (m.astype(np.uint8) * 255) if m is not None else empty
        Image.fromarray(arr, mode="L").save(per_class_root / cls / f"{stem}.png")

    # 3. Overlay visualization
    Image.fromarray(overlay_rgb).save(out / "overlay" / f"{stem}.png")


def write_class_legend(output_dir: str, classes: Dict[str, int]):
    """Drop the classes.json next to the outputs so the integer mask is interpretable."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "classes.json", "w") as f:
        json.dump({"unsegmented": 0, **classes}, f, indent=2)
