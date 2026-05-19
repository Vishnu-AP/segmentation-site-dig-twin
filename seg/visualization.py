from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image


def _palette(n: int) -> np.ndarray:
    """Deterministic, visually distinct colors for up to ~20 classes."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab20", max(n, 1))
    cols = (np.array([cmap(i)[:3] for i in range(n)]) * 255).astype(np.uint8)
    return cols


def make_overlay(
    image: Image.Image,
    semantic_mask: np.ndarray,
    classes: Dict[str, int],
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend the colored semantic mask onto the image. Returns RGB uint8 (H, W, 3)."""
    img = np.array(image.convert("RGB"))
    H, W = img.shape[:2]

    class_ids = sorted(classes.values())
    palette = _palette(len(class_ids))
    id_to_color = {cid: palette[i] for i, cid in enumerate(class_ids)}

    overlay = img.copy()
    for cid, color in id_to_color.items():
        m = semantic_mask == cid
        if not m.any():
            continue
        overlay[m] = (alpha * color + (1 - alpha) * overlay[m]).astype(np.uint8)
    return overlay


def display_overlay(overlay_rgb: np.ndarray, classes: Dict[str, int], window_title: str = "overlay"):
    """Show the overlay with a class-color legend. Press any key to advance."""
    import cv2
    legend_h = 30 * (len(classes) + 1)
    legend = np.full((legend_h, 220, 3), 255, dtype=np.uint8)
    palette = _palette(len(classes))
    for i, (name, cid) in enumerate(sorted(classes.items(), key=lambda kv: kv[1])):
        color = palette[i].tolist()
        y = 20 + 30 * i
        cv2.rectangle(legend, (10, y - 15), (40, y + 5), color[::-1], -1)
        cv2.putText(legend, f"{cid}: {name}", (50, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    img_bgr = overlay_rgb[..., ::-1]
    h = img_bgr.shape[0]
    if legend.shape[0] < h:
        pad = np.full((h - legend.shape[0], legend.shape[1], 3), 255, dtype=np.uint8)
        legend = np.vstack([legend, pad])
    elif legend.shape[0] > h:
        legend = legend[:h]
    panel = np.hstack([img_bgr, legend])
    cv2.imshow(window_title, panel)
    cv2.waitKey(0)
    cv2.destroyWindow(window_title)
