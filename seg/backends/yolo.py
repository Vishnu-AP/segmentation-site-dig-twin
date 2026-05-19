"""Ultralytics YOLO segmentation backend.

The model's own class names are filtered against classes.json keys (case-insensitive).
Classes present in classes.json but absent from the model are silently ignored — the
unified output writer still creates a (possibly empty) per-class folder for them.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from PIL import Image

from seg.base import BaseSegmenter, SegmentationResult, register_backend


@register_backend("yolo")
class YoloSegmenter(BaseSegmenter):

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument("--yolo-model", default="yolo11x-seg.pt",
                            help="Path or name of an Ultralytics YOLO segmentation model.")
        parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
        parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")

    def __init__(self, classes: Dict[str, int], device: str = "cpu",
                 yolo_model: str = "yolo11x-seg.pt",
                 imgsz: int = 640, conf: float = 0.25, **kwargs):
        super().__init__(classes=classes, device=device)
        from ultralytics import YOLO
        self.model = YOLO(yolo_model)
        self.imgsz = imgsz
        self.conf = conf

        target_lower = {k.lower(): k for k in classes}
        self._yolo_id_to_class = {
            cid: target_lower[name.lower()]
            for cid, name in self.model.names.items()
            if name.lower() in target_lower
        }

    def predict(self, images: List[Image.Image]) -> List[SegmentationResult]:
        np_images = [np.array(img) for img in images]
        results = self.model(
            np_images, imgsz=self.imgsz, conf=self.conf, device=self.device,
            verbose=False, retina_masks=True,
        )

        out: List[SegmentationResult] = []
        for img, r in zip(images, results):
            W, H = img.size
            per_class: Dict[str, np.ndarray] = {}
            if r.masks is not None and r.boxes is not None:
                mask_data = r.masks.data.cpu().numpy().astype(bool)   # (N, H, W)
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for m, cid in zip(mask_data, cls_ids):
                    cls_name = self._yolo_id_to_class.get(int(cid))
                    if cls_name is None:
                        continue
                    if cls_name in per_class:
                        per_class[cls_name] |= m
                    else:
                        per_class[cls_name] = m
            out.append(SegmentationResult(image_size=(W, H), per_class_masks=per_class))
        return out
