"""YOLOE (Ultralytics open-vocabulary segmentation) backend.

YOLOE is the open-vocab successor to YOLO-World. Unlike plain YOLO-seg it does
NOT use a fixed COCO label set — you call `model.set_classes(names, text_pe)`
before inference, and the model returns masks for those exact class names.

This means classes.json keys are used directly as text prompts.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from PIL import Image

from seg.base import BaseSegmenter, SegmentationResult, register_backend


@register_backend("yoloe")
class YoloESegmenter(BaseSegmenter):

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument("--yoloe-model", default="yoloe-11l-seg.pt",
                            help="Path or hub name of an Ultralytics YOLOE seg model "
                                 "(e.g. yoloe-11s-seg.pt, yoloe-11l-seg.pt).")
        parser.add_argument("--yoloe-imgsz", type=int, default=640,
                            help="Inference image size.")
        parser.add_argument("--yoloe-conf", type=float, default=0.25,
                            help="Confidence threshold.")

    def __init__(self, classes: Dict[str, int], device: str = "cpu",
                 yoloe_model: str = "yoloe-11l-seg.pt",
                 yoloe_imgsz: int = 640, yoloe_conf: float = 0.25, **kwargs):
        super().__init__(classes=classes, device=device)
        try:
            from ultralytics import YOLOE
        except ImportError as e:
            raise ImportError(
                "YOLOE requires a recent ultralytics package: pip install -U ultralytics"
            ) from e

        self.model = YOLOE(yoloe_model)
        self.imgsz = yoloe_imgsz
        self.conf = yoloe_conf

        # The order of class_names matters: model emits class ids 0..N-1
        # in the same order we pass them here.
        self._class_names = list(classes.keys())
        self.model.set_classes(self._class_names,
                               self.model.get_text_pe(self._class_names))

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
                    if cid < 0 or cid >= len(self._class_names):
                        continue
                    cls_name = self._class_names[cid]
                    if cls_name in per_class:
                        per_class[cls_name] |= m
                    else:
                        per_class[cls_name] = m
            out.append(SegmentationResult(image_size=(W, H), per_class_masks=per_class))
        return out
