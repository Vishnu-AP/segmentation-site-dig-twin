"""Grounding-DINO (open-vocab detector) + SAM2 (segmenter).

The class names from classes.json are passed as text prompts to GroundingDINO;
detected boxes feed SAM2 to produce per-instance masks, which are then merged
per class into the unified SegmentationResult format.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from seg.base import BaseSegmenter, SegmentationResult, register_backend


@register_backend("gdino_sam2")
class GDinoSam2Segmenter(BaseSegmenter):

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument("--gdino-model", default="IDEA-Research/grounding-dino-tiny",
                            help="HF model id for Grounding-DINO.")
        parser.add_argument("--sam-model", default="facebook/sam2.1-hiera-large",
                            help="HF model id for SAM2.")
        parser.add_argument("--box-threshold", type=float, default=0.5,
                            help="GroundingDINO box confidence threshold.")
        parser.add_argument("--text-threshold", type=float, default=0.3,
                            help="GroundingDINO text confidence threshold.")
        parser.add_argument("--iou-threshold", type=float, default=0.5,
                            help="Minimum SAM IoU score to keep a mask.")

    def __init__(self, classes: Dict[str, int], device: str = "cpu",
                 gdino_model: str = "IDEA-Research/grounding-dino-tiny",
                 sam_model: str = "facebook/sam2.1-hiera-large",
                 box_threshold: float = 0.5,
                 text_threshold: float = 0.3,
                 iou_threshold: float = 0.5,
                 **kwargs):
        super().__init__(classes=classes, device=device)
        from transformers import (
            AutoProcessor,
            AutoModelForZeroShotObjectDetection,
            Sam2Model,
            Sam2Processor,
        )
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()

        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_model).to(device)
        self.gdino_processor = AutoProcessor.from_pretrained(gdino_model)
        self.sam_model = Sam2Model.from_pretrained(sam_model).to(device)
        self.sam_processor = Sam2Processor.from_pretrained(sam_model)

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.iou_threshold = iou_threshold

    @torch.no_grad()
    def predict(self, images: List[Image.Image]) -> List[SegmentationResult]:
        class_names = list(self.classes.keys())
        prompts = [class_names for _ in images]

        inputs = self.gdino_processor(
            images=images, text=prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        outputs = self.gdino_model(**inputs)

        gdino_results = self.gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=torch.tensor([img.size[::-1] for img in images]).to(self.device),
        )

        results: List[SegmentationResult] = []
        for image, det in zip(images, gdino_results):
            W, H = image.size
            per_class: Dict[str, np.ndarray] = {}

            boxes = det["boxes"].cpu().numpy().tolist()
            labels = det["text_labels"]

            if len(boxes) == 0:
                results.append(SegmentationResult(image_size=(W, H), per_class_masks=per_class))
                continue

            sam_in = self.sam_processor(
                images=image, input_boxes=[boxes], return_tensors="pt"
            ).to(self.device)
            sam_out = self.sam_model(**sam_in)

            masks = self.sam_processor.post_process_masks(
                sam_out.pred_masks.cpu(), sam_in["original_sizes"]
            )[0]  # (N, 3, H, W) — 3 candidate masks per box

            best_ids = sam_out.iou_scores.argmax(dim=-1).cpu().squeeze(0)
            best_ious = sam_out.iou_scores.max(dim=-1).values.cpu().squeeze(0)

            for j, label in enumerate(labels):
                if best_ious[j].item() < self.iou_threshold:
                    continue
                # GroundingDINO may emit substrings/phrases; match to closest class key.
                cls_name = self._resolve_label(label)
                if cls_name is None:
                    continue
                mask = masks[j, best_ids[j]].cpu().numpy().astype(bool)
                if cls_name in per_class:
                    per_class[cls_name] |= mask
                else:
                    per_class[cls_name] = mask

            results.append(SegmentationResult(image_size=(W, H), per_class_masks=per_class))

        return results

    def _resolve_label(self, label: str):
        if label in self.classes:
            return label
        label_l = label.lower().strip()
        for cls in self.classes:
            if cls.lower() == label_l or cls.lower() in label_l or label_l in cls.lower():
                return cls
        return None
