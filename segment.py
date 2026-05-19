#!/usr/bin/env python3
"""CLI entry-point for site-digital-twin segmentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

from seg import get_backend, list_backends
from seg.io_utils import (
    build_semantic_mask,
    discover_images,
    load_classes,
    save_outputs,
    write_class_legend,
)
from seg.visualization import display_overlay, make_overlay


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="segment.py",
        description="Segment a single image or a folder of images into per-pixel class labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True,
                   help="Path to an image file or a folder of images.")
    p.add_argument("-o", "--output", default="./outputs",
                   help="Directory to write semantic masks, per-class masks, and overlays.")
    p.add_argument("-c", "--classes", default="./configs/classes.json",
                   help="JSON mapping {class_name: class_id}. id 0 is reserved for 'unsegmented'.")
    p.add_argument("-b", "--backend", default="gdino_sam2", choices=list_backends(),
                   help="Segmentation backend.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Overlay transparency (0=image only, 1=mask only).")
    p.add_argument("--display", action="store_true",
                   help="Show each overlay in a window (press any key to advance).")
    p.add_argument("--verbose", action="store_true")

    # Inject per-backend args under their respective group.
    for name in list_backends():
        group = p.add_argument_group(f"{name} backend options")
        get_backend(name).add_cli_args(group)

    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available — falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    classes = load_classes(args.classes)
    image_paths = discover_images(args.input)

    if args.verbose:
        print(f"Backend:     {args.backend}")
        print(f"Device:      {args.device}")
        print(f"Classes:     {classes}")
        print(f"Images:      {len(image_paths)} file(s)")
        print(f"Output dir:  {args.output}")

    backend_cls = get_backend(args.backend)
    backend_kwargs = {k.replace("-", "_"): v for k, v in vars(args).items()}
    for k in ("classes", "device", "input", "output", "backend", "batch_size",
              "alpha", "display", "verbose"):
        backend_kwargs.pop(k, None)
    segmenter = backend_cls(classes=classes, device=args.device, **backend_kwargs)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    write_class_legend(args.output, classes)

    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Batches", unit="batch"):
        batch_paths = image_paths[i : i + args.batch_size]
        images: List[Image.Image] = []
        kept_paths: List[str] = []
        for path in batch_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                kept_paths.append(path)
            except Exception as e:
                print(f"[WARN] Skipping {path}: {e}", file=sys.stderr)

        if not images:
            continue

        results = segmenter.predict(images)

        for path, image, result in zip(kept_paths, images, results):
            semantic = build_semantic_mask(result.per_class_masks, classes, result.image_size)
            overlay = make_overlay(image, semantic, classes, alpha=args.alpha)
            save_outputs(
                output_dir=args.output,
                image_name=Path(path).name,
                semantic_mask=semantic,
                per_class_masks=result.per_class_masks,
                overlay_rgb=overlay,
                classes=classes,
            )
            if args.display:
                display_overlay(overlay, classes, window_title=Path(path).name)

    print(f"Done. Outputs written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
