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


def build_parser():
    """Returns (parser, backend_dests) where backend_dests maps
    backend_name -> set of argparse dest names that backend owns."""
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

    backend_dests = {}
    for name in list_backends():
        group = p.add_argument_group(f"{name} backend options")
        before = {a.dest for a in p._actions}
        get_backend(name).add_cli_args(group)
        after = {a.dest for a in p._actions}
        backend_dests[name] = after - before

    return p, backend_dests


def print_config_banner(args, classes, image_paths, backend_kwargs):
    bar = "=" * 60
    print(bar)
    print(" segmentation-site-dig-twin")
    print(bar)
    print(f"  Backend     : {args.backend}")
    print(f"  Device      : {args.device}")
    print(f"  Input       : {args.input}  ({len(image_paths)} image(s))")
    print(f"  Output dir  : {args.output}")
    print(f"  Classes     : {args.classes}")
    for name, cid in classes.items():
        print(f"                  {cid}: {name}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Overlay α   : {args.alpha}    Display: {args.display}")
    if backend_kwargs:
        print(f"  {args.backend} options:")
        for k, v in backend_kwargs.items():
            print(f"      --{k.replace('_', '-')}: {v}")
    print(bar)


def main(argv: List[str] | None = None) -> int:
    parser, backend_dests = build_parser()
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available — falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    classes = load_classes(args.classes)
    image_paths = discover_images(args.input)

    # Warn about flags belonging to inactive backends that were set to non-defaults.
    defaults = {a.dest: a.default for a in parser._actions}
    for name, dests in backend_dests.items():
        if name == args.backend:
            continue
        stray = [d for d in dests if getattr(args, d) != defaults[d]]
        if stray:
            flags = ", ".join(f"--{d.replace('_','-')}" for d in stray)
            print(f"[WARN] {flags} belong(s) to backend '{name}', "
                  f"but active backend is '{args.backend}'. "
                  f"Use --backend {name} to apply them.", file=sys.stderr)

    backend_cls = get_backend(args.backend)
    backend_kwargs = {d: getattr(args, d) for d in backend_dests[args.backend]}

    print_config_banner(args, classes, image_paths, backend_kwargs)

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
