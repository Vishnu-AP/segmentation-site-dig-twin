# segmentation-site-dig-twin

Class-aware image segmentation for site digital-twin pipelines. Point it at an
image (or a folder of images) and a class list, and it produces:

1. **A single semantic mask** per image — one `(H, W)` integer label per pixel,
   where `0` is *unsegmented* and `1..N` are the classes from `classes.json`.
2. **Per-class binary masks** — one folder per class, one PNG per image.
3. **Overlay visualisations** — the semantic mask blended onto the original
   image, saved to disk and optionally shown in a window.

The pipeline is **backend-agnostic**: there is a small registry of segmentation
backends (`gdino_sam2`, `yolo`, …) and adding your own is a single file. Pick one
with `--backend`.

---

## Install

```bash
git clone https://github.com/Vishnu-AP/segmentation-site-dig-twin.git
cd segmentation-site-dig-twin
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If you don't plan to use the YOLO backend you can omit `ultralytics`.

GPU is strongly recommended for the default `gdino_sam2` backend.

---

## Quick start

```bash
# Single image
python segment.py -i path/to/image.jpg -o ./outputs

# Folder of images
python segment.py -i path/to/images/ -o ./outputs

# Different backend, custom classes, also show overlays in a window
python segment.py -i imgs/ -o out/ \
    --backend yolo --classes configs/classes.json --display
```

---

## Classes file

`configs/classes.json` is a flat `{name: id}` mapping. **Id `0` is reserved**
for *unsegmented*; use ids `>= 1`. Lower ids win when two classes overlap on a
pixel.

```json
{
    "chair":  1,
    "bottle": 2,
    "human":  3
}
```

The same file is copied into `<output>/classes.json` next to the masks so that
the integer label map is interpretable on its own.

---

## Output layout

```
outputs/
├── classes.json                       # copy of the class legend (incl. 0=unsegmented)
├── semantic_mask/
│   ├── frame_001.png                  # (H, W) uint8 label map (uint16 if >255 classes)
│   └── frame_001.npy                  # same data, full precision
├── per_class/
│   ├── chair/   frame_001.png         # binary mask, 0/255
│   ├── bottle/  frame_001.png
│   └── human/   frame_001.png
└── overlay/
    └── frame_001.png                  # colored mask blended on the image
```

Every class in `classes.json` gets its own folder — even classes that were
never detected (you get all-zero masks). This keeps downstream tooling simple.

---

## Available backends

Listed by `python segment.py --help` under "... backend options".

### `gdino_sam2` (default)
Grounding-DINO (open-vocab detector) produces boxes from the class-name text
prompts; SAM2 turns each box into a mask. Works with arbitrary class names —
you do not need a model trained on your classes.

Backend args:
- `--gdino-model` (default `IDEA-Research/grounding-dino-tiny`)
- `--sam-model` (default `facebook/sam2.1-hiera-large`)
- `--box-threshold`, `--text-threshold`, `--iou-threshold`

### `yolo`
Ultralytics YOLO segmentation. Fast, but only detects classes the model knows
about. Class-name matching is case-insensitive; classes the model doesn't know
will simply be empty in the output.

Backend args:
- `--yolo-model` (path or hub name, e.g. `yolo11x-seg.pt`)
- `--imgsz`, `--conf`

### `yoloe`
Ultralytics **YOLOE** — open-vocabulary segmentation (the YOLO-Worldv2
successor). Unlike plain YOLO-seg, YOLOE accepts your class names directly as
text prompts via `model.set_classes(...)` and returns masks for them, so
classes.json keys are used verbatim — no name-matching required.

> ℹ️ Plain **YOLO-World** does **not** work with the `yolo` backend: it's
> detection-only (no masks) and needs `set_classes` to bind prompts. Use the
> `yoloe` backend for open-vocab YOLO-style segmentation.

Backend args:
- `--yoloe-model` (path or hub name, e.g. `yoloe-11s-seg.pt`, `yoloe-11l-seg.pt`)
- `--yoloe-imgsz`, `--yoloe-conf`

```bash
python segment.py -i imgs/ -o out/ \
    --backend yoloe \
    --yoloe-model yoloe-11l-seg.pt \
    -c configs/classes.json
```

Requires a recent `ultralytics` version with YOLOE support (`pip install -U ultralytics`).

---

## Adding a new backend

1. Drop `seg/backends/your_backend.py`:

   ```python
   from typing import Dict, List
   from PIL import Image
   import numpy as np

   from seg.base import BaseSegmenter, SegmentationResult, register_backend

   @register_backend("your_name")
   class YourSegmenter(BaseSegmenter):
       @classmethod
       def add_cli_args(cls, parser):
           parser.add_argument("--your-flag", default="...")

       def __init__(self, classes, device="cpu", **kwargs):
           super().__init__(classes=classes, device=device)
           # load your model

       def predict(self, images: List[Image.Image]) -> List[SegmentationResult]:
           # return one SegmentationResult per image, where
           # per_class_masks maps class_name (matching classes.json keys)
           # to a (H, W) bool numpy array.
           ...
   ```

2. Add `from seg.backends import your_backend` to `seg/backends/__init__.py`.

Your backend is now available via `--backend your_name`, with its CLI flags
auto-shown in `--help`.

---

## CLI reference

```
python segment.py --help
```

Top-level flags:

| Flag | Default | Meaning |
|---|---|---|
| `-i, --input` | *(required)* | Image file or folder of images |
| `-o, --output` | `./outputs` | Output directory |
| `-c, --classes` | `./configs/classes.json` | Class mapping JSON |
| `-b, --backend` | `gdino_sam2` | One of the registered backends |
| `--device` | auto | `cuda` or `cpu` |
| `--batch-size` | `4` | Images per forward pass |
| `--alpha` | `0.5` | Overlay transparency |
| `--display` | off | Show each overlay in a window |
| `--verbose` | off | Print config |
