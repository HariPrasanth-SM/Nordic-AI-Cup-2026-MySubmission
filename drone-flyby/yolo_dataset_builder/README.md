# L1/L2 YOLO dataset builder

## What this produces

Two YOLO detection datasets, with separate data.yaml files. Every saved detector
image is a native 480x270 PNG. Its label is a standard YOLO TXT file:
`class_id center_x center_y width height` normalized to that final image.
Negative images have empty TXT files. No bounding boxes are drawn on training
images; overlays live in the preview directory only.

Verified against the supplied official drone-flyby local_evaluator.py render_view,
README resolution table, and dtos.py SOURCE_REGION_SIZES:

| Stage | L1 | L2 |
|---|---|---|
| Full input source | 3840x2160 | 3840x2160 |
| Camera window | 1920x1080 | 960x540 |
| Transmitted view | 960x540, OpenCV INTER_AREA downsample | 960x540, native pixels |
| Detector input tile | 480x270 | 480x270 |

The five detector tiles are four non-overlapping quadrants plus one central
480x270 tile with origin (240,135). This central tile is our implementation of
“one overlap”; it is YOUR detector tiling policy, not a camera protocol rule.
Use the same `five_tiles()` definition at inference. The overlap catches objects
cut at the central quadrant boundaries, but does not guarantee that every object
fits wholly inside a tile.

L1 camera training windows use a 2x2 grid plus center; L2 uses a 4x4 grid plus
center. Four random legal windows are added at each level for intermediate
centers and negative variety. These sample camera positions across the full
source; they are not an exhaustive camera trajectory simulation. L2 is always cropped from the full source, never from an
already downsampled L1 view.

IMPORTANT: images/ must contain the original 3840x2160 frames. A transmitted
960x540 L0 image has lost detail; the script rejects it instead of upscaling it.

## Input

```
src/helsinki/
  images/frame_000000.png ... frame_000024.png
  annotations/frame_000000.json ... frame_000024.json
  target_objects/hangar.png ... spacecraft.png
```

JPG source frames are supported. Matching image/JSON stems are required.
Annotations use the attached challenge format: frame number and annotations with
object_id and pixel XYXY bbox. Right/bottom endpoints are treated as exclusive.
Assets must be RGBA PNGs named EXACTLY with these classes, one per class:

```
hangar helicopter jet_plane large_launcher large_tower medium_launcher
medium_plane mine_roller small_launcher small_plane small_tower ta-ta tank
condor jammer spacecraft
```

The script also accepts a target_object/ folder if target_objects/ is absent.
It does not infer class names from the earlier Roboflow export. Correct the
condor/medium_plane/hangar label mismatches before naming these assets.

## Install and test

Run these commands from your project root, after extracting the ZIP there:

```bash
python -m pip install -r yolo_dataset_builder/requirements.txt
python yolo_dataset_builder/self_test.py
```

The self-test creates temporary synthetic inputs and checks native pixel scale,
L1 downsampling, tile boundaries, alpha-derived bounding boxes, placement gaps,
preview/build execution, label validity, and exact positive/negative quotas.
It does not require your source dataset and deletes its temporary fixtures.

## First run: previews

```bash
python yolo_dataset_builder/build_dataset.py preview \
  --root src/helsinki \
  --output results/yolo_preview_v1 \
  --preview-positives 8 --preview-negatives 3 \
  --val-frames 5 --seed 42
```

Open `results/yolo_preview_v1/preview/`. It contains 8 positive and 3 negative
final tiles for EACH level and EACH split (44 annotated previews total):

* Red: original source annotation.
* Green: spawned object annotation.
* Yellow text on negatives: reminder to inspect for unannotated objects.

An individual positive tile can contain original objects, spawned objects, or
both. Source boxes can cross tile boundaries; their visible part remains labeled.
No tiny intersecting object is silently dropped and turned into background.
For spawned objects, the final box is recomputed from transformed alpha pixels.
For real objects only boxes are available, so intersection boxes are retained
conservatively even when a corner of the box may contain no object pixels.

Check original annotation accuracy, cutout masks, plausible scale, clipping,
class names, and negative tiles. Repeat with a fresh output path for another seed.
Preview mode intentionally allows more negatives than the full-build percentage.
Previews are distribution checks, not a promise that full generation will select
the identical samples: changing quotas changes RNG consumption.

## Full generation

After auditing source annotations for completeness:

```bash
python yolo_dataset_builder/build_dataset.py build \
  --root src/helsinki \
  --output datasets/helsinki_yolo_v1 \
  --train-count 1600 --val-count 240 \
  --negative-fraction 0.125 \
  --spawn-per-frame 16 \
  --scale-jitter 0.15 --rotation 15 --gap 8 \
  --val-frames 5 --seed 42 \
  --annotations-complete

python yolo_dataset_builder/build_dataset.py verify \
  --output datasets/helsinki_yolo_v1
```

Per model/level: 1,600 training images (1,400 positive, 200 negative) and 240
validation images (210 positive, 30 negative). Across both models: 3,680 images.
These are requested counts, not independent scenes. The program fails clearly
if the available UNIQUE safe negative tiles cannot satisfy the quota. Reduce
both counts while preserving the fraction, or add audited source frames; it does
not manufacture negatives by deleting annotations, erasing objects, or silently
repeating the same image. Use a fresh output directory after any failure.

Look for COMPLETE at the dataset root before training. It is only written after
all requested counts and checks pass. summary.json reports actual class counts;
classes are cycled during placement, but retained tile-level counts need not be
balanced. config.json, split.json, and manifest.jsonl preserve settings, frame
membership, source windows, tile indices, full-scene placement boxes, augmentation
scale/rotation, box provenance, and decoded pixel hashes.

## Pixel scale and augmentation

A 120x60 visible source object stays 120x60 before augmentation. Transparent
margins are removed without resizing visible pixels. The initial asset dimensions
must correspond to full-source pixels, not pixels already downsampled at L0/L1.

Scale is sampled uniformly from 0.85 to 1.15. Rotation is sampled from -15 to +15
degrees by default. Independent horizontal/vertical flips have probability 0.5;
use --no-flip to disable. Rotation expands the canvas to avoid cutting the asset.
Rotated axis-aligned boxes can grow more than 15% even though scale stays within
15%. Premultiplied-alpha warping prevents hidden background RGB from creating
edge halos. Masks still need to be clean: this cannot remove baked-in shadows or
background pixels incorrectly included in an opaque object mask.

Objects are pasted onto full-resolution frames BEFORE camera cropping/downsampling.
Thus a native 120x60 object becomes about 60x30 in an L1 transmission and remains
120x60 in L2. The subsequent five crops do not resize pixels.

Every spawned object's bounding box stays at least --gap source pixels away
from original boxes and other spawned boxes. Original-original overlaps, if any,
are preserved; original annotations are not moved. Incomplete original labels
also prevent any guarantee that placement avoids every real object.

The sampler requests 16 objects per augmented full frame, cycling through all 16
classes. Original full frames are chosen instead of augmented ones with probability
--real-fraction (default 0.25). This is a scene sampling probability, not an exact
final-image quota. Both train and val contain synthetic and real samples.

## Negatives and validation limitations

Negatives come only from unaugmented source tiles with zero intersection with
ANY original box, expanded by --negative-margin (8 source pixels by default).
They have empty label files. Counts are enforced separately for train/val AND
L1/L2. Positive and negative duplicate decoded pixels are rejected within a
level, including across train/val.

The annotations-complete flag is your explicit assertion that every target in
the source images is labeled. The program cannot verify semantic absence from
boxes alone. If your “we know some objects” means labels are incomplete, finish
labeling all real targets before using the full dataset. Inspecting only 3
negative previews cannot certify the rest of the frames.

The first 20 source frames train and last 5 validate when 25 frames are supplied.
All camera views, overlapping tiles and augmented copies stay with their parent
frame. Nevertheless, consecutive frames share backgrounds and object instances.
The same cutout assets are used in both splits; if a cutout was extracted from a
held-out frame, even its appearance crosses the split. This split is a practical
pipeline diagnostic, not an independent estimate of challenge generalization.
For trustworthy model selection, acquire another flight and reserve it entirely
for real, unaugmented validation, using training-only cutouts for augmentation.

## Training integration and aspect ratio

Use `datasets/helsinki_yolo_v1/L1/data.yaml` for the L1 detector and the L2 YAML
for the L2 detector. YAML paths are absolute and generated for the current output
location; update the path field if you later move the datasets.

Files retain their 16:9 aspect ratio. At model input, use your YOLO implementation's
aspect-preserving letterbox preprocessing, not a direct stretch to a square.
For Ultralytics-style training, rectangular batching is exposed as rect=True;
match inference preprocessing. Internal resizing still changes object pixel size
at the tensor stage, so use consistent training/inference input sizes. To isolate
this prepared dataset's geometric augmentation, disable additional random scale,
rotation, flips, mosaic and mixup in your trainer initially; otherwise it may add
much stronger transformations than the 15% configured here.

At inference, offset tile detections into the 960x540 view, undo model letterbox,
and then map through the camera source window into full-frame coordinates. Merge
duplicates from overlapping tiles. Training labels in this package are tile-local,
not the challenge's frame-global response coordinates.

Memory: full-resolution images are cached per split (~0.5 GB of raw RGB pixels
for 20 frames), plus working buffers. Final images are compressed PNGs. More
synthetic copies increase augmentation variety, not the number of independent
object appearances or flight sequences.
