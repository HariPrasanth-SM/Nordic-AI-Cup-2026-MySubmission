# Train separate L1 and L2 detectors from DOTA-pretrained YOLO11 features

Recommended first experiment: YOLO11s, five warm-up epochs with only the new
prediction head trainable, then up to 80 full fine-tuning epochs per level.
L1 and L2 start independently from the same initial features. Training is
sequential on one GPU, not simultaneous.

## Why this transfer is necessary

The official yolo11s-obb.pt checkpoint is trained on DOTA v1 aerial imagery.
It predicts oriented boxes, whereas your generated dataset has ordinary YOLO
five-column axis-aligned labels. Passing those labels directly into an OBB
trainer is not the right task.

This package builds a standard YOLO11 Detect model with your exact 16 classes,
transfers all matching backbone and neck tensors, and initializes a fresh
prediction head. No old DOTA class classifier or angle output is retained. The
regression prediction head is also new. Every feature tensor is checked for
compatible shape and exact equality after copying; initialization is saved and
reloaded to ensure the Ultralytics trainer actually uses the transferred weights.
The resulting best.pt files are ordinary detection models exposing result.boxes.
No OBB-label conversion or inference adapter is needed.

DOTA's aerial viewpoint is a plausible prior, not a guarantee of superiority over
COCO for your simulated objects. --pretraining coco offers the same feature-transfer
procedure for a controlled comparison. LoRA is not implemented: standard staged
fine-tuning is simpler here, preserves the usual export/deployment path, and lets
the features adapt to the simulated imagery after a short head warm-up.

## Installation

Extract the ZIP into your drone-flyby project root. Use a separate environment
if changing Ultralytics/PyTorch might disrupt an existing API environment.

```bash
python3 -m venv .train_venv
source .train_venv/bin/activate
python -m pip install --upgrade pip
```

For a modern NVIDIA workstation, including your RTX 5090, use CUDA-enabled
PyTorch. One concrete pairing published by PyTorch is:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r yolo_level_training/requirements.txt
```

This requires an NVIDIA driver compatible with the CUDA wheel. Keep an existing
working CUDA PyTorch installation if you already have one. Do not install CPU-only
PyTorch for your GPU training. Check the GPU and a real CUDA operation:

```bash
python - <<'PY'
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA is unavailable'
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(32, 32, device='cuda')
print('CUDA matmul:', (x @ x).sum().item())
PY
```

Ultralytics is pinned to 8.3.203 because this package touches model internals.
Do not upgrade it casually. Model checkpoints download from the official
Ultralytics GitHub assets release on first use into weights/. No Hugging Face
account or access token is needed. Weights are not bundled in this ZIP.

## Quick verification before a full run

Dataset expected: datasets/helsinki_yolo_v1/L1 and L2, each with images/train,
images/val, labels/train, labels/val, and data.yaml from the earlier generator.
All images must be final 480x270 tiles, not 960x540 transmitted views.

A one-epoch-per-phase run on your actual dataset exercises the same code path:

```bash
python yolo_level_training/train.py \
  --dataset datasets/helsinki_yolo_v1 \
  --output runs/helsinki_dota11s_smoke \
  --device 0 --batch 16 --smoke
```

This still processes the full dataset, so it is a short integration run, not
instant. For an even smaller test independent of your data:

```bash
python yolo_level_training/smoke_test.py \
  --output runs/yolo_integration_test --device cpu
```

The latter uses four artificial images per split, YOLO11n, and tiny 160-pixel
network inputs. It verifies execution, not accuracy. It downloads YOLO11n DOTA
weights if needed. Use a fresh output path for each invocation.

To only verify data and transfer weights, use --prepare-only with a fresh output
path. This does not create trained detectors; starting full training requires
another fresh output directory.

## Train both levels

```bash
python yolo_level_training/train.py \
  --dataset datasets/helsinki_yolo_v1 \
  --output runs/helsinki_dota11s \
  --device 0 --batch 16
```

Defaults are in config.yaml. CLI flags override them. Data, output and weight
paths are resolved from the working directory, not from the script directory.
If memory is insufficient, use --batch 8 or --batch 4. Use --workers 0 to diagnose
multiprocessing/data-loader issues. GPU AMP is enabled; set amp: false in the
config to diagnose numerical/AMP issues. The two trained deployment checkpoints:

```
runs/helsinki_dota11s/L1/best.pt
runs/helsinki_dota11s/L2/best.pt
```

To train just one level or change capacity:

```bash
python yolo_level_training/train.py --levels L1 --size n \
  --output runs/helsinki_dota11n_L1 --device 0 --batch 16

python yolo_level_training/train.py --levels L2 --size m \
  --output runs/helsinki_dota11m_L2 --device 0 --batch 8
```

Nano favors latency; small is the starting compromise; medium is an accuracy
experiment, not a guaranteed improvement. Measure on the actual challenge API.

For a matched COCO feature-pretraining comparison:

```bash
python yolo_level_training/train.py --pretraining coco \
  --output runs/helsinki_coco11s --device 0 --batch 16
```

The COCO comparison also replaces the entire prediction head, so it compares
feature pretraining rather than claiming to be an optimized COCO baseline.

## Training behavior

* Warm-up: all layers before the final Detect module are frozen. Their BatchNorm
  statistics are also held fixed by the pinned trainer. The new head learns for
  five epochs with AdamW and learning rate 1e-3.
* Fine-tuning: all ordinary trainable layers are unfrozen; AdamW starts a NEW
  optimizer/schedule with learning rate 3e-4. It runs for up to 80 epochs, with
  standard Ultralytics early stopping at patience 20. Fixed DFL projection weights
  remain non-trainable as designed by Ultralytics.
* Each phase separately keeps best_ap50.pt. The final L1/best.pt and L2/best.pt
  copy the best AP50 checkpoint within the full fine-tuning phase. Ultralytics'
  own weights/best.pt is also kept, selected by its standard fitness metric;
  these checkpoints need not be identical. Early stopping still uses the
  standard fitness metric, not our AP50 callback.
* Geometric online augmentation, mosaic, mixup and copy-paste are disabled because
  the dataset generator already applies your controlled geometric augmentation.
  Mild brightness/saturation/hue jitter remains enabled.
* The existing negative labels are retained; the script does not regenerate or
  rebalance the dataset. Dataset checks report negative counts and missing classes.

## Aspect ratio and network resolution

Training/validation/prediction use aspect-preserving letterboxing and rectangular
batches (rect=True). A 480x270 tile is resized proportionally to the configured
long edge (default 640), then padded to a model-compatible shape. This preserves
shape but changes tensor-space object pixel sizes; it does not recover lost
L1 detail. Use the same imgsz at inference. Rectangular batching disables normal
image shuffling in this Ultralytics version; all your tiles share one aspect ratio.
Change imgsz to 480 for a near-native input-speed experiment, using a new run.

Ultralytics maps result.boxes.xyxy back to the input tile's original coordinates.
Those coordinates are NOT yet challenge-global. Your serving pipeline still
needs tile offsets, camera crop offsets/scales, and duplicate suppression for
the overlapping center tile.

## Outputs and diagnosis

Each level has:

* warmup/ and finetune/: loss curves, results.csv, confusion matrix, PR plots,
  checkpoints, and training/validation batch images.
* best.pt: fine-tuning checkpoint selected by AP50.
* evaluation/: full validation output at confidence 0.001 for AP computation.
* metrics.json: AP50, AP50-95, precision/recall, per-class AP50 for classes present,
  and negative-image false-positive rate at preview_conf (default 0.25).
* qualitative/: sampled held-out tiles with GREEN ground truth and RED predictions.
  By default it selects up to 20 positives and 10 negatives.
* validation_predictions.jsonl: all validation detections at preview_conf.
* dataset_check.json and data.yaml: checked class counts and resolved dataset paths.

The root has transfer_report.json (donor hash, feature coverage, source classes),
effective_config.json, environment.json, and results.json for both models.
Use negative-image false positives to diagnose hallucinations; changing the
visualization threshold does not change AP evaluation, which uses low confidence.

## Inference example

```python
from ultralytics import YOLO
l1 = YOLO('runs/helsinki_dota11s/L1/best.pt')
l2 = YOLO('runs/helsinki_dota11s/L2/best.pt')
results = l1.predict('path/to/480x270_tile.png', imgsz=640, rect=True,
                     device=0, conf=0.25, iou=0.7)
for r in results:
    print(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)
```

For OpenCV arrays pass BGR images, as expected by Ultralytics. If your code holds
RGB NumPy arrays, convert them to BGR or pass PIL images. Confidence 0.25 is a
starting diagnostic threshold, not a calibrated optimum for the challenge.

## Interrupted training

An interrupted phase may be resumed using its last.pt, including AP50 tracking:

```bash
python yolo_level_training/resume.py \
  runs/helsinki_dota11s/L1/finetune/weights/last.pt
```

Only that phase resumes. It does not automatically run remaining levels or copy
its result into L1/best.pt. After it completes, the deployable AP50 checkpoint is
in that phase's weights/best_ap50.pt. To train a remaining level, use --levels L2
with a new output path. Fully completed checkpoints are not resumable as an
interrupted run; start a new fine-tuning run instead.

## Limits of the experiment

No full training on your real dataset was run here: those images remain on your
workstation. The code's small integration test is not evidence of accuracy.
The first-20/last-5 split shares one flight and the same synthetic cutouts, so high
validation AP can reflect memorized appearances. Validate on a new real flight
and the challenge service before treating it as generalization. Neither DOTA
pretraining nor LoRA can create missing independent visual diversity.

## Primary sources

* YOLO11 model variants and OBB/DOTA task: https://docs.ultralytics.com/models/yolo11/
* Official checkpoint: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-obb.pt
* OBB task and different label format: https://docs.ultralytics.com/tasks/obb/
* Training options: https://docs.ultralytics.com/modes/train/
* PyTorch CUDA wheel commands: https://pytorch.org/get-started/previous-versions/

Ultralytics code/models use its published AGPL-3.0 or Enterprise licensing terms.
