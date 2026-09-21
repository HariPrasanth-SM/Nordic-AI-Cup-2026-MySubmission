# DINOv2-S/14 verifier overlay

Adds a frozen DINOv2 ViT-S/14 reference matcher AFTER the existing L1/L2 YOLO
five-tile merge and BEFORE the tracker. No SAM, no YOLO retraining, no bbox edits,
no class replacement, and no changes to camera policy. Surviving detections keep
their original confidence for AP ranking. Rejected candidates cannot create new
tracks. Existing memory can still propagate older boxes until normal expiry;
restart the API when changing modes to start with an empty track bank.

Official model: https://github.com/facebookresearch/dinov2
This is the 21M-parameter `dinov2_vits14` backbone. The wrapper uses its normalized
384-dimensional CLS embedding, not a trained DINO detection head. Similarities
are not calibrated probabilities and have no direct relationship to IoU 0.5.

## Install (from your current drone_flyby_v2 folder)

```bash
unzip /path/to/dino_verifier_patch.zip -d .
python scripts/test_dino_verifier.py
python scripts/setup_dino_verifier.py --base configs/tracking.yaml \
  --output configs/dino_verifier.yaml
```

No existing source files/configurations are overwritten. The new config copies
all current settings and changes only the detector backend to the plugin adapter.
Keep your current working PyTorch/Ultralytics environment; do not upgrade them.
The first bank build downloads official DINOv2 code/weights using torch.hub, so
internet is needed then. Subsequent loads use the hub cache. If using a local
clone of the official repository, set `DINO_REPO=/absolute/path/to/dinov2` in BOTH
bank-building and API terminals. It must contain the official hubconf.py. The
weights hash and preprocessing version must match the bank at runtime.

## Build the reference bank once

```bash
python scripts/build_dino_bank.py \
  --root src/helsinki \
  --split datasets/helsinki_balanced_v2/split.json \
  --output weights/dino_bank.npz \
  --device 0 --annotations-complete
```

This uses ONLY the source frames listed in split.json's training split:

- Real annotated object crops at both L1 and L2 sampling resolutions.
- 16 slightly transformed cutout views per class, composited onto safe training
  backgrounds, also sampled at both resolutions.
- 256 background crops which do not intersect annotated targets (with a margin).

Crop context is 10% on each side; aspect ratio is letterboxed to 224×224. Bank
and online crops use identical BGR-to-RGB conversion and normalization. All
feature extraction is frozen; no optimizer or training epochs are involved.

Inspect `weights/dino_bank.samples/`, especially `background/`. Missing original
annotations can poison the negative bank. The explicit annotations-complete flag
means you have audited those labels, not that the script can prove completeness.
Transparent assets may have been extracted from held-out frames; their provenance
cannot be inferred automatically. Use training-derived cutouts for a clean test.

Optional: add audited false-positive crops (roofs, trees, roads, shadows) collected
from TRAINING images. Save them as PNG/JPG under a separate folder, then rebuild:

```bash
python scripts/build_dino_bank.py --root src/helsinki \
  --split datasets/helsinki_balanced_v2/split.json \
  --hard-negatives data/dino_hard_negatives \
  --output weights/dino_bank_v2.npz --device 0 --annotations-complete
```

These should include the candidate box plus about 10% context, with no real target.
Do not put images used for your held-out score into this bank. Do not assume a
low-confidence detection is a true negative without inspecting it. Set DINO_BANK
to the new filename in subsequent API commands if you rebuild.

## First run: shadow mode (records decisions, preserves baseline predictions)

Stop the current API with Ctrl+C, then:

```bash
DRONE_CONFIG=configs/dino_verifier.yaml \
DINO_BANK=weights/dino_bank.npz DINO_MODE=shadow \
DRONE_TRACE_DIR=results/dino_shadow \
python api_diagnostic.py
```

Use `python api.py` if the previous diagnostic overlay is not installed. Both
entrypoints use the same detector factory. Your existing camera boundary guard
remains active when you run api_diagnostic.py.

Run one LOCAL replay with matching ground truth (full flight is an in-sample
pipeline diagnostic; the held-out scene is the more appropriate accuracy check):

```bash
python scripts/evaluate_local.py --scene helsinki_holdout \
  --output results/dino_shadow_local.json
```

If the holdout scene does not exist yet, create it with the existing script:

```bash
python scripts/make_eval_scene.py --dataset datasets/helsinki_balanced_v2 \
  --source src/helsinki --scene helsinki_holdout
```

Stop the server gracefully, substitute its actual pipeline trace directory:

```bash
python scripts/summarize_dino_verifier.py \
  --trace results/dino_shadow/RUN_DIRECTORY/trace.jsonl \
  --output results/dino_shadow_summary

python scripts/evaluate_dino_trace.py \
  --trace results/dino_shadow/RUN_DIRECTORY/trace.jsonl \
  --annotations src/helsinki/annotations \
  --output results/dino_before_after.json
```

If the trace contains multiple sessions, the AP tool requires --session SESSION_KEY.
The summary tool reports verifier p50/p95 latency and every accept/reject/bypass.
The comparison computes 101-point interpolated AP50 for the raw candidate stream
before/after gating on the SAME received views. Ground truth is full-frame, so
objects outside the camera are misses in both. This is a diagnostic AP50, not the
full official COCO scorer, not tracker AP, and not organizer acceptance scoring.
Only classes with GT are averaged. Missing frames are not added to this offline
comparison. Use the official real-time replay to assess the entire online system.
NEVER use Helsinki labels for remote frames just because frame numbers match.

## Enable filtering

If shadow results preserve useful detections and reduce false positives, restart:

```bash
DRONE_CONFIG=configs/dino_verifier.yaml \
DINO_BANK=weights/dino_bank.npz DINO_MODE=filter \
DRONE_TRACE_DIR=results/dino_filter \
python api_diagnostic.py
```

For each eligible crop, the verifier compares:

1. Maximum cosine similarity to the YOLO-predicted class's reference bank.
2. Maximum similarity to the background bank.
3. Maximum similarity to any competing class.

It keeps a candidate only when target similarity >= DINO_MIN_SIM, target minus
background >= DINO_NEG_MARGIN, and target minus competing class >= DINO_CLASS_MARGIN.
Defaults are **0.55 / 0.03 / 0.00**, respectively. These are starting values,
NOT validated thresholds for your data. Override all explicitly for an experiment:

```bash
DRONE_CONFIG=configs/dino_verifier.yaml DINO_MODE=filter \
DINO_MIN_SIM=0.55 DINO_NEG_MARGIN=0.03 DINO_CLASS_MARGIN=0.00 \
DINO_BANK=weights/dino_bank.npz python api_diagnostic.py
```

If true targets are rejected, inspect reasons before changing values. Reduce the
relevant threshold/margin to be less strict. Raising them is more strict and can
hurt recall. Nearest-reference similarity can still confuse background with the
object; it is not a guarantee of semantic recognition or zero false positives.

## Latency and conservative bypasses

- Reference features are computed once offline; only candidate crops are encoded
  online, in batches of at most 32. Both existing YOLO models remain resident.
- Default cap: DINO_MAX_CANDIDATES=48 per frame. Candidates are processed in the
  base detector's confidence order. Overflow candidates pass unchanged and are
  logged as candidate_budget. Lowering this cap saves time but verifies fewer boxes.
- Partial/tile-edge boxes and boxes under 8 pixels on either side pass unchanged:
  whole-object references may be unreliable for tiny fragments. These are logged
  as partial_box / too_small. Thus not every false positive is subject to rejection.
- Runtime encoder errors fail OPEN, preserving baseline detections and recording
  the exception. Missing or incompatible banks fail startup instead.
- DINO_MODE=off skips bank/encoder loading and returns the baseline detector output.
- p50/p95 verifier time includes embedding transfer back to CPU; compare overall
  pipeline response time and real-time frame gaps too. No 333 ms claim is made.

The existing trace stores full decisions under detector_info.verifier, including
all pre-verifier YOLO boxes, class/background/competitor similarities, rejection
reasons, bank hash and thresholds. The pipeline's normal detections field contains
post-verifier survivors in filter mode. In shadow mode it contains the unchanged
baseline. Existing viewers keep working; use decisions.jsonl from the summary
script to inspect exactly what DINO rejected. This overlay does not modify the
HTML viewer or silently erase historical tracking outputs.

## Rollback

```bash
DRONE_CONFIG=configs/tracking.yaml python api_diagnostic.py
```

Or keep the copied config and set DINO_MODE=off. Restart to clear tracker memory.

## Verification and limits

Included automated tests check preprocessing color/shape, crop handling,
class/background rejection logic, shadow vs filter behavior, conservative
partial/budget bypasses, and fail-open logging using a synthetic encoder. They
require no model download. The trace AP helper is checked on perfect/missed/false
predictions. Compilation and plugin configuration parsing were checked locally.
Real DINO weights, GPU latency, and the actual reference bank must be exercised
on your machine. No new challenge score or false-positive reduction is claimed.
