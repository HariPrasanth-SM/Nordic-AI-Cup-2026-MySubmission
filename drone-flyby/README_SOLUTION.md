# Drone Flyby — solution overview

Nordic AI Cup 2026: submission by The Winter Soldier.

**Idea:** Two resident YOLO detectors (one per zoom level) feed a frozen DINOv2 verifier, a motion-compensated
tracker and a causal multi-frame fusion step that decides what is sent for the current frame, while a stateful camera gate
guarantees that every camera command the server sees is legal and a deadline watchdog guarantees that every request is answered.

| | |
|---|---|
| Score | COCO mAP@IoU 0.50, macro-averaged over 16 classes, no server-side NMS |
| Frame cadence / budget | 333 ms per frame, 3333 ms response timeout (the evaluator sends the newest frame after each answer) |
| Endpoint | `POST /predict` on port 9053, health/identity at `GET /api` |
| Validation results so far | 0.0124 (first submission, 15 refused camera moves) -> 0.0367 (1 refused move). Add later runs to the table in section 11 |

This file is the overview. Details live in `README_REALTIME_FUSION.md` (fusion, tuning, camera gate),
`README_DINO_VERIFIER.md`, `TRACKING_SPEC.md`, `yolo_dataset_builder/README.md`, `yolo_level_training/README.md`
and `SUBMISSION.md` (relay/tunnel and organizer portal).

---

## 1. The problem in 60 seconds

* A drone flies a straight line at 600 m, 13.89 m between frames, 3840x2160 source frames at 3 fps.
* Each request carries only a **960x540 view** of the current source frame; the answer must cover the **whole frame** in normalized coordinates.
* Three zoom levels: L0 = whole frame (4x downsampled), L1 = quarter frame (2x downsampled), L2 = 1/16 of the frame (native pixels).
  Levels change only between neighbours (L0<->L1<->L2).
* The response may include a **camera command** for the next view. The move limit belongs to the *current* level: L0 2203 px, L1 1102 px, L2 551 px
  (source pixels). A bad command is ignored (detections in the same response still count).
* Frames that arrive while we are still busy are skipped and scored as empty. Speed is recall.
* Objects keep their size in source pixels (constant altitude) and the flight is a straight line at constant speed, so the background
  shifts by (almost) the same vector every frame. The solution leans on both facts.

## 2. Approach

1. **Detect at the native scale of each zoom level.** Separate YOLO11s models for L1 and L2, trained on synthetic scenes built from the 16 object cutouts
   pasted into real terrain at the exact scale each level sees. L0 has no detector (4x downsampling loses the small targets); it is only used
   as an overview/recovery view.
2. **Verify with DINOv2 (frozen, no fine-tuning).** Every candidate crop is embedded and compared with a training-only reference bank and with
   K = 8 calibrated k-means prototypes per class. The result is *evidence* that scales the score. It does not delete boxes.
3. **Track through the drone's motion.** SIFT registration between views gives the background shift; a straight-line ego prior replaces failed
   registrations. A constant-velocity tracker keeps identities across views, zoom changes and skipped frames.
4. **Fuse evidence over the last 2.7 s (8 frames), causally.** Class evidence, trajectory stability, object size, in-view misses and recency
   combine into one score per object. Only past frames exist when the answer is due.
5. **Steer the camera safely.** A policy chooses where to look (L1 for discovery, L2 for small/uncertain targets, unvisited regions,
   predicted positions); a *shadow* of the organizer's camera makes sure the command is legal even if the received view lags behind the real camera.
6. **Never miss a deadline.** The endpoint answers from the last published state if processing exceeds ~2.6 s.

The design principle throughout: **AP is rank based**. A wrong box only hurts when it is scored above a right one, and a deleted low-score box can cost
the recall of a rare class (macro mAP). So precision comes from *better scores*, not from aggressive thresholds. The gates that do remove boxes
(anomaly rule, unconfirmed one-frame objects) are explicit and configurable.

## 3. Pipeline (per request)

```
request (960x540 view, view metadata, camera constraints, feedback on last command)
   |
   v
CameraShadow.observe        what camera is the organizer REALLY at? (lag-aware, resyncs from refusal text)
   |
   v
LevelDetector               L1/L2 YOLO on 5 tiles (4 quadrants + centre), tile-merge, map to source pixels   [L0: skipped]
   |
   v
DINO verifier (shadow)      crop -> DINOv2 CLS embedding -> bank similarity + K-prototype z-scores, attached to each detection
   |
   v
Motion + EgoPrior           SIFT + RANSAC similarity transform between views; locked straight-line prior replaces failed/outlier fits
   |
   v
Tracker                     Kalman constant-velocity in source coordinates, Mahalanobis gate, colour-histogram cue, dormant memory
   |
   v
Fusion (publish.mode)       window evidence -> class, score, hedge class, orphans, anomaly removal  -> boxes for THIS frame
   |
   v
CameraPolicy + Shadow.guard choose next view; project the command into the legal region of every plausible real camera
   |
   v
response (annotations + optional requested_view)      watchdog: same answer built from last publish if > deadline
```

Every stage writes into a JSON trace record (`trace.jsonl`, optional PNGs) so any frame can be replayed and inspected.

## 4. Components

### 4.1 Detectors (`drone_pipeline/level_detector.py`, `yolo_dataset_builder/`, `yolo_level_training/`)
* Two independent YOLO11s Detect models, initialised from DOTA backbone/neck tensors, fresh 16-class heads, 5 frozen warm-up epochs then up to 100 epochs with early stopping. `best.pt` is chosen by balanced-validation AP50.
* Training data is generated, not just augmented: objects are pasted at the level's native scale (x0.5 at L1, x1.0 at L2, +-15 % scale, +-15 degrees), targeting class quotas per level (>= 1000 train instances per class), negatives from target-free regions, mild photometric noise, flips.
* At run time each 960x540 view is cut into the same five tiles used in training. Boxes are un-letterboxed, merged (a complete box beats an edge fragment), and converted through the *actually received* `source_region_xyxy` to normalized full-frame coordinates.
* Synthetic validation AP is optimistic: the 16 cutouts are the only object identities. More real examples are the biggest single lever.

### 4.2 DINOv2 verifier (`dino_verifier.py`, `prototypes.py`)
* Backbone: `dinov2_vitb14` by default (768-d CLS); `vits14` (384-d) and `vitl14` (1024-d) are selectable with `detector.dino_model`.
* **Reference bank** (`scripts/build_dino_bank.py`): asset cutouts pasted on target-free backgrounds with the training augmentations, plus background crops, training frames only.
* **Prototypes** (`scripts/build_prototypes.py`): K = 8 spherical k-means centres per class and for background. For each class, z = 0 is the 5th-percentile similarity of genuine members. A detection that is close to *no* prototype of its class is likely a false positive or a wrong label; its score is multiplied down (never to zero).
* Modes: `shadow` (default, evidence only), `filter` (also drops boxes), `off`.
* Latency grows with model size and with `dino_max_candidates` (48 crops per view by default).

### 4.3 Motion and ego prior (`motion.py`, `ego.py`)
* SIFT features and a RANSAC partial-affine fit register consecutive views (also across zoom levels).
* Because flight is straight and constant-speed, the per-frame shift converges to one vector. Once it is stable, a failed or outlier registration is replaced by that prior instead of "assume no motion" (the usual way tracks were lost).

### 4.4 Tracker (`tracker.py`)
* Kalman filter per object in source pixels, gated by Mahalanobis distance and object size; class evidence and a small colour histogram as weak appearance cues.
* Dormant identities are kept for a bounded time; misses are only penalised when the object should be observable in the current view.

### 4.5 Fusion publisher (`fusion.py`) — what is actually sent
For each tracked object, over the last `publish.window_ms` (2700 ms), measured in frame ids so skipped frames age evidence correctly:

* **Observation evidence** = YOLO confidence x (0.85 full / 0.35 partial box) x DINO-bank agreement x prototype match x size prior. DINO's class distribution supplies 25 % of the class vote.
* **Noisy-OR** over the window per class -> class evidence; best class published, a second class ("hedge") published at 0.4x if its evidence >= 0.35.
* **Score** = evidence x trajectory stability (constant-velocity residual in object sizes) x recency (exp(-age/8 frames)) x localisation confidence x single-sighting factor x miss penalty.
* **Size prior** (`priors.py`, `scripts/build_size_prior.py`): per-class size statistics in *source* pixels from the training labels; atypical size scales evidence by at least 0.6.
* **Miss-in-view anomaly:** an object seen earlier, inside the current L1/L2 view and large enough to be detectable, but not detected, gets x0.5 per miss (an L1 miss counts 0.6, L2 counts 1.0). Seen at most twice and missed -> removed; established tracks are removed after three misses; a new detection brings them back. Not counted when the camera looks elsewhere, at L0, or when the detector failed.
* **Unconfirmed one-frame objects** are published at a reduced score (strong DINO/prototype evidence lifts it) and are not dead-reckoned until a second sighting confirms them.
* **Orphans** (unmatched detections) are published only above 0.5 YOLO confidence at 0.35x.
* `publish.mode`: `fusion` (this), `tracker` (raw tracker export), `legacy` (older strict precision gate). `tracker` is the safe fallback.

### 4.6 Camera (`camera.py`, `camera_shadow.py`)
* **Policy:** L1 for discovery, L2 for small or uncertain targets, coverage of regions not yet seen (transported through the motion model), predicted target centres as extra candidates; an L0 overview every 60 frames at most. Modes: `active`, `full`, `hold`, `sweep_l1` (baseline).
* **Shadow gate:** the organizer's real camera can differ from the view we received (the view may lag one command). `CameraShadow` tracks the real camera, classifies each request as lag-0 / lag-1 / desync, and projects every command into the intersection of the legal regions (bounds x movement disk) of all plausible cameras, with a 10 px safety margin (exact limit as fallback). A refusal message that names the real camera ("from L1 (1885, 1620)") resyncs the shadow immediately; after any anomaly the gate is also legal from recent commands for 12 frames.
* Random-command stress tests: 0 refusals with the gate vs 17-50 per 249 frames with a view-only check when the view lags.

### 4.7 Real time and failure handling (`api.py`, `pipeline.py`)
* One worker; the endpoint thread waits with a deadline of `min(deadline_ms = 2600, response_timeout - 700)`. On overrun the request is marked abandoned in the shadow and answered from the last published state moved by the ego velocity, with no camera command.
* Exceptions never escape: an error yields a valid (possibly empty) response.
* Latency is recall: at ~700 ms per request only about half of the 249 frames are answered; near 333 ms nearly all are. Watch `timing_ms` in the trace after any model or `dino_max_candidates` change.
* A frame that is never requested cannot be answered (the response must carry the requested frame). Skipped frames are handled by ageing evidence and dead-reckoning, not by filling them in.

## 5. Repository layout

```
api.py, api_diagnostic.py     FastAPI server (watchdog, /predict, /api)
dtos.py, utils.py             protocol models and coordinate helpers (from the organizers)
local_evaluator.py            organizer's local replay + scorer
drone_pipeline/               camera*.py, detector*.py, level_detector.py, dino_verifier.py, prototypes.py, priors.py,
                              motion.py, ego.py, tracker.py, fusion.py, pipeline.py, config.py, trace.py
configs/                      fusion.yaml (submitted), fusion_record.yaml, tracking.yaml, sweep.yaml, detector_only.yaml, ...
scripts/                      dataset/bank/prototype/size-prior builders, evaluate_*, review.py, score_trace.py,
                              tune_fusion.py, sweep_publish.py, camera_audit.py, preflight.py, benchmark.py
yolo_dataset_builder/         synthetic scene generator          yolo_level_training/   L1/L2 training
tests/                        camera gate, ego, fusion, prototypes, watchdog, tuning (no GPU needed)
weights/                      L1.pt, L2.pt, dino_bank_*.npz, dino_protos_*.npz, size_prior.json   (not in git)
```

## 6. Run it locally

All commands from the repository root, inside the CUDA environment that has PyTorch.

```bash
# environment
python -m pip freeze > environment_before.txt
python -m pip install -r requirements-tracking.txt -r yolo_level_training/requirements.txt
# train and deploy with the same Ultralytics version (the training README pins 8.3.203)
python -c "import torch,ultralytics; print(torch.__version__, ultralytics.__version__, torch.cuda.is_available())"

# data: import the organizer's 3840x2160 frames + corrected annotations, then build the synthetic sets
python scripts/prepare_source.py --root /abs/path/to/src/helsinki --dest src/helsinki
python yolo_dataset_builder/build_balanced.py preview --root src/helsinki --output results/preview_v2 --seed 42
python yolo_dataset_builder/build_balanced.py build --root src/helsinki --output datasets/helsinki_balanced_v2 \
  --train-count 8000 --val-count 1600 --min-train-instances 1000 --min-val-instances 150 \
  --min-objects 2 --max-objects 5 --val-frames 5 --gap-frames 2 --negative-fraction 0.125 --annotations-complete --seed 42
python yolo_dataset_builder/build_balanced.py verify --output datasets/helsinki_balanced_v2

# train the two detectors, then deploy the checkpoints
python yolo_level_training/train.py --config yolo_level_training/config.yaml \
  --dataset datasets/helsinki_balanced_v2 --output runs/helsinki_v2_dota11s --device 0 --batch 16
mkdir -p weights
cp runs/helsinki_v2_dota11s/L1/best.pt weights/L1.pt && cp runs/helsinki_v2_dota11s/L2/best.pt weights/L2.pt

# DINO assets (once; downloads DINOv2 through torch.hub, or set DINO_REPO to a local clone)
python scripts/build_dino_bank.py --model dinov2_vitb14 --root src/helsinki \
  --split datasets/helsinki_balanced_v2/split.json --output weights/dino_bank_dinov2_vitb14.npz --device 0 --annotations-complete
python scripts/build_prototypes.py --bank weights/dino_bank_dinov2_vitb14.npz --k 8
python scripts/build_size_prior.py --dataset datasets/helsinki_balanced_v2 --annotations src/helsinki/annotations --output weights/size_prior.json

# checks (the two suites are run separately: scripts/test_precision_patch.py sets PRECISION_ENABLED=1 on import)
python scripts/preflight.py --config configs/fusion.yaml
PRECISION_ENABLED=0 python -m pytest -q tests
PRECISION_ENABLED=1 python -m pytest -q scripts/test_precision_patch.py
python scripts/benchmark.py --iterations 30            # detector latency only; stop training first

# serve and replay the local scene against it
DRONE_CONFIG=configs/fusion.yaml python api.py                                  # terminal 1
curl --fail http://127.0.0.1:9053/api                                          # terminal 2: revision + code hash
python scripts/evaluate_local.py --scene helsinki --output results/local_offline.json
python scripts/evaluate_local.py --scene helsinki --realtime --output results/local_realtime.json   # honours the 333 ms cadence

# inspect a run (use --session <key> if the trace holds several sessions)
python scripts/review.py --trace results/live/<run>/trace.jsonl --output results/review     # open results/review/index.html
python scripts/score_trace.py --trace results/live/<run>/trace.jsonl --scene helsinki
python scripts/camera_audit.py --trace results/live/<run>/trace.jsonl                        # why was a move refused?
```

The viewer draws raw detections (cyan), fresh tracks (green), coasting tracks (orange), what was **sent** to the server (thick red) and
ground truth (pink). Only the 25-frame `helsinki` scene has ground truth; a trace from the organizer's hidden images cannot be scored locally.

### Tune the parameters on the local scene
```bash
DRONE_CONFIG=configs/fusion_record.yaml python api.py                # low YOLO floor, images saved (GPU, once)
python local_evaluator.py --scene helsinki
python scripts/tune_fusion.py --trace results/record/<run>/trace.jsonl --config configs/fusion.yaml \
  --scene helsinki --trials 150 --jobs 4                              # offline replay, no GPU
```
Output: `results/tune/<time>/summary.md`, `tuned_config.yaml` (use with `DRONE_CONFIG=`), `trials.jsonl`. With about 25 frames the recommendation
is the median of the best trials and is checked on both halves of the frames. Treat gains as indicative. `scripts/sweep_publish.py` compares a few
hand-picked `publish.*` settings the same way.

## 7. Run it on the server

The organizer connects to the endpoint from outside, so the API must be reachable on a public address.

```bash
# on the GPU machine: keep this running, do not train on the GPU meanwhile
DRONE_CONFIG=configs/fusion.yaml \
DRONE_TRACE_DIR=results/live \
DRONE_TRACE_IMAGES=1 \
python api.py                                   # listens on 0.0.0.0:9053 (DRONE_PORT to change)
curl --fail http://127.0.0.1:9053/api           # wait for warm-up to finish first
```

**Option A — the API runs on a publicly reachable GPU VM.** Open TCP 9053 in the cloud firewall and the guest firewall.

**Option B — GPU workstation behind a relay VM** (see `SUBMISSION.md` for the details and cautions):
```bash
# on the relay VM, once (needs sudo; skip if forwarding is already allowed)
printf 'GatewayPorts clientspecified\nAllowTcpForwarding yes\n' | sudo tee /etc/ssh/sshd_config.d/90-drone-forwarding.conf
sudo sshd -t && sudo systemctl reload ssh

# on the GPU workstation, keep alive
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -R 0.0.0.0:9053:127.0.0.1:9053 VM_USER@VM_PUBLIC_IP

# on the VM
ss -ltn | grep ':9053'                          # must listen on 0.0.0.0, not only 127.0.0.1
# from another network
curl --fail http://VM_PUBLIC_IP:9053/api
```
Allow TCP 9053 in the VPC firewall. If you edit an existing rule, keep its other ports.

**Organizer portal (https://cases.nordicaicup.com):** submit the URL exactly as `http://PUBLIC_HOST:9053/predict` (the path is used as given),
run **Verify**, then **Validation** (249 frames). `python scripts/submission_info.py --public-url http://PUBLIC_HOST:9053/predict` writes the checklist.
Record the attempt id, score and the matching trace folder. Do not run validations concurrently: state lives in one process on one GPU.

Useful environment variables: `DRONE_CONFIG`, `DRONE_PORT`, `DRONE_DEVICE`, `DRONE_DEADLINE_MS`, `DRONE_WEIGHTS_L1`, `DRONE_WEIGHTS_L2`,
`DRONE_TRACE_DIR`, `DRONE_TRACE_IMAGES`, `DINO_MODEL`, `DINO_BANK`, `DINO_PROTOS`, `DINO_MODE`, `DINO_REPO`, `DINO_MAX_CANDIDATES`.
`GET /api` returns a revision string and a **code hash over `drone_pipeline/*.py`**; compare the hash locally and through the public URL to be sure
the endpoint runs the code you think it does (the revision string is informational only).

## 8. Configuration cheat-sheet (`configs/fusion.yaml`)

| Key | Default | Effect |
|---|---|---|
| `detector.conf` / `iou` | 0.12 / 0.55 | YOLO floor and merge threshold. Record with a lower floor if you plan to tune |
| `detector.dino_model` | `dinov2_vitb14` | DINO size; bank and prototypes must be rebuilt when it changes |
| `detector.dino_max_candidates` | 48 | crops embedded per view: the main DINO latency knob |
| `publish.mode` | `fusion` | `tracker` = safe fallback |
| `publish.window_ms` | 2700 | evidence window (frames = window / 333 ms) |
| `publish.min_publish` | 0.04 | smallest score sent; a low tail is cheap for AP and protects rare classes |
| `publish.single_sight_factor` | 0.55 | score multiplier for an unconfirmed one-frame object |
| `publish.orphan_min_conf` | 0.5 | YOLO confidence needed for an unmatched detection |
| `publish.miss_factor`, `anomaly_*`, `drop_misses` | 0.5, 2 hits/1.0, 3.0 | miss-in-view rules |
| `publish.proto_floor`, `dino_floor`, `size_floor` | 0.3, 0.35, 0.6 | worst-case multipliers of the three evidence sources |
| `camera.mode` | `active` | `full`, `hold`, `sweep_l1` for baselines |
| `camera.safety_margin_px` | 10 | commands stay this far inside the movement limits |
| `deadline_ms` | 2600 | watchdog; the answer is never held longer than the request budget minus 700 ms |

## 9. Troubleshooting

| Symptom | Look at |
|---|---|
| Refused camera moves in the score output | `scripts/camera_audit.py`; `camera.shadow.mode` and `stats` in the trace (`lag1` vs `lag0` vs `desync`) |
| Low score but no errors | `score_trace.py` for raw detector vs exported boxes on the same views; check `timing_ms` (skipped frames) |
| `AP50 = 0` in `sweep_publish.py` / `tune_fusion.py` | the trace was recorded on images that do not match the scene you score against |
| Start-up error about the DINO bank | rebuild bank + prototypes for the configured `dino_model`, or set `dino_model: dinov2_vits14` to use the old bank |
| `DEADLINE` in the server log | the request was answered from the last publish; reduce `dino_max_candidates` or check GPU contention |
| Session error in trace tools | validation opens a one-frame probe session first; the tools pick the longest session, or pass `--session` |

## 10. Other approaches considered

Short notes on alternatives; some are implemented as baselines in this repo, others are ideas that were not built.

* **One detector for all levels / a dedicated L0 detector.** Object appearance and scale differ strongly between L1 (2x down) and L2 (native); separate models train faster and specialise. A 4x-downsampled L0 view loses the small targets, so L0 is used only for overview. A dedicated L0 detector for large objects (hangar, launcher) is a possible addition.
* **Static coverage instead of an active camera.** Sweeping L1 tiles is simple and predictable (`camera.mode: sweep_l1`, `configs/sweep.yaml`) and is the baseline the active policy has to beat. `scripts/oracle_visible.py` measures an upper bound: what a perfect detector would find in the crops the camera actually chose.
* **Detector only, or tracker only.** `configs/detector_only.yaml` disables memory; `publish.mode: tracker` sends raw tracks. Both are ablation baselines for the fusion step.
* **DINO as a hard filter.** `dino_mode: filter` drops boxes below similarity thresholds. DINO similarities are not calibrated probabilities and only 16 object identities are available as references, so hard thresholds are brittle; DINO is used as soft evidence (and a strict legacy `PrecisionGate` remains available as `publish.mode: legacy`).
* **Nearest-neighbour vs prototypes.** A single closest reference is dominated by one crop; k-means prototypes with per-class calibration describe the modes of a class and give a comparable "how typical is this" z-score across classes. A training-free few-shot classifier on DINO features alone (no YOLO class head) is a natural extension, especially for classes with few real examples.
* **Box refinement with a segmentation model (SAM).** Mask-tight boxes could raise IoU at 0.5 for odd shapes. Not used: it adds latency per candidate and IoU 0.5 is already forgiving.
* **Filling in skipped frames from neighbours.** Not possible online: a response must carry the requested frame and the next frame is only sent after the answer. Bidirectional smoothing is useful *offline* to analyse traces, not to answer.
* **Learned camera policy (RL / imitation).** Attractive but needs a faithful simulator of the organizer's camera, including the view lag found here, and far more than 25 frames. The current policy is a hand-written utility function.
* **Higher-level modelling of the straight-line flight.** The ego prior uses it for registration; a full structure-from-motion or homography-to-ground model could predict object positions across zoom changes more precisely.

## 11. Results log

| Date | Version | Validation score | Refused camera moves | Notes |
|---|---|---|---|---|
| before the fusion rewrite | first submission | 0.0124 | 15 | view-only camera check |
| 2026-09-20 | realtime fusion (shadow gate, ego, fusion) | 0.0367 | 1 (frame 155, 1107.31 px vs 1102 px) | |
| | | | | |

Local numbers on the 25-frame `helsinki` scene are in-sample (the detectors were trained on frames from it) and are not an estimate of the
hidden-set score.

## 12. What to include when handing over the source

Code, `configs/` (including the exact config that was running), `scripts/`, `tests/`, the dataset-builder config and split, `requirements*.txt` plus a
`pip freeze`, the weights (`L1.pt`, `L2.pt`, DINO bank, prototypes, size prior) with a `SHA256SUMS` file, and the DINOv2 checkpoint or a note on how it is
fetched (`torch.hub`). Exclude `results/`, caches, backup files, credentials and relay IPs. Tag the git commit that was running when the attempt was made.
Ultralytics YOLO is AGPL-3.0 and DINOv2 is Apache-2.0; state both if licences are asked for.
