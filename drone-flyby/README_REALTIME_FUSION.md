# realtime-fusion-v1

## 1. Camera rejections ("ignored camera request ... exceeds the L1 limit")
Every legality check used `request.view` as the camera the move starts from. The rejected requests in
the logs are only explained if the organizer's real camera is sometimes one command ahead of the view
that came with the request (the frame was rendered before the previous command was applied). A
simulator with that behaviour (`tests/sim.py`, `lag=1`) reproduces the errors with the old gate
(17-50 refusals per 249 frames) and shows 0 refusals with the new one.

`drone_pipeline/camera_shadow.py` keeps its own estimate of the real camera (last command it sent,
checked against the incoming views) and only emits a command that is legal from *every* plausible
current camera (rectangle and movement disk, Dykstra projection, integer rounding). The API layer no
longer re-clamps against `request.view` (`identity_only`).

To confirm on the real server: in the trace, `camera.shadow.mode` is `lag0` (view already reflects
the last command), `lag1` (view is one command behind) or `desync`; `camera.shadow.stats` has the
counts. If it is always `lag0` and errors persist, send that trace back.

## 2. Realtime fusion (`publish.mode: fusion`)
* `ego.py`: straight-line, constant-speed flight gives one constant source-pixel shift per frame. Once locked, failed or outlying image registrations are replaced by that shift instead of "no motion".
* `fusion.py`: for each tracked object, evidence from all observations inside `publish.window_ms` (2700 ms = 8 frames, past only, since the future is not available live): noisy-OR of YOLO confidence, DINO agreement and optional DINO class similarities, times trajectory stability (constant-velocity residual) and recency. Publishes the best class, a hedged second class if it has enough evidence, and low-weight orphan detections.
* `api.py`: `predict()` runs under a deadline (`deadline_ms`, default 2600, never more than `response_timeout_ms - 700`). On overrun the last publish is moved by the ego velocity and returned without any camera command, and the shadow is told the request was abandoned.
* `publish.mode: tracker` (raw tracker) and `legacy` (old PrecisionGate) remain available for A/B comparison.

## 2b. Stricter publication (v2)
AP is rank based: a wrong box hurts only when it outranks a correct one, and a deleted low-score box can cost the
recall of a rare class (macro mAP). So precision comes from *evidence*, not from a high cut-off. On the synthetic
harness the raw cut-off alone cost ~0.03 mAP; the settings below kept mAP level while publishing ~35% fewer boxes and
far fewer high-confidence ones. Real data will differ: tune with `scripts/sweep_publish.py` (section 5).
* **Multi-prototype DINO** (`prototypes.py`, `scripts/build_prototypes.py`): K = 8 spherical-k-means prototypes per class and for background. Every detection is compared with all prototypes; per class the weakest 5% of genuine members define z = 0. A box that is not close to ANY prototype of its class (z << 0) is likely a false positive or wrong label: its score is multiplied down to `proto_floor` and the class distribution is taken from the prototype similarities.
* **Larger DINO**: `detector.dino_model` = `dinov2_vitb14` (default) or `dinov2_vitl14`; bank, prototypes and encoder are checked against each other at start-up. Bank and prototypes must be rebuilt (below). `dinov2_vits14` still works with the old bank.
* **Size prior** (`priors.py`, `scripts/build_size_prior.py`): per-class size statistics in SOURCE pixels (a helicopter is ~240 px in an L2 view and in the source, ~120 px in an L1 view), taken from the training labels. An atypical size lowers that class' evidence to at worst `size_floor` (0.6); truncated boxes are exempt.
* **Miss-in-view anomaly**: a track that was seen but is not detected although the current L1/L2 view covers >= 70% of its predicted box (and it is large enough to be detectable) gets score x 0.5 per miss (an L1 miss counts 0.6, an L2 miss 1.0). Seen at most twice and missed once at L2 (or twice at L1): removed. Established tracks: removed after 3. A new detection brings it back. Not counted when the camera looks elsewhere, at L0, or when the detector failed.
* **One-sighting objects** are published at a reduced score (DINO/prototype strength lifts it) and are not dead-reckoned until a second sighting confirms them.

### Skipped or lost frames
Frame ids are consecutive and are used as the clock (evidence ages by frame id, so a skipped frame ages it correctly; the trace has `precision.skipped_frames_before`). A frame that was never requested cannot be answered: the evaluator scores `predictions[frame]` for the frame in the request, needs `frame` to match, and sends the next frame only after our answer. So "fill frame 35 from 34 and 36" is not possible online. What is done instead: (1) frame 36's answer is dead-reckoned across the gap, (2) a request whose processing overruns the deadline is answered from the last publish moved by the ego velocity, and (3) latency is the real lever - every frame that takes longer than 333 ms costs the next frame. A bigger DINO adds latency: watch `timing_ms.detector` and lower `dino_max_candidates` if needed.

## 3. Local verification
The trace now has `sent` (exactly what went to the server), `ego`, per-detection `dino`, and `camera.shadow`.
`scripts/viewer.html` draws **SENT in red** (thick); raw detections cyan, tracks green/orange, GT pink.

    DRONE_CONFIG=configs/fusion.yaml python api.py
    python scripts/review.py --trace results/live/<run>/trace.jsonl --output results/review [--annotations <gt folder>]

## 4. Rebuild the DINO assets (once, GPU)
    python scripts/build_dino_bank.py --model dinov2_vitb14 --root src/helsinki --split datasets/helsinki_balanced_v2/split.json \
        --output weights/dino_bank_dinov2_vitb14.npz --device 0 --annotations-complete
    python scripts/build_prototypes.py --bank weights/dino_bank_dinov2_vitb14.npz --k 8     # -> weights/dino_protos_dinov2_vitb14.npz
    python scripts/build_size_prior.py --dataset datasets/helsinki_balanced_v2 --annotations src/helsinki/annotations --output weights/size_prior.json
`build_prototypes.py` prints, per class, how tightly it clusters and which other class it is most similar to: inspect this before trusting the DINO evidence.

## 5. Tune offline, no GPU
Record a run with `trace.images: true`, then replay the same views under different publisher settings (cached detections and DINO evidence):

    python scripts/sweep_publish.py --trace results/live/<run>/trace.jsonl --config configs/fusion.yaml --scene helsinki \
        --set strict='{"min_publish":0.08,"single_sight_factor":0.4}' --set nomiss='{"miss_enabled":false}' --set nosize='{"size_floor":1.0}'

## 6. Tests
    PRECISION_ENABLED=0 python -m pytest tests -q
    python -m pytest scripts/test_precision_patch.py -q      # run separately: it sets PRECISION_ENABLED=1 at import

`tests/test_strict_fusion.py` covers prototypes, size prior, the anomaly rules, verifier wiring (torch stubbed) and the sweep. No weights or GPU are needed. The synthetic harness (`tests/world.py`) uses a noisy fake detector, so it
checks plumbing (no refusals, valid output, fallback, trace), not real accuracy.
