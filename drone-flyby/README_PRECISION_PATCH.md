# Precision and navigation overlay v1

For the supplied `drone_flyby_v2` repository with the previous DINOv2 verifier
patch installed. No new model download or training. This patch adds source files
and an installer that backs up the exact existing files before editing them.
It refuses unrecognized source layouts instead of guessing. Existing weights,
reference banks, training data and original configuration are preserved.

## Install and run

Run from your current project root, in your existing working virtual environment:

```bash
unzip /path/to/precision_navigation_patch.zip -d .
python scripts/install_precision_patch.py --base configs/dino_verifier.yaml
python scripts/test_precision_patch.py
```

If your current DINO configuration has another filename, supply that as `--base`.
The installer creates `configs/precision.yaml`. It requires the previous
`drone_pipeline/dino_verifier.py` and the existing reference bank.
Stop the previous API process before restarting on the same port. Use ONE worker:

```bash
DRONE_CONFIG=configs/precision.yaml \
DINO_MODE=filter DINO_BANK=weights/dino_bank.npz \
PRECISION_ENABLED=1 \
DRONE_TRACE_DIR=results/precision_trace \
DRONE_AUDIT_DIR=results/precision_http \
python api_diagnostic.py
```

Retain any `DINO_REPO`, device or checkpoint environment variables from your
working DINO deployment. If the diagnostic overlay was not installed, use
`python api.py`; the final API camera guard still runs, but HTTP audit logging
requires `api_diagnostic.py` from the earlier diagnostic patch.

In another terminal:

```bash
curl -s http://127.0.0.1:9053/api
curl -s http://127.0.0.1:9053/diagnostics
```

Check `revision` is `precision-navigation-v1`. Also check `/api` through the
PUBLIC URL actually submitted to the competition. A local check does not prove
that the public tunnel points at this process. Restarting also clears stale
tracking state and cached responses.

## What changes

**Navigation.** Current request metadata is authoritative, not the preceding
requested camera position. Movement uses the minimum of the official current-level
limit and the request's limit: L0 2203 px, L1 1102 px, L2 551 px. Only adjacent
level changes are legal. L2 -> L0 becomes an intermediate L1 request. The regular
policy must subsequently request L0; this patch never assumes its L1 request was
accepted. Bounds, movement and integer rounding are checked together. L0 reset's
movement exemption never bypasses adjacency. Inconsistent request geometry causes
a hold. The old 2 px margin is reduced to 0.01 px because the closest L1 center
from an L2 corner is 550.727 px away, which is legal under the 551 px limit.

The pipeline, final API response and diagnostic middleware enforce the new guard.
HTTP logs preserve the exact final response and corresponding request. The error
list alone cannot identify whether the former problem was stale state, an old
process, another response path or a different public endpoint. Matching final
HTTP logs with the server's request IDs is how to distinguish those cases.

**Publication.** All exports, including the old fresh-detection shortcut, pass a
single gate after tracking. Defaults:

| Evidence | Requirement |
|---|---|
| YOLO | confidence >= 0.55 |
| DINO | target cosine similarity >= 0.60 |
| Background separation | target minus background >= 0.05 |
| Class separation | target minus strongest other class >= 0.03 |
| Track class support | >= 0.80 |
| Localization uncertainty | center std / box side <= 0.20 |
| Temporal evidence | 2 strong geometrically consistent observations |
| Trajectory | forward/backprojected center error <= 0.35 box diagonals |
| Size consistency | <= 1.6x predicted width/height ratio |
| Combined score | >= 0.50 |

The combined ranking score is the minimum of YOLO confidence, nonnegative DINO
similarity, trajectory quality `exp(-2*error)`, track class support and existence.
These are heuristics, not calibrated probabilities. Thresholds are starting
points; tune on held-out scenes, not the competition test score.

Only a fresh, complete, DINO-verified observation may be published. Missing DINO
evidence, verifier errors, crop-budget bypasses, partial boxes, weak registration,
ambiguous classes and unobserved/coasting tracks do not produce annotations.
The normal tracker may keep those tracks internally for association and navigation.
A strong first observation holds the current view for another observation so it
can be confirmed; the earlier frame's response remains unchanged.

Forward prediction transports the previous measurement using registered scene
motion and residual object velocity. Backprojection transforms the current
measurement back using the inverse of that same motion. This is a causal
consistency check, NOT independent bidirectional optical flow, offline smoothing,
or evidence from future frames. It cannot guarantee a consistent false positive
will be rejected. Lost registration resets confirmation.

**Revisit.** Once confirmed, an object is due for a revisit after five SOURCE
frame steps (roughly 1.665 seconds at 333 ms cadence, longer if deliveries skip).
If it is already seen strongly at that time, the revisit is unnecessary and the
schedule advances. Otherwise, request its current predicted location, not its old
pixel location. Navigation constraints may require intermediate steps. There are
at most three focus requests, at least two source frames apart, before exploration
continues; old targets expire. Actual arrival after exactly five frames is not
guaranteed. A scene frame is never rewound and earlier predictions are not revised.

## Evaluate before another competition attempt

```bash
python local_evaluator.py --url http://127.0.0.1:9053/predict --scene helsinki
python local_evaluator.py --url http://127.0.0.1:9053/predict --scene helsinki --realtime
```

Use the scene name/path supported by your existing evaluator. Preserve both runs'
outputs. Strict suppression can LOWER AP50 by omitting real objects, especially
objects outside the current crop. Compare precision, recall, AP50, skipped frames
and latency against the previous DINO run. No score improvement or zero false
positives is promised. This patch adds no neural-network forward passes; DINO,
YOLO, registration and image logging still determine real latency. The 3333 ms
response timeout is not a free delay: exceeding the 333 ms capture cadence can
skip frames. ASGI send completion does not establish server receipt time.

The API prints timestamped trace/audit directories. Substitute those exact paths:

```bash
python scripts/summarize_precision.py \
  --trace results/precision_trace/RUN/trace.jsonl \
  --http results/precision_http/RUN/http.jsonl \
  --output results/precision_summary.json

python scripts/build_diagnostic_report.py \
  --audit results/precision_http/RUN \
  --trace results/precision_trace/RUN/trace.jsonl \
  --output results/precision_report
```

The existing HTML viewer still shows detections/tracks and final wire responses.
Detailed gate decisions are in each trace record's `precision.decisions`:
YOLO/DINO scores, trajectory errors, publication and rejection reasons. The summary
counts these decisions and reports latency percentiles, deadline overruns and
camera corrections. Counts alone are not accuracy; AP requires ground truth.
Check writer_status.json after stopping the API for dropped records/images.

## Settings and controlled comparisons

Optional environment overrides (defaults shown):

```bash
export PRECISION_YOLO=0.55
export PRECISION_DINO=0.60
export PRECISION_NEG_MARGIN=0.05
export PRECISION_CLASS_MARGIN=0.03
export PRECISION_HITS=2
export PRECISION_MAX_TRAJECTORY_ERROR=0.35
export PRECISION_MIN_SCORE=0.50
export PRECISION_REVISIT_FRAMES=5
```

Restart between experiments. More required hits can further reduce recall. In
`DINO_MODE=off`, the strict gate intentionally emits no detections. Shadow mode
still supplies evidence and the precision gate still suppresses detections.

For a detector/tracker baseline with the new camera safety gate retained, restart
with `PRECISION_ENABLED=0` and `DRONE_CONFIG=configs/dino_verifier.yaml` (your original
config). This restores that configuration's fresh-export setting. Merely disabling
the gate while using precision.yaml leaves export_fresh=false.

For full source rollback, restore the `.before_precision_v1` backups for
`api.py`, `drone_pipeline/pipeline.py`, `drone_pipeline/camera.py`, and
`api_diagnostic.py` if present. Restart using your original configuration.

## Validation scope

CPU tests cover all 15 supplied camera failures, 2,000 randomized commands checked
by the competition local evaluator, L2 corner zoom-out, tighter request constraints,
metadata/identity mismatch, missing/NaN DINO evidence, weak YOLO/class ambiguity,
trajectory jumps, registration loss, bounded revisit, and the actual pipeline's
fresh-export bypass. The original pipeline tests were also run with the precision
gate disabled. These tests use synthetic model evidence; they do not establish
GPU latency, DINO accuracy or a new competition score.
