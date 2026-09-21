# Camera error investigation overlay

This patch adds files only. It uses your existing models, configuration, tracker,
class thresholds and navigation policy. It does not retrain or tune accuracy.
Run from your current `drone_flyby_v2` directory.

## Install and run

Stop your current drone API process with Ctrl+C in its terminal. Leave other
challenge services running. Extract this overlay into the current repository:

```bash
unzip /path/to/drone_diagnostic_patch.zip -d .
python scripts/test_diagnostic_patch.py
DRONE_CONFIG=configs/tracking.yaml \
DRONE_AUDIT_DIR=results/api_audit \
DRONE_TRACE_DIR=results/pipeline_audit \
python api_diagnostic.py
```

Use `api_diagnostic.py`, not `api.py`, for this experiment. The same `/predict`
route and default port 9053 remain. Your model paths and detection/tracking
settings come from your existing selected configuration. Trace recording must
be enabled; image recording is forced on for this diagnostic run. The existing
pipeline trace manifest records model hashes, dependency versions and config.
Only run one worker, one validation attempt at a time; don't train on the GPU
while testing. Recording adds overhead, which this experiment measures.

In another terminal:

```bash
curl --fail http://127.0.0.1:9053/diagnostics
curl --fail http://YOUR_PUBLIC_HOST:9053/diagnostics
```

Both must show `camera-egress-audit-1` and matching PID/file fingerprints. If your
public URL does not expose this new route, fix which process/tunnel serves it
before submitting another validation. The PID is diagnostic metadata, not a
command to kill any process.

Run your existing local or organizer validation against the same `/predict` URL.
Save the organizer's returned score/errors separately. Stop the API gracefully
after the attempt to flush both writers. The startup line `HTTP AUDIT:` gives
the exact folder needed below.

## What is corrected

Immediately before sending a serialized successful response, the wrapper checks:

- response request ID and frame match the current request;
- camera center agrees with the actual received source-region center;
- target level is reachable and target center respects supplied bounds;
- movement obeys the CURRENT level's limit, capped by official constants,
  including when zooming out, with 2 pixels of margin.

An illegal target is projected into the permitted movement disk intersected with
the target level's bounds. If no safe integer point exists, the command is omitted
and the current camera view is retained. Existing detection annotations are
preserved. Only a response whose request/frame identity is wrong is replaced by
an empty correctly identified response, because its detections belong to another
frame. Original and final wire response bodies are both recorded.

The included test reproduces all 15 errors from your latest message, checks their
correction, and tests the serialized middleware output and timing events.

This guard guarantees legality relative to the received request metadata, not an
unobservable future camera position on the organizer's server. A different
server-side current position could still cause rejection. The goal of the new
recording is to distinguish those cases.

## Recordings

Each run has two directories, linked by the HTTP manifest:

1. `results/api_audit/<run>/`: request PNGs, `http.jsonl`, `manifest.json`,
   and `writer_status.json` after clean shutdown.
2. `results/pipeline_audit/<run>/`: existing images, raw detections, tracker
   states/covariances, match/birth/delete events, suppressed export reasons,
   motion estimates, original pipeline responses, component timings and config.

The HTTP journal records a unique `call_id` for EVERY call, including repeated
request IDs. Events are `received`, `response_ready`, `send_completed`, and
`exception`. The response also carries `X-Drone-Audit` with that call ID.

Recorded timing:

- wall-clock UTC Unix timestamps at ASGI arrival, response readiness, and send completion;
- monotonic elapsed time to body receipt, corrected response readiness, and send completion;
- existing pipeline queue/decode/detector/motion/tracking timings;
- current concurrent call IDs, request/response hashes, camera feedback and complete
  request metadata (image stored separately), plus timeout/cadence flags.

No authorization headers are recorded. Images and request metadata can still be
sensitive; share only the intended diagnostic folder.

ASGI arrival happens AFTER upstream network/proxy buffering. ASGI send completion
means the application handed off the response, not that the remote server received
or accepted it. Organizer acceptance and actual scoring cannot be inferred from
local timing alone. A 3333 ms or longer response is still recorded and displayed;
so are requests with exceptions or missing responses. The request's own timeout
and frame interval are used for flags rather than assuming values never change.

Both recording systems have bounded queues. Inspect both `writer_status.json`
files. A process crash or full queue can leave missing records; the code logs
those conditions instead of claiming a complete trace. HTTP request images are
saved independently of the pipeline's optional image-dropping policy.

## Build the viewer

With the API stopped, substitute the exact HTTP audit directory:

```bash
python scripts/build_diagnostic_report.py \
  --audit results/api_audit/RUN_DIRECTORY \
  --output results/diagnostic_report
```

Open `results/diagnostic_report/index.html` directly in a browser. It is portable
and embeds images; it may be large for a full validation run. Use a fresh output
folder for each report. If you moved the recordings to another machine, override
the pipeline trace path:

```bash
python scripts/build_diagnostic_report.py \
  --audit /path/to/api_audit/RUN_DIRECTORY \
  --trace /path/to/pipeline_audit/RUN_DIRECTORY/trace.jsonl \
  --output results/diagnostic_report_moved
```

If several sequences were recorded, the tool lists them and asks for
`--sequence EXACT_SEQUENCE_ID`. Repeated HTTP calls can reuse a cached pipeline
record; their separate HTTP timing/response entries remain visible.

Viewer:

- Left: current raw detections in green; previous detections in dashed purple,
  projected by the estimated background motion when trusted. This projection
  does not use future frames and is only a visual reference.
- Right: exact sent annotations, with fresh boxes cyan and propagated boxes
  orange when their origin can be joined to the pipeline record. Boxes without
  matched origin also use cyan; inspect `fresh: null` in the data when uncertain.
- Below: the previous RECEIVED frame, which may not be the immediately preceding
  source frame if frames were skipped.
- Both main panels use source coordinates with the actual received crop placed
  inside a dark full-frame canvas. This lets you inspect off-camera memory boxes
  without inventing unseen pixels.
- Optional suppressed-track overlays show withheld track boxes/reasons in gray.
  Details include all tracker states and birth/match/delete events.
- Ground-truth unmatched predictions are red only when GT was supplied. Otherwise
  quality is explicitly unknown. Mark/undo false detections manually in the table;
  export those review notes as JSON. Marks hide only that recorded hypothesis
  in the viewer; they do not alter the server or retrospectively change scores.
- Bottom: exact serialized response body, even for late responses. Expanded
  details include original response, modified camera target, timing and feedback.

The visual canvas assumes the official 3840×2160 source / 960×540 display geometry.
Use the included official protocol rather than repurposing it for other sizes.

## Compare detection and tracking quality

For a LOCAL replay whose ground truth matches those source frames:

```bash
python scripts/build_diagnostic_report.py \
  --audit results/api_audit/LOCAL_RUN_DIRECTORY \
  --annotations src/helsinki/annotations \
  --output results/local_diagnostic_report
```

Never attach Helsinki ground truth to a remote/private validation recording just
because the frame numbers match. Without matching GT, objective precision/recall
and false-positive labels are unavailable; manual review is still useful.

`summary.json` reports class-matched IoU≥0.5 TP/FP/FN and precision/recall for raw
detection and actual sent tracking results. It evaluates full-frame GT on RECEIVED
calls only, so unseen objects count as misses and duplicates count per call.
These counts diagnose coverage/memory effects; they are not challenge COCO AP.
All scores are included by default. For a common operating threshold use
`--conf 0.20`; this filters both panels and diagnostic metrics, while the exact
wire response remains unfiltered.

The existing official scorer can additionally compare pipeline detections and
exports over the same captured views:

```bash
python scripts/score_trace.py \
  --trace results/pipeline_audit/LOCAL_RUN_DIRECTORY/trace.jsonl \
  --scene helsinki --detector-only
python scripts/score_trace.py \
  --trace results/pipeline_audit/LOCAL_RUN_DIRECTORY/trace.jsonl \
  --scene helsinki
```

Add `--session SESSION_KEY` if necessary. These use pipeline outputs; inspect the
HTTP viewer for any identity-mismatch fallback or differences at the final API
boundary. Use your existing real-time evaluator for latency-induced dropped-frame
loss; offline AP alone does not simulate organizer acceptance of late responses.

## Locate the failure before changing accuracy

For a rejected frame, compare the organizer message to that frame's HTTP record:

1. Different public fingerprint or no corresponding request: wrong serving process,
   tunnel target, incomplete recording, or a different run.
2. `camera_audit.action=projected_to_reachable`: the proposal was invalid; the
   exact original/final responses establish where it was repaired.
3. Final target legal relative to input, but organizer reports a different origin:
   investigate concurrent requests, delayed/cached responses, frame IDs and camera
   feedback. Do not assume either side's current state without comparing records.
4. Final command and current origin exactly match but organizer still rejects it:
   verify the applicable constraints/protocol version and preserve the complete
   request/response pair for an organizer report.
5. Scores remain poor without camera rejection: inspect detector misses vs track
   propagation, false positives, coverage and latency using the same frames.

Send the HTTP audit directory, its linked pipeline trace/manifest/writer status,
and organizer result together for analysis. This patch makes no claim of improved
accuracy or remote score; it isolates command and timing behavior.
