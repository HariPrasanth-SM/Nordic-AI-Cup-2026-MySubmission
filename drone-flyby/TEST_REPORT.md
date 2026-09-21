# Verification and limits

Executed on v2:

- 52 runtime tests passed, including all 15 camera violations supplied in the
  request, stale permissive movement metadata, partial-box geometry expiry,
  dormant appearance gating, and the prior protocol/motion/detector tests.
- Three dataset tests passed: whole-scene transforms/native scale, end-to-end
  generation with verified quotas and separate source splits, and pixel-exact
  equivalence between final-tile source cropping and the official-style renderer.
- A small CPU training run produced independent L1/L2 Detect checkpoints,
  transfer reports and validation metrics using generated fixtures. These
  fixtures are an integration check, not an accuracy benchmark.

The real Ultralytics adapter loaded both fixture checkpoints and produced valid
L0, L1, and L2 protocol responses on CPU. No real
training images or newly trained competition weights are included. No GPU latency,
private-set improvement, or new organizer score has been measured here.

The supplied training log reports selected-checkpoint AP50 of 0.733 (L1) and
0.868 (L2) on the old synthetic validation data. Neither is a full-scene score.
The weak L1 classes and illegal camera commands motivate the changes, but do not
prove the cause of the reported 0.01 remote score without the corresponding trace.

Run all included tests:

```bash
python -m pytest -q tests yolo_dataset_builder/test_balanced.py
```

Optional training integration test (downloads a small official DOTA checkpoint
unless cached, and uses generated images):

```bash
python yolo_level_training/smoke_test.py --output results/training_smoke --device cpu
```
