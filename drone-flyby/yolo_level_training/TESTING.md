# Validation performed

Environment: Python 3.12, Ultralytics 8.3.203, PyTorch 2.14.0+cpu.
The CUDA installation recipe in README uses the published PyTorch 2.8.0 /
torchvision 0.23.0 CUDA 12.8 pairing; GPU training was not executed here.

1. Downloaded the real official yolo11s-obb.pt checkpoint from the Ultralytics
   v8.3.0 assets release. Its training metadata names runs/DOTAv1-ms.yaml.
2. Constructed a 16-class YOLO11s Detect model and copied all 378 feature tensors
   (8,633,567 values including buffers). All feature values matched exactly.
3. Saved and reloaded the initialized standard Detect checkpoint successfully.
4. Ran smoke_test.py using official YOLO11n DOTA weights and artificial data for
   both levels. Exercised one frozen-feature epoch and one unfrozen epoch,
   AP50 checkpoint saving, validation, qualitative overlays, and negative counts.

The artificial fixture is an execution test only. Zero or low AP after this tiny
run is expected and does not measure transfer effectiveness. Full training on the
user's real dataset remains to be run on their workstation.
