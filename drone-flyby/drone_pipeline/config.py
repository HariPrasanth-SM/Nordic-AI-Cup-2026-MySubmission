"""Validated configuration. Paths are relative to the drone-flyby working directory."""
import os
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

class StrictConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

class DetectorConfig(StrictConfig):
    backend: str = 'level_yolo'  # or module:factory accepting this config
    weights: str = 'weights/best.pt'
    weights_l1: str = 'weights/L1.pt'
    weights_l2: str = 'weights/L2.pt'
    l0_mode: Literal['skip', 'l1'] = 'skip'
    merge_containment: float = Field(0.8, gt=0, le=1)
    device: str = '0'
    imgsz: int = Field(640, ge=32)
    conf: float = Field(0.12, ge=0, le=1)
    iou: float = Field(0.55, gt=0, le=1)
    max_det: int = Field(300, ge=1, le=500)
    aliases: dict[str, str] = Field(default_factory=dict)
    strict_classes: bool = True
    tiles: bool = False
    tile_size: int = Field(640, ge=64)
    tile_overlap: float = Field(0.2, ge=0, lt=0.8)
    predict_args: dict = Field(default_factory=dict)
    warmup: int = Field(2, ge=0)
    # DINO verifier (drone_pipeline.dino_verifier:create). Env DINO_MODEL / DINO_BANK / DINO_PROTOS / DINO_MODE override.
    dino_model: Literal['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'] = 'dinov2_vitb14'   # 384 / 768 / 1024-d CLS
    dino_bank: str = ''      # '' = weights/dino_bank_<model>.npz (legacy weights/dino_bank.npz for vits14)
    dino_protos: str = ''    # '' = weights/dino_protos_<model>.npz when it exists; multi-prototype (K per class) scoring
    dino_mode: Literal['off', 'shadow', 'filter'] = 'shadow'
    dino_max_candidates: int = Field(48, ge=1)

class MotionConfig(StrictConfig):
    enabled: bool = True
    features: int = Field(1800, ge=100)
    keyframes: int = Field(4, ge=1, le=12)
    max_keyframe_age: int = Field(18, ge=1)
    min_inliers: int = Field(12, ge=4)
    min_ratio: float = Field(0.35, gt=0, le=1)
    min_coverage: float = Field(0.015, ge=0, le=1)
    ransac_source_px: float = Field(12, gt=0)
    max_rotation_deg: float = Field(20, gt=0)
    max_scale_change: float = Field(1.3, gt=1)
    max_translation_px_per_frame: float = Field(350, gt=0)
    failure_sigma_px: float = Field(65, gt=0)

class TrackerConfig(StrictConfig):
    enabled: bool = True
    high_conf: float = Field(0.35, ge=0, le=1)
    birth_conf: float = Field(0.45, ge=0, le=1)
    immediate_conf: float = Field(0.7, ge=0, le=1)
    confirm_hits: int = Field(2, ge=1)
    measurement_view_px: float = Field(2, gt=0)
    process_px_per_second: float = Field(12, gt=0)
    gate_mahalanobis: float = Field(16, gt=0)
    max_center_distance_sizes: float = Field(2, gt=0)
    max_coast_seconds: float = Field(2.5, gt=0)
    memory_seconds: float = Field(12, gt=0)
    max_relative_sigma: float = Field(0.22, gt=0)
    min_export_score: float = Field(0.08, ge=0, le=1)
    min_observable_view_px: float = Field(10, gt=0)
    miss_detection_probability: float = Field(0.55, ge=0, lt=1)
    duplicate_iou: float = Field(0.75, gt=0, le=1)
    max_tracks: int = Field(500, ge=1, le=1500)
    @model_validator(mode='after')
    def thresholds(self):
        if not self.high_conf <= self.birth_conf <= self.immediate_conf:
            raise ValueError('Require high_conf <= birth_conf <= immediate_conf')
        return self

class CameraConfig(StrictConfig):
    mode: Literal['active', 'full', 'hold', 'sweep_l1'] = 'active'
    overview_every: int = Field(9, ge=2)
    explore_every: int = Field(3, ge=1)
    grid_columns: int = Field(8, ge=2)
    grid_rows: int = Field(4, ge=2)
    # The organizer's camera can be one command ahead of the received view: gate against a shadow of it.
    shadow: bool = True
    shadow_trust_after: int = Field(3, ge=1)   # consecutive lag-1 observations before the view is ignored
    safety_margin_px: float = Field(.01, ge=0, lt=.25)  # >=.25 would make L2-corner -> L1 (550.73 px) impossible

class TraceConfig(StrictConfig):
    enabled: bool = True
    directory: str = 'results/tracking'
    images: bool = True
    every: int = Field(1, ge=1)
    queue_size: int = Field(64, ge=1)

class EgoConfig(StrictConfig):
    enabled: bool = True
    window: int = Field(24, ge=3)
    min_samples: int = Field(4, ge=2)
    max_sigma_px: float = Field(8., gt=0)          # robust spread (px/frame) allowed for a lock
    fallback_sigma_px: float = Field(3., gt=0)
    trust_fallback: bool = True                    # prior motion counts as registered (misses become informative)
    reject_outlier_px: float = Field(60., gt=0)    # registration this far (px/frame) from a locked prior is discarded
    max_innovation_px: float = Field(150., gt=0)
    max_speed_px_per_frame: float = Field(600., gt=0)

class PublishConfig(StrictConfig):
    """legacy = previous behaviour (PrecisionGate, env-controlled); tracker = raw tracker export; fusion = causal window fusion."""
    mode: Literal['legacy', 'fusion', 'tracker'] = 'legacy'
    window_ms: int = Field(2700, ge=333)           # PAST evidence window (frames = window_ms / frame_interval_ms)
    coast_max_frames: int = Field(15, ge=1)        # publish an unobserved object for at most this many frames
    min_publish: float = Field(.04, ge=0, le=1)   # AP is rank based: a small low-score tail costs little and keeps recall of rare classes
    max_out: int = Field(200, ge=1, le=500)
    full_weight: float = Field(.85, gt=0, le=1)    # evidence weight of a complete box
    partial_weight: float = Field(.35, gt=0, le=1)
    dino_floor: float = Field(.35, ge=0, le=1)     # score multiplier when DINO clearly disagrees
    dino_margin0: float = 0.
    dino_scale: float = Field(.08, gt=0)
    dino_missing: float = Field(1., gt=0, le=1)
    dino_class_weight: float = Field(.25, ge=0, le=1)
    dino_tau: float = Field(.05, gt=0)
    stab_sigma: float = Field(.5, gt=0)            # tolerated constant-velocity residual, in object sizes
    recency_tau: float = Field(8., gt=0)           # frames
    loc_gain: float = Field(2., ge=0)
    partial_only_factor: float = Field(.4, ge=0, le=1)
    orphan_min_conf: float = Field(.5, ge=0, le=1)  # unmatched detections need at least this YOLO confidence
    orphan_factor: float = Field(.35, ge=0, le=1)
    hedge_min: float = Field(.35, ge=0, le=1)      # second class published when its evidence is at least this
    hedge_scale: float = Field(.4, ge=0, le=1)
    nms_iou: float = Field(.5, gt=0, le=1)
    fallback_decay: float = Field(.85, gt=0, le=1)
    # -- stricter publication ----------------------------------------------------------
    single_sight_factor: float = Field(.55, gt=0, le=1)   # first sighting only (not yet confirmed by a second frame)
    confirm_hits: int = Field(2, ge=1)                     # sightings that count as confirmed
    propagate_min_evidence: float = Field(.25, ge=0, le=1)  # an unobserved (dead-reckoned) object needs this much evidence
    # -- miss-in-view anomaly: seen before, but not detected although the current view covers it --
    miss_enabled: bool = True
    miss_visible_frac: float = Field(.7, gt=0, le=1)      # share of the predicted box that must lie inside the current view
    miss_min_view_px: float = Field(18., ge=0)            # smaller (in view pixels) objects may legitimately be undetectable
    miss_l1_weight: float = Field(.6, gt=0, le=1)         # an L1 miss counts less than an L2 (native resolution) miss
    miss_cross_level_weight: float = Field(.5, ge=0, le=1)  # miss counts less when the object was only seen at a finer level
    miss_max_uncertainty: float = Field(.6, gt=0)         # skip if the predicted position is worse than this many object sizes
    miss_factor: float = Field(.5, gt=0, le=1)            # score multiplier per consecutive in-view miss
    anomaly_max_hits: int = Field(2, ge=1)                # a track with at most this many sightings is an anomaly ...
    anomaly_misses: float = Field(1., gt=0)               # ... after this many in-view misses: removed for good
    drop_misses: float = Field(3., gt=0)                  # established tracks are removed after this many
    # -- DINO multi-prototype evidence (needs weights/dino_protos_<model>.npz) -----------
    proto_z0: float = 0.                                   # z = 0 is the weakest 5% of genuine members of the class
    proto_scale: float = Field(.6, gt=0)
    proto_floor: float = Field(.3, ge=0, le=1)             # multiplier when no prototype of the class is close
    proto_bg_scale: float = Field(.08, gt=0)
    # -- size prior (weights/size_prior.json from scripts/build_size_prior.py) -----------
    size_prior: str = ''                                   # '' = disabled
    size_floor: float = Field(.6, ge=0, le=1)
    size_sigma_scale: float = Field(1.5, gt=0)
    size_aspect_weight: float = Field(.5, ge=0)

class PipelineConfig(StrictConfig):
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    ego: EgoConfig = Field(default_factory=EgoConfig)
    publish: PublishConfig = Field(default_factory=PublishConfig)
    deadline_ms: int = Field(2600, ge=100)   # api answers with a valid fallback response after this long
    motion: MotionConfig = Field(default_factory=MotionConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    trace: TraceConfig = Field(default_factory=TraceConfig)
    max_sessions: int = Field(4, ge=1)
    export_fresh: bool = True
    fresh_conf: float = Field(0.20, ge=0, le=1)
    response_cache: int = Field(512, ge=1)
    opencv_threads: int = Field(2, ge=1)


def load_config(path=None):
    path = path or os.getenv('DRONE_CONFIG', 'configs/tracking.yaml')
    cfg = PipelineConfig.model_validate(yaml.safe_load(Path(path).read_text()) or {})
    for level in ('L1','L2'):
        if os.getenv('DRONE_WEIGHTS_'+level):
            setattr(cfg.detector, 'weights_'+level.lower(), os.environ['DRONE_WEIGHTS_'+level])
    if os.getenv('DRONE_WEIGHTS'):
        cfg.detector.weights = os.environ['DRONE_WEIGHTS']
    if os.getenv('DRONE_DEVICE'):
        cfg.detector.device = os.environ['DRONE_DEVICE']
    if os.getenv('DRONE_TRACE_DIR'):
        cfg.trace.directory = os.environ['DRONE_TRACE_DIR']
    if os.getenv('DRONE_TRACE_IMAGES') is not None:
        cfg.trace.images = os.environ['DRONE_TRACE_IMAGES'].lower() in ('1','true','yes')
    return cfg
