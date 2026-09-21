#!/usr/bin/env python3
"""Replay identical saved crops. Ignore proposed camera commands intentionally.

--cached-detections isolates tracker changes and avoids loading a YOLO model.
Without it the selected model is rerun on the exact PNGs previously observed.
"""
import argparse
import base64
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dtos import DroneFlybyPredictRequestDto
from drone_pipeline.config import load_config
from drone_pipeline.detector import Detection
from drone_pipeline.pipeline import Pipeline

class CachedDetector:
    def __init__(self): self.row=None
    def detect(self,image):
        if any(e.startswith('detector:') for e in self.row.get('errors',[])):
            raise RuntimeError('Original recorded detector failed')
        return [Detection(np.array(d['view_box'],float),int(d['cls']),float(d['score']),
            bool(d['partial']),int(d.get('tile',0)),d.get('dino')) for d in self.row['detections']]   # d['dino'] = recorded DINO evidence

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',required=True);p.add_argument('--config',default='configs/tracking.yaml')
    p.add_argument('--cached-detections',action='store_true');p.add_argument('--detector-only',action='store_true')
    p.add_argument('--output',default='results/replay');a=p.parse_args()
    cfg=load_config(a.config);cfg.trace.directory=a.output
    if a.detector_only: cfg.tracker.enabled=False
    cached=CachedDetector() if a.cached_detections else None
    pipeline=Pipeline(cfg,detector=cached)
    root=Path(a.trace).parent
    try:
        for line in Path(a.trace).read_text().splitlines():
            row=json.loads(line)
            path=root/row.get('image','__missing__')
            if not path.is_file():
                raise FileNotFoundError(f"Frame {row['frame']} has no saved PNG; replay requires every image. Use DRONE_TRACE_IMAGES=1 and trace.every=1.")
            request=row['request'];request['sequence_id']='replay-'+row['session_key']
            request['view']['image']=base64.b64encode(path.read_bytes()).decode()
            if cached: cached.row=row
            pipeline.predict(DroneFlybyPredictRequestDto.model_validate(request))
    finally: pipeline.close()
    print('Replay written to',pipeline.trace.root)
