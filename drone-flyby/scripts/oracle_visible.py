#!/usr/bin/env python3
"""OFFLINE ONLY: causal current-crop detector oracle; never imported by API.

Only boxes wholly contained in the observed crop and above --minimum-pixels
are exposed. Partial objects are excluded, so hidden full extents never leak.
Uses the real received-image registration and active camera, not GT motion.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dtos import OBJECT_CLASSES,DroneFlybyPredictRequestDto
from utils import load_frame,load_annotations,frame_numbers
from local_evaluator import Camera,render_view,build_request,score
from drone_pipeline.config import load_config
from drone_pipeline.detector import Detection
from drone_pipeline.pipeline import Pipeline

class VisibleOracle:
    def __init__(self,minimum): self.minimum=minimum; self.request=None;self.annotations=[]
    def detect(self,image):
        r=np.array(self.request.view.source_region_xyxy);scale=(r[2:]-r[:2])/np.array([960,540])
        out=[]
        for g in self.annotations:
            b=np.array(g['bbox'],float)
            if (b[:2]<r[:2]).any() or (b[2:]>r[2:]).any(): continue
            if ((b[2:]-b[:2])/scale).min()<self.minimum: continue
            out.append(Detection((b-np.tile(r[:2],2))/np.tile(scale,2),OBJECT_CLASSES.index(g['object_id']),1.))
        return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',default='helsinki')
    p.add_argument('--config',default='configs/tracking.yaml');p.add_argument('--minimum-pixels',type=float,default=4.)
    a=p.parse_args();cfg=load_config(a.config);cfg.trace.directory='results/visible_oracle'
    detector=VisibleOracle(a.minimum_pixels);pipeline=Pipeline(cfg,detector);camera=Camera();predictions={}
    try:
        for index,frame in enumerate(frame_numbers(a.scene)):
            request=DroneFlybyPredictRequestDto.model_validate(build_request(frame,index,camera,render_view(load_frame(frame,a.scene),camera),None))
            detector.request=request;detector.annotations=load_annotations(frame,a.scene)
            response=pipeline.predict(request)
            predictions[frame]=[{'object_id':d.object_id,'bbox':(np.array(d.bbox)*[3840,2160,3840,2160]).tolist(),'confidence':d.confidence} for d in response.annotations]
            if response.requested_view: camera.apply(**response.requested_view.model_dump())
        print('Causal visible-crop oracle AP50:',score(a.scene,predictions))
    finally:pipeline.close()
