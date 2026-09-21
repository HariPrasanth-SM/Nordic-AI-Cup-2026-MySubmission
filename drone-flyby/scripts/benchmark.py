#!/usr/bin/env python3
"""Measure warm five-tile detector latency, excluding motion/API/network overhead."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
from drone_pipeline.config import load_config
from drone_pipeline.level_detector import LevelDetector

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/tracking.yaml')
    p.add_argument('--image',help='Optional 960x540 received-view image');p.add_argument('--iterations',type=int,default=20)
    a=p.parse_args(); assert a.iterations>0
    image=cv2.imread(a.image) if a.image else np.zeros((540,960,3),np.uint8)
    if image is None or image.shape[:2]!=(540,960):raise ValueError('Provide a 960x540 image')
    cfg=load_config(a.config); detector=LevelDetector(cfg.detector);report={}
    for level in (1,2):
        elapsed=[]
        for _ in range(a.iterations):
            t=time.perf_counter();detector.detect(image,level);elapsed.append((time.perf_counter()-t)*1000)
        report[f'L{level}']={'p50_ms':float(np.percentile(elapsed,50)),'p95_ms':float(np.percentile(elapsed,95)),
                             'tiles_per_request':5,'image':a.image or 'blank timing fixture'}
    print(json.dumps(report,indent=2))
