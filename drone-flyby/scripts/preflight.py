#!/usr/bin/env python3
"""Check deployable weights, scene files, and optional public health endpoint."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from drone_pipeline.config import load_config
from dtos import OBJECT_CLASSES
from PIL import Image

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/tracking.yaml')
    p.add_argument('--scene',default='helsinki');p.add_argument('--public-url')
    p.add_argument('--skip-data',action='store_true');a=p.parse_args()
    cfg=load_config(a.config)
    from ultralytics import YOLO
    from drone_pipeline.detector import map_classes
    for level,path in ((1,cfg.detector.weights_l1),(2,cfg.detector.weights_l2)):
        if not Path(path).is_file(): raise FileNotFoundError(path)
        model=YOLO(path); assert model.task=='detect','Use trained Detect best.pt, not original OBB weights'
        map_classes(model.names,{},True); print(f'L{level}: Detect model, all 16 classes OK ({path})')
    if not a.skip_data:
        root=Path('src')/a.scene
        images=sorted((root/'images').glob('frame_*.png'))
        if not images: raise ValueError(f'No local images in {root}/images; official ZIP omits image pixels')
        for path in images:
            with Image.open(path) as im: assert im.size==(3840,2160),str(path)
            annotations=json.loads((root/'annotations'/f'{path.stem}.json').read_text())['annotations']
            for obj in annotations:
                assert obj['object_id'] in OBJECT_CLASSES
                x1,y1,x2,y2=map(float,obj['bbox']);assert 0<=x1<x2<=3840 and 0<=y1<y2<=2160
        print(f'{len(images)} full-resolution frames and annotation files OK')
    if a.public_url:
        from urllib.parse import urlsplit,urlunsplit
        import requests
        u=urlsplit(a.public_url)
        assert u.scheme in ('http','https') and u.netloc and u.path=='/predict','Pass the full http(s)://host:port/predict endpoint'
        health=urlunsplit((u.scheme,u.netloc,'/api','',''))
        response=requests.get(health,timeout=10);response.raise_for_status()
        print('Health reply:',response.json())
        print('Reachable from this machine only. Use organizer Verify to confirm their network can reach it.')
