"""Adapters return original received-view pixel boxes, never normalized boxes.

A custom factory(config) returns an object with detect(image)->list[Detection].
The image is BGR uint8. Class indices MUST refer to dtos.OBJECT_CLASSES.
"""
from dataclasses import dataclass
from pathlib import Path
import importlib
import numpy as np
from dtos import OBJECT_CLASSES
from .geometry import clip_box, nms

@dataclass
class Detection:
    box: np.ndarray
    cls: int
    score: float
    partial: bool = False
    tile: int = 0
    extra: dict | None = None   # verifier evidence, e.g. DINO similarities


def map_classes(names, aliases, strict=True):
    names = dict(enumerate(names)) if isinstance(names,list) else names
    mapping={}
    for index,name in names.items():
        canonical=aliases.get(str(name),str(name))
        if canonical in OBJECT_CLASSES:
            mapping[int(index)]=OBJECT_CLASSES.index(canonical)
        elif strict:
            raise ValueError(f'Checkpoint class {name!r} is not a challenge class. Fine-tune or set an explicit alias.')
    if strict and set(mapping.values()) != set(range(len(OBJECT_CLASSES))):
        missing=set(OBJECT_CLASSES)-{OBJECT_CLASSES[i] for i in mapping.values()}
        raise ValueError(f'Checkpoint is missing challenge classes: {sorted(missing)}')
    if not mapping:
        raise ValueError('No checkpoint classes map to the challenge.')
    return mapping


class UltralyticsDetector:
    def __init__(self,cfg):
        from ultralytics import YOLO
        if not Path(cfg.weights).is_file():
            raise FileNotFoundError(f'Place your fine-tuned checkpoint at {cfg.weights}, or set DRONE_WEIGHTS.')
        self.cfg=cfg
        self.model=YOLO(cfg.weights,task='detect')
        self.mapping=map_classes(self.model.names,cfg.aliases,cfg.strict_classes)
        forbidden={'source','stream','save','verbose','conf','iou','imgsz','device','max_det'} & cfg.predict_args.keys()
        if forbidden:
            raise ValueError(f'Use dedicated config fields for {forbidden}; source/stream/save/verbose are owned by the adapter.')
        for _ in range(cfg.warmup):
            self.detect(np.zeros((540,960,3),np.uint8))

    def detect(self,image):
        h,w=image.shape[:2]
        windows=[(0,0,w,h)]
        if self.cfg.tiles and (w>self.cfg.tile_size or h>self.cfg.tile_size):
            size=self.cfg.tile_size
            step=max(1,int(size*(1-self.cfg.tile_overlap)))
            xs=sorted(set(list(range(0,max(1,w-size+1),step))+[max(0,w-size)]))
            ys=sorted(set(list(range(0,max(1,h-size+1),step))+[max(0,h-size)]))
            windows += [(x,y,min(x+size,w),min(y+size,h)) for y in ys for x in xs]
        crops=[image[y:y2,x:x2] for x,y,x2,y2 in windows]
        results=self.model.predict(crops,imgsz=self.cfg.imgsz,device=self.cfg.device,
            conf=self.cfg.conf,iou=self.cfg.iou,max_det=self.cfg.max_det,
            verbose=False,save=False,stream=False,**self.cfg.predict_args)
        out=[]
        for tile,(result,(x,y,x2,y2)) in enumerate(zip(results,windows)):
            if result.boxes is None:
                continue
            boxes=result.boxes.xyxy.cpu().numpy()
            scores=result.boxes.conf.cpu().numpy()
            classes=result.boxes.cls.cpu().numpy().astype(int)
            for b,score,cls in zip(boxes,scores,classes):
                if cls not in self.mapping:
                    continue
                # Ultralytics already inverts letterboxing to original crop pixels.
                # Ignore boxes cut by internal tile edges; the full-view pass remains.
                if tile and ((x>0 and b[0]<2) or (y>0 and b[1]<2) or
                             (x2<w and b[2]>x2-x-2) or (y2<h and b[3]>y2-y-2)):
                    continue
                b=clip_box(b+np.array([x,y,x,y]),w,h)
                if b is None:
                    continue
                partial=bool(b[0]<2 or b[1]<2 or b[2]>w-2 or b[3]>h-2)
                out.append(Detection(b,self.mapping[cls],float(score),partial,tile))
        return nms(out,self.cfg.iou)


def create_detector(cfg):
    if cfg.backend=='level_yolo':
        from .level_detector import LevelDetector
        return LevelDetector(cfg)
    if cfg.backend=='ultralytics':
        return UltralyticsDetector(cfg)
    module,sep,factory=cfg.backend.partition(':')
    if not sep:
        raise ValueError('backend must be ultralytics or module:factory')
    return getattr(importlib.import_module(module),factory)(cfg)
