"""Two resident YOLO Detect models; exact quadrant+center training tile layout."""
from pathlib import Path
import numpy as np
from .detector import Detection, map_classes
from .geometry import clip_box, iou


def five_tiles(width=960,height=540):
    if (width,height)!=(960,540): raise ValueError('Expected 960x540 challenge view')
    return [(0,0,480,270),(480,0,960,270),(0,270,480,540),
            (480,270,960,540),(240,135,720,405)]


def containment(a,b):
    extent=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    return float(np.prod(extent)/max(1e-9,min(np.prod(a[2:]-a[:2]),np.prod(b[2:]-b[:2]))))


def merge_tiles(detections,threshold,containment_threshold):
    # Prefer a complete observation over a tile-edge fragment of the same object.
    ordered=sorted(detections,key=lambda d:(d.partial,-d.score))
    keep=[]
    for d in ordered:
        duplicate=any(d.cls==old.cls and (
            iou(d.box,old.box)>threshold or
            (d.partial and d.tile!=old.tile and containment(d.box,old.box)>containment_threshold)
        ) for old in keep)
        if not duplicate: keep.append(d)
    return sorted(keep,key=lambda d:d.score,reverse=True)


class LevelDetector:
    supports_levels=True
    def __init__(self,cfg,models=None):
        self.cfg=cfg
        if models is None:
            from ultralytics import YOLO
            models={}
            for level,path in ((1,cfg.weights_l1),(2,cfg.weights_l2)):
                if not Path(path).is_file(): raise FileNotFoundError(f'L{level} checkpoint missing: {path}')
                model=YOLO(path)
                if model.task!='detect': raise ValueError(f'{path}: requires fine-tuned Detect best.pt, not the original DOTA OBB checkpoint')
                models[level]=model
        self.models=models
        self.mappings={level:map_classes(model.names,cfg.aliases,cfg.strict_classes) for level,model in models.items()}
        forbidden={'source','stream','save','verbose','conf','iou','imgsz','device','max_det','rect'} & cfg.predict_args.keys()
        if forbidden: raise ValueError(f'Adapter owns prediction arguments {forbidden}')
        self.last_info={}
        for level in (1,2):
            for _ in range(cfg.warmup): self.detect(np.zeros((540,960,3),np.uint8),level)

    def detect(self,image,level):
        if level==0 and self.cfg.l0_mode=='skip':
            self.last_info={'resolution_level':0,'model':'none','skipped':True,'reason':'no_L0_trained_detector'}
            return []
        selected=1 if level==0 else level
        model=self.models[selected]; mapping=self.mappings[selected]
        h,w=image.shape[:2]; windows=five_tiles(w,h)
        crops=[image[y:b,x:r].copy() for x,y,r,b in windows]
        results=model.predict(crops,imgsz=self.cfg.imgsz,rect=True,device=self.cfg.device,
            conf=self.cfg.conf,iou=self.cfg.iou,max_det=self.cfg.max_det,
            verbose=False,save=False,stream=False,**self.cfg.predict_args)
        detections=[]
        for tile,(result,(x,y,r,b)) in enumerate(zip(results,windows)):
            if result.boxes is None: continue
            boxes=result.boxes.xyxy.cpu().numpy()
            for box,score,cls in zip(boxes,result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy().astype(int)):
                if cls not in mapping: continue
                partial=bool(box[0]<=2 or box[1]<=2 or box[2]>=r-x-2 or box[3]>=b-y-2)
                # Ultralytics has already inverted its letterbox; only tile offset is needed.
                shifted=clip_box(box+np.array([x,y,x,y]),w,h)
                if shifted is not None:
                    detections.append(Detection(shifted,mapping[cls],float(score),partial,tile))
        merged=merge_tiles(detections,self.cfg.iou,self.cfg.merge_containment)
        self.last_info={'resolution_level':level,'model':f'L{selected}','tiles':len(windows),
                       'raw_count':len(detections),'merged_count':len(merged),'skipped':False}
        return merged
