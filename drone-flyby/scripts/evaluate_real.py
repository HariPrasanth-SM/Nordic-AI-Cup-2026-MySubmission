#!/usr/bin/env python3
"""Evaluate untouched held-out tiles separately from balanced synthetic validation."""
import argparse,json
from pathlib import Path
from ultralytics import YOLO
import yaml

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True); p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--level',choices=['L1','L2'],required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='0'); p.add_argument('--imgsz',type=int,default=640); p.add_argument('--batch',type=int,default=16)
    a=p.parse_args()
    if a.output.exists(): raise ValueError('Use a fresh output directory')
    a.output.mkdir(parents=True)
    spec=yaml.safe_load((a.dataset/a.level/'real.yaml').read_text()); spec['path']=str((a.dataset/a.level).resolve())
    data=a.output/'real_absolute.yaml'; data.write_text(yaml.safe_dump(spec,sort_keys=False))
    model=YOLO(str(a.weights))
    metrics=model.val(data=str(data.resolve()),imgsz=a.imgsz,batch=a.batch,device=a.device,rect=True,
                      conf=.001,iou=.55,plots=True,project=str(a.output),name='metrics')
    report={'results':{k:float(v) for k,v in metrics.results_dict.items()},
            'per_class_AP50':{model.names[int(c)]:float(ap) for c,ap in zip(metrics.box.ap_class_index,metrics.box.ap50)},
            'note':'Untouched held-out source frames. Missing classes cannot be assessed; this is tile AP, not challenge full-scene AP.'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
