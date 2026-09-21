#!/usr/bin/env python3
"""Train independent axis-aligned L1/L2 YOLO11 detectors from DOTA features."""
import argparse
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT

CLASSES=['hangar','helicopter','jet_plane','large_launcher','large_tower',
         'medium_launcher','medium_plane','mine_roller','small_launcher',
         'small_plane','small_tower','ta-ta','tank','condor','jammer','spacecraft']


def dump(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=str)+'\n')


def feature_transfer(donor, target):
    """Copy all shared backbone/neck tensors, deliberately exclude prediction head."""
    source=donor.model.state_dict(); dest=target.state_dict()
    last=len(target.model)-1
    if len(donor.model.model)-1!=last: raise ValueError('Donor/target architecture mismatch')
    keys=[k for k in dest if not k.startswith(f'model.{last}.')]
    missing=[k for k in keys if k not in source or source[k].shape!=dest[k].shape]
    if missing: raise ValueError(f'Incompatible feature tensors: {missing[:10]}')
    for i in range(last):
        if type(donor.model.model[i]) is not type(target.model[i]):
            raise ValueError(f'Layer {i} differs between donor and target')
    target.load_state_dict({k:source[k].float() for k in keys},strict=False)
    for k in keys:
        if not torch.equal(target.state_dict()[k],source[k].to(target.state_dict()[k])):
            raise AssertionError(f'Transfer verification failed for {k}')
    return {'feature_tensors_copied':len(keys),
            'feature_values_copied':sum(dest[k].numel() for k in keys),
            'feature_coverage':1.0,'prediction_head_index':last,
            'prediction_head':'fresh Detect head; no donor classification/angle/regression head copied'}


def initialize(cfg, output):
    torch.manual_seed(cfg['seed'])
    size=cfg['size']; pre=cfg['pretraining']
    weights=Path(cfg.get('weights_dir','weights'))/f'yolo11{size}{"-obb" if pre=="dota" else ""}.pt'
    weights.parent.mkdir(parents=True,exist_ok=True)
    donor=YOLO(str(weights))  # official Ultralytics release, auto-downloaded on first use
    if donor.task!=('obb' if pre=='dota' else 'detect'): raise ValueError('Unexpected donor task')
    target=DetectionModel(f'yolo11{size}.yaml',nc=len(CLASSES),verbose=False)
    report=feature_transfer(donor,target)
    target.names=dict(enumerate(CLASSES))
    target.args={**DEFAULT_CFG_DICT,'task':'detect','model':f'yolo11{size}.yaml'}
    report.update(source=str(weights.resolve()),pretraining=pre,source_names=donor.names,
                  sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),ultralytics=ultralytics.__version__)
    # Save and reload so Ultralytics train() sees a checkpoint, not a fresh YAML.
    # Otherwise some versions reconstruct random weights when self.ckpt is empty.
    init=output/'initial_detect.pt'
    torch.save({'model':target.cpu(),'train_args':target.args,'epoch':-1,'optimizer':None,
                'version':ultralytics.__version__},init)
    reloaded=YOLO(str(init))
    if reloaded.task!='detect' or reloaded.model.model[-1].nc!=16:
        raise AssertionError('Initialization did not produce 16-class Detect model')
    dump(output/'transfer_report.json',report)
    del donor,target,reloaded
    gc.collect()
    return init,report['prediction_head_index']


def check_dataset(root,level,out):
    base=root/level
    supplied=yaml.safe_load((base/'data.yaml').read_text())
    names=supplied['names']
    names=[names[i] if i in names else names[str(i)] for i in range(len(names))] if isinstance(names,dict) else names
    if names!=CLASSES: raise ValueError(f'{level}: class names/order do not match the 16 challenge classes')
    summary={}
    for split in ('train','val'):
        images=sorted(p for p in (base/'images'/split).iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg'))
        if not images: raise ValueError(f'No {level}/{split} images')
        counts=Counter(); negatives=0
        for image in images:
            with Image.open(image) as im:
                if im.size!=(480,270): raise ValueError(f'{image}: expected 480x270 final tile, got {im.size}')
            label=base/'labels'/split/(image.stem+'.txt')
            rows=label.read_text().splitlines()  # missing files must not silently become negatives
            if not rows: negatives+=1
            for row in rows:
                v=list(map(float,row.split()))
                if len(v)!=5 or not np.isfinite(v).all(): raise ValueError(f'Invalid detection label: {label}')
                c,x,y,w,h=v
                if int(c)!=c or not 0<=c<16 or min(w,h)<=0 or min(x-w/2,y-h/2)<-1e-6 or max(x+w/2,y+h/2)>1+1e-6:
                    raise ValueError(f'Out of range label: {label}')
                counts[CLASSES[int(c)]]+=1
        summary[split]={'images':len(images),'negative_images':negatives,'class_instances':dict(counts)}
        missing=set(CLASSES)-set(counts)
        if missing: print(f'WARNING {level}/{split}: missing classes: {sorted(missing)}')
    # Use explicit absolute paths, also fixing stale YAML path fields after moving datasets.
    prepared=out/'data.yaml'
    prepared.write_text(yaml.safe_dump({'path':str(base.resolve()),'train':'images/train','val':'images/val','names':dict(enumerate(CLASSES))},sort_keys=False))
    dump(out/'dataset_check.json',summary)
    return prepared


class SaveBestAP50:
    """Keep a separate deployable checkpoint selected by axis-aligned AP50."""
    def __init__(self): self.best=-1.0
    def __call__(self,trainer):
        value=float(trainer.metrics.get('metrics/mAP50(B)',-1))
        if value>self.best:
            self.best=value
            model=deepcopy(trainer.ema.ema if trainer.ema else trainer.model).half().cpu()
            torch.save({'model':model,'train_args':vars(trainer.args),'epoch':-1,'optimizer':None,
                        'version':ultralytics.__version__},trainer.wdir/'best_ap50.pt')
            dump(trainer.save_dir/'best_ap50.json',{'epoch':trainer.epoch+1,'map50':value})


def train_stage(checkpoint,data,cfg,out,name,epochs,freeze,lr):
    model=YOLO(str(checkpoint)); model.add_callback('on_model_save',SaveBestAP50())
    model.train(data=str(data),project=str(out),name=name,exist_ok=False,
        epochs=epochs,patience=cfg['patience'],imgsz=cfg['imgsz'],batch=cfg['batch'],
        device=cfg['device'],workers=cfg['workers'],seed=cfg['seed'],
        optimizer='AdamW',lr0=lr,lrf=0.1,weight_decay=cfg['weight_decay'],
        warmup_epochs=min(2.0,epochs/2),warmup_bias_lr=0.0,freeze=freeze,cos_lr=True,amp=cfg['amp'],
        rect=False,cache=False,deterministic=True,plots=True,save=True,val=True,
        mosaic=0.0,mixup=0.0,copy_paste=0.0,degrees=0.0,translate=0.0,
        scale=0.0,shear=0.0,perspective=0.0,flipud=0.0,fliplr=0.0,
        hsv_h=0.005,hsv_s=0.15,hsv_v=0.15,close_mosaic=0,
        save_period=10,pretrained=True)
    stage=Path(model.trainer.save_dir)
    selected=stage/'weights/best_ap50.pt'
    if not selected.exists(): raise RuntimeError(f'No AP50 checkpoint saved in {stage}')
    del model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return selected


def evaluate(weights,data,cfg,out):
    model=YOLO(str(weights))
    metrics=model.val(data=str(data),imgsz=cfg['imgsz'],batch=cfg['batch'],
                      device=cfg['device'],workers=cfg['workers'],rect=True,
                      conf=0.001,iou=0.7,plots=True,project=str(out),name='evaluation',exist_ok=True)
    report={k:float(v) for k,v in metrics.results_dict.items()}
    report['class_ap50']={CLASSES[int(cid)]:float(ap) for cid,ap in zip(metrics.box.ap_class_index,metrics.box.ap50)}
    spec=yaml.safe_load(data.read_text()); base=Path(spec['path'])
    paths=sorted((base/'images/val').glob('*'))
    paths=[p for p in paths if p.suffix.lower() in ('.png','.jpg','.jpeg')]
    positive=[]; negative=[]
    for p in paths:
        (positive if (base/'labels/val'/f'{p.stem}.txt').read_text().strip() else negative).append(p)
    rng=random.Random(cfg['seed']); rng.shuffle(positive); rng.shuffle(negative)
    n=cfg['qualitative_count']; selected=set(negative[:n//3]+positive[:n-n//3])
    vis=out/'qualitative'; vis.mkdir(exist_ok=True)
    neg_total=neg_fp=neg_detections=0
    with (out/'validation_predictions.jsonl').open('w') as f:
        for result in model.predict(source=str(base/'images/val'),stream=True,imgsz=cfg['imgsz'],
                                    batch=cfg['batch'],device=cfg['device'],rect=True,
                                    conf=cfg['preview_conf'],iou=0.7,verbose=False):
            path=Path(result.path); rows=(base/'labels/val'/f'{path.stem}.txt').read_text().splitlines()
            pred=result.boxes.data.cpu().tolist()
            if not rows:
                neg_total+=1; neg_fp+=int(bool(pred)); neg_detections+=len(pred)
            f.write(json.dumps({'image':str(path),'negative':not rows,'xyxy_conf_class':pred})+'\n')
            if path in selected:
                image=Image.open(path).convert('RGB'); draw=ImageDraw.Draw(image); w,h=image.size
                for row in rows:
                    c,x,y,bw,bh=map(float,row.split()); b=((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h)
                    draw.rectangle(b,outline='lime',width=2); draw.text((b[0],b[1]),'GT '+CLASSES[int(c)],fill='lime')
                for x1,y1,x2,y2,conf,c in pred:
                    draw.rectangle((x1,y1,x2,y2),outline='red',width=2)
                    draw.text((x1,max(0,y1-12)),f'{CLASSES[int(c)]} {conf:.2f}',fill='red')
                image.save(vis/f'{path.stem}.png')
    report.update(negative_images=neg_total,negative_images_with_false_positives=neg_fp,
                  negative_false_positive_image_rate=neg_fp/neg_total if neg_total else None,
                  negative_detection_count=neg_detections,diagnostic_confidence=cfg['preview_conf'])
    dump(out/'metrics.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    p.add_argument('--dataset'); p.add_argument('--output'); p.add_argument('--device')
    p.add_argument('--size',choices=['n','s','m','l','x'])
    p.add_argument('--pretraining',choices=['dota','coco'])
    p.add_argument('--levels',nargs='+',choices=['L1','L2'])
    p.add_argument('--batch',type=int); p.add_argument('--imgsz',type=int)
    p.add_argument('--warmup-epochs',type=int); p.add_argument('--finetune-epochs',type=int)
    p.add_argument('--workers',type=int)
    p.add_argument('--smoke',action='store_true',help='One epoch per phase; separate output required')
    p.add_argument('--prepare-only',action='store_true',help='Check datasets and transfer weights, no training')
    a=p.parse_args(); cfg=yaml.safe_load(a.config.read_text())
    for k,v in vars(a).items():
        if k not in ('config','smoke','prepare_only') and v is not None: cfg[k]=v
    if a.smoke: cfg.update(warmup_epochs=1,finetune_epochs=1,qualitative_count=6)
    if cfg['size'] not in 'nsmlx' or cfg['pretraining'] not in ('dota','coco'): raise ValueError('Invalid model configuration')
    if min(cfg['batch'],cfg['imgsz'],cfg['finetune_epochs'])<=0 or cfg['warmup_epochs']<0: raise ValueError('Invalid counts')
    if cfg['device']!='cpu' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; install the appropriate PyTorch wheel or use --device cpu for testing')
    if cfg['device']!='cpu':
        # Fail immediately for incompatible CUDA builds rather than after downloads/training setup.
        device='cuda:'+str(cfg['device']).split(',')[0]
        x=torch.ones((32,32),device=device); (x@x).sum().item()
    root=Path(cfg['dataset']).resolve(); output=Path(cfg['output']).resolve()
    if output.exists(): raise ValueError(f'{output} exists. Use a fresh output path; resume instructions are in README.')
    if (root/'config.json').exists() and not (root/'COMPLETE').exists(): raise ValueError('Dataset generation is incomplete; fix quota/verification failure first')
    output.mkdir(parents=True); dump(output/'effective_config.json',cfg)
    dump(output/'environment.json',{'python':sys.version,'torch':torch.__version__,'ultralytics':ultralytics.__version__,
                                  'cuda':torch.version.cuda,'device':cfg['device']})
    prepared={}
    for level in cfg['levels']:
        folder=output/level; folder.mkdir()
        prepared[level]=check_dataset(root,level,folder)
    initial,last=initialize(cfg,output)
    if a.prepare_only: print('Dataset checks and DOTA/COCO feature transfer passed.'); return
    results={}
    for level,data in prepared.items():
        out=output/level
        # Each level starts from the SAME initial checkpoint, never the other level's model.
        checkpoint=initial
        if cfg['warmup_epochs']:
            checkpoint=train_stage(checkpoint,data,cfg,out,'warmup',cfg['warmup_epochs'],list(range(last)),cfg['warmup_lr'])
        best=train_stage(checkpoint,data,cfg,out,'finetune',cfg['finetune_epochs'],0,cfg['finetune_lr'])
        shutil.copy2(best,out/'best.pt')
        results[level]=evaluate(out/'best.pt',data,cfg,out)
        dump(output/'results.json',results)
    print('Finished. Deployment checkpoints:')
    for level in prepared: print(output/level/'best.pt')

if __name__=='__main__': main()
