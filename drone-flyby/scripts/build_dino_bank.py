#!/usr/bin/env python3
"""Build a training-only DINOv2 reference/background bank, without fine-tuning."""
import argparse,json,random,sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dtos import OBJECT_CLASSES
from drone_pipeline.dino_verifier import Encoder,crop,PREPROCESS
from yolo_dataset_builder.build_dataset import load_assets,transform,intersection,expanded


def build(a):
    if not a.annotations_complete:raise ValueError('Audit original annotations before using --annotations-complete; negatives must be target-free')
    if a.output.exists():raise ValueError('Choose a new bank output filename')
    rng=random.Random(a.seed);names=list(OBJECT_CLASSES);split=json.loads(a.split.read_text());stems=split['train']
    if not stems or set(stems)&set(split['val']):raise ValueError('Invalid train/validation split')
    frames=[]
    for stem in stems:
        paths=[p for p in (a.root/'images').glob(stem+'.*') if p.suffix.lower() in ('.png','.jpg','.jpeg')]
        if len(paths)!=1:raise ValueError('Need unique source image for '+stem)
        im=cv2.imread(str(paths[0]));ann=json.loads((a.root/'annotations'/(stem+'.json')).read_text())['annotations']
        if im is None or im.shape[:2]!=(2160,3840):raise ValueError('Need original full-resolution frames')
        frames.append((stem,im,ann))
    images=[];labels=[];provenance=[];counts=[0]*len(names)
    def add(im,c,meta):
        if not im.size:raise ValueError('Empty bank crop')
        images.append(im);labels.append(c);provenance.append(meta)
    for stem,im,ann in frames:
        for obj in ann:
            c=names.index(obj['object_id'])
            if counts[c]>=32:continue
            for factor in (1,2):
                view=cv2.resize(im,(3840//factor,2160//factor),interpolation=cv2.INTER_AREA) if factor==2 else im
                add(crop(view,np.array(obj['bbox'])/factor),c,{'type':'real','frame':stem,'factor':factor,'box':obj['bbox']});counts[c]+=1
    def background(w,h):
        for _ in range(3000):
            stem,im,ann=rng.choice(frames);x=rng.randrange(3840-w+1);y=rng.randrange(2160-h+1);b=[x,y,x+w,y+h]
            if any(intersection(b,expanded(o['bbox'],12)) for o in ann):continue
            return im[y:y+h,x:x+w].copy(),{'frame':stem,'source_box':b}
        raise ValueError('Cannot find safe background; inspect annotations or asset sizes')
    assets=load_assets(a.root)
    for c,asset in enumerate(assets):
        for n in range(a.asset_views):
            rgba,params=transform(asset,rng,.15,15,True);h,w=rgba.shape[:2];pad=max(4,int(max(h,w)*.15))
            if w+2*pad>3840 or h+2*pad>2160:raise ValueError('Asset too large')
            bg,meta=background(w+2*pad,h+2*pad);alpha=rgba[:,:,3:4].astype(np.float32)/255
            bg[pad:pad+h,pad:pad+w]=np.uint8(np.round(rgba[:,:,:3][:,:,::-1]*alpha+bg[pad:pad+h,pad:pad+w]*(1-alpha)))
            factor=1+n%2
            if factor==2:bg=cv2.resize(bg,(bg.shape[1]//2,bg.shape[0]//2),interpolation=cv2.INTER_AREA)
            add(crop(bg,np.array([pad,pad,pad+w,pad+h])/factor),c,dict(meta,type='asset_on_background',asset=names[c],augmentation=params,factor=factor))
    for n in range(a.negatives):
        w=rng.randint(24,260);h=rng.randint(24,180);bg,meta=background(w,h)
        # Negatives already represent full context crops.
        if n%2:bg=cv2.resize(bg,(max(1,w//2),max(1,h//2)),interpolation=cv2.INTER_AREA)
        add(bg,-1,dict(meta,type='background'))
    if a.hard_negatives:
        for p in sorted(a.hard_negatives.rglob('*')):
            if p.suffix.lower() in ('.png','.jpg','.jpeg'):
                im=cv2.imread(str(p))
                if im is None:raise ValueError('Unreadable negative '+str(p))
                add(im,-1,{'type':'audited_hard_negative','path':str(p)})
    encoder=Encoder(a.device,model=a.model);features=encoder.encode(images)
    if not np.isfinite(features).all():raise ValueError('Nonfinite bank features')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    preview=a.output.with_suffix('.samples');preview.mkdir(exist_ok=False)
    for i,(im,c) in enumerate(zip(images,labels)):
        folder=preview/('background' if c==-1 else names[c]);folder.mkdir(exist_ok=True)
        if not cv2.imwrite(str(folder/f'{i:05d}.png'),im):raise IOError('Failed writing bank preview')
    meta={'classes':names,'model':a.model,'encoder_sha256':encoder.sha,'preprocess':PREPROCESS,'train_frames':stems,'heldout_frames':split['val'],
          'seed':a.seed,'provenance':provenance,'note':'Asset source-frame provenance cannot be inferred; user must keep held-out-derived assets out of the bank.'}
    np.savez_compressed(a.output,features=features,labels=np.array(labels),metadata=json.dumps(meta))
    a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print('Saved',a.output,'references:',len(labels),'dimension:',features.shape[1]);print('Inspect samples:',preview)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('src/helsinki'))
    p.add_argument('--split',type=Path,default=Path('datasets/helsinki_balanced_v2/split.json'));p.add_argument('--output',type=Path,default=Path('weights/dino_bank.npz'))
    p.add_argument('--device',default='0');p.add_argument('--model',default='dinov2_vitb14',choices=['dinov2_vits14','dinov2_vitb14','dinov2_vitl14']);p.add_argument('--negatives',type=int,default=256);p.add_argument('--asset-views',type=int,default=16)
    p.add_argument('--hard-negatives',type=Path);p.add_argument('--seed',type=int,default=42);p.add_argument('--annotations-complete',action='store_true')
    a=p.parse_args()
    if a.output.suffix!='.npz' or min(a.negatives,a.asset_views)<1:p.error('Need .npz output and positive sample counts')
    build(a)
