#!/usr/bin/env python3
"""Balanced final-tile copy/paste. Source geometry first, camera sampling last."""
import argparse, collections, hashlib, json, random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from build_dataset import CLASSES, load_frames, load_assets, transform, intersection, expanded, bbox, save_sample, windows, render, five_tiles, SIZES


def scene_transform(im, rows, rng, enabled):
    """Exact H/V/180-degree transforms preserve dimensions and label geometry."""
    h,w=im.shape[:2]; rows=[dict(o,bbox=list(o['bbox'])) for o in rows]
    fx=enabled and rng.random()<.5; fy=enabled and rng.random()<.5
    if fx:
        im=im[:,::-1].copy()
        for o in rows:
            x,y,r,b=o['bbox']; o['bbox']=[w-r,y,w-x,b]
    if fy:
        im=im[::-1].copy()
        for o in rows:
            x,y,r,b=o['bbox']; o['bbox']=[x,h-b,r,h-y]
    return im,rows,{'flip_horizontal':fx,'flip_vertical':fy,'rotation_degrees':180 if fx and fy else 0}


def photometric(im,rng):
    # Applied to background AND pasted objects, to avoid a synthetic style cue.
    a=im.astype(np.float32)*rng.uniform(.85,1.15)+rng.uniform(-10,10)
    a=np.uint8(np.clip(a,0,255))
    if rng.random()<.2: a=cv2.GaussianBlur(a,(3,3),rng.uniform(.2,.7))
    if rng.random()<.3:
        ok,data=cv2.imencode('.jpg',cv2.cvtColor(a,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,rng.randint(75,98)])
        if not ok: raise RuntimeError('JPEG encoding failed')
        a=cv2.cvtColor(cv2.imdecode(data,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    return a


def crop_source(frame,base,level,rng):
    vw,vh=SIZES[level]; factor=vw//960
    vx=rng.randint(0,3840-vw); vy=rng.randint(0,2160-vh)
    ti=rng.randrange(5); tx,ty,tr,tb=five_tiles()[ti]
    x,y=vx+tx*factor,vy+ty*factor; w,h=480*factor,270*factor
    patch=base[y:y+h,x:x+w].copy(); rows=[]
    for o in frame['objects']:
        b=intersection(o['bbox'],[x,y,x+w,y+h])
        if b: rows.append(dict(o,bbox=[b[0]-x,b[1]-y,b[2]-x,b[3]-y]))
    return patch,rows,{'source_frame':frame['id'],'camera_window':[vx,vy,vx+vw,vy+vh], 'tile_index':ti,'source_tile':[x,y,x+w,y+h],'source_pixels_per_output_pixel':factor}


def paste(patch, rows, asset, cid, rng, gap, jitter=True):
    sprite,aug=transform(asset,rng,.15 if jitter else 0,15 if jitter else 0,jitter)
    h,w=sprite.shape[:2]; H,W=patch.shape[:2]
    if h>H or w>W: return False
    for _ in range(250):
        x=rng.randint(0,W-w); y=rng.randint(0,H-h); b=[x,y,x+w,y+h]
        if any(intersection(expanded(b,gap),o['bbox']) for o in rows): continue
        alpha=sprite[:,:,3:4].astype(np.float32)/255
        patch[y:y+h,x:x+w]=np.uint8(np.clip(np.round(sprite[:,:,:3]*alpha+patch[y:y+h,x:x+w]*(1-alpha)),0,255))
        mask=np.zeros((H,W),np.uint8); mask[y:y+h,x:x+w]=sprite[:,:,3]
        rows.append({'class_id':cid,'bbox':b,'origin':'spawned','mask':mask,'augmentation':aug})
        return True
    return False


def downsample(patch, rows, factor):
    im=cv2.resize(patch,(480,270),interpolation=cv2.INTER_AREA) if factor!=1 else patch
    out=[]
    for o in rows:
        b=(np.array(o['bbox'])/factor).tolist()
        if 'mask' in o:
            m=cv2.resize(o['mask'],(480,270),interpolation=cv2.INTER_AREA) if factor!=1 else o['mask']
            b=bbox(m>0)
            if b is None: continue
        out.append({k:v for k,v in dict(o,bbox=b).items() if k!='mask'})
    return im,out


def write_yaml(folder,val='images/val',name='data.yaml'):
    names='\n'.join(f'  {i}: {c}' for i,c in enumerate(CLASSES))
    (folder/name).write_text(f'path: {json.dumps(str(folder.resolve()))}\ntrain: images/train\nval: {val}\nnames:\n{names}\n')


def build(a):
    if a.output.exists(): raise ValueError('Use a new output directory')
    if a.mode=='build' and not a.annotations_complete: raise ValueError('Audit all original labels, then pass --annotations-complete')
    split=load_frames(a.root,a.val_frames)
    # Exclude adjacent frames before validation to reduce temporal leakage.
    if a.gap_frames:
        if len(split['train'])<=a.gap_frames: raise ValueError('Insufficient train frames after gap')
        excluded=split['train'][-a.gap_frames:]; split['train']=split['train'][:-a.gap_frames]
    else: excluded=[]
    assets=load_assets(a.root); rng=random.Random(a.seed); out=a.output; out.mkdir(parents=True)
    (out/'split.json').write_text(json.dumps({s:[f['id'] for f in fs] for s,fs in split.items()},indent=2))
    (out/'config.json').write_text(json.dumps(dict(vars(a),root=str(a.root),output=str(a.output),excluded_frames=[f['id'] for f in excluded]),indent=2))
    summary={}; hashes=set(); manifest=(out/'manifest.jsonl').open('w')
    def save(im,rows,meta,s,lev,index,preview):
        digest=hashlib.sha256(im.tobytes()).hexdigest()
        key=(lev,digest)
        if key in hashes: return False
        hashes.add(key)
        rec=save_sample(out,s,lev,index,im,rows,dict(meta,sha256_pixels=digest),preview)
        manifest.write(json.dumps(rec)+'\n'); return True
    try:
        for s,frames in split.items():
            cache={f['id']:np.array(Image.open(f['path']).convert('RGB')) for f in frames}
            for lev in (1,2):
                total=a.train_count if s=='train' else a.val_count
                neg=round(total*a.negative_fraction); pos=total-neg
                minimum=a.min_train_instances if s=='train' else a.min_val_instances
                if a.mode=='preview': pos,neg,minimum=a.preview_positives,a.preview_negatives,0
                counts=np.zeros(16,dtype=int); n=0; tries=0; attempts=max(2000,total*100)
                # Negatives are verified against source annotations; nothing is erased.
                while n<neg:
                    tries+=1
                    if tries>attempts: raise RuntimeError('Could not find enough unique safe negatives')
                    f=rng.choice(frames); patch,rows,meta=crop_source(f,cache[f['id']],lev,rng)
                    if rows or any(intersection(expanded(o['bbox'],a.negative_margin),meta['source_tile']) for o in f['objects']): continue
                    patch,rows,aug=scene_transform(patch,rows,rng,s=='train')
                    im,rows=downsample(patch,rows,meta['source_pixels_per_output_pixel'])
                    if s=='train': im=photometric(im,rng)
                    if save(im,rows,dict(meta,scene_augmentation=aug),s,lev,n,a.mode=='preview'): n+=1
                p=0; tries=0
                while p<pos:
                    tries+=1
                    if tries>attempts: raise RuntimeError('Cannot meet quota at native scale; inspect assets / reduce density')
                    f=rng.choice(frames); patch,rows,meta=crop_source(f,cache[f['id']],lev,rng)
                    patch,rows,aug=scene_transform(patch,rows,rng,s=='train')
                    # Quota is measured on SAVED final images, not attempted source placements.
                    # Random tie breaking prevents a fixed class sequence.
                    order=list(range(16)); rng.shuffle(order); order.sort(key=lambda c:counts[c])
                    real_only=s=='train' and rows and rng.random()<a.real_fraction and counts.min()>=minimum
                    if not real_only:
                        density=rng.randint(a.min_objects,a.max_objects)
                        inserted=0
                        for cid in order:
                            if paste(patch,rows,assets[cid],cid,rng,a.gap,jitter=True): inserted+=1
                            if inserted>=density: break
                        if inserted==0: continue
                    native=[{k:v for k,v in o.items() if k!='mask'} for o in rows]
                    im,rows=downsample(patch,rows,meta['source_pixels_per_output_pixel'])
                    if not rows: continue
                    if s=='train': im=photometric(im,rng)
                    if not save(im,rows,dict(meta,scene_augmentation=aug,native_objects=native),s,lev,n+p,a.mode=='preview'): continue
                    for o in rows: counts[o['class_id']]+=1
                    p+=1
                if counts.min()<minimum: raise RuntimeError(f'{s} L{lev}: minimum {counts.min()} < {minimum}; raise image count. Partial output is NOT trainable.')
                summary[f'{s}_L{lev}']={'positive':p,'negative':n,'instances':dict(zip(CLASSES,map(int,counts)))}
                print(s,f'L{lev}',summary[f'{s}_L{lev}'],flush=True)
                if s=='val':
                    # Separate untouched deployment-style evaluation, retaining every tile,
                    # including natural negatives and edge fragments. No balance resampling.
                    ni=0; rc=collections.Counter(); rn=0
                    for f in frames:
                        for view in windows(lev,random.Random(a.seed)):
                            for ti,tile,im,rows in render(cache[f['id']],f['objects'],lev,view):
                                meta={'source_frame':f['id'],'camera_window':view,'tile_index':ti,'untouched_real':True}
                                if save(im,rows,meta,'real_val',lev,ni,a.mode=='preview'):
                                    ni+=1; rn+=not rows; rc.update(CLASSES[o['class_id']] for o in rows)
                    summary[f'real_val_L{lev}']={'images':ni,'negative':rn,'instances':dict(rc)}
                write_yaml(out/f'L{lev}')
                write_yaml(out/f'L{lev}','images/real_val','real.yaml')
    finally: manifest.close()
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    verify(out)
    (out/'COMPLETE').write_text('Balanced generation and verification passed\n')


def verify(out):
    splits=json.loads((out/'split.json').read_text()); assert not set(splits['train'])&set(splits['val'])
    cfg=json.loads((out/'config.json').read_text()); hashes=set(); counts=collections.Counter(); classes=collections.defaultdict(collections.Counter)
    with (out/'manifest.jsonl').open() as f:
        for line in f:
            r=json.loads(line); s=r['split']; lev=r['level']; assert r['source_frame'] in splits['val' if s=='real_val' else s]
            im=np.array(Image.open(out/r['image'])); assert im.shape==(270,480,3)
            digest=hashlib.sha256(im.tobytes()).hexdigest(); assert digest==r['sha256_pixels']; assert (lev,digest) not in hashes; hashes.add((lev,digest))
            labels=(out/r['label']).read_text().splitlines(); assert len(labels)==len(r['objects'])
            for line,o in zip(labels,r['objects']):
                c,x,y,w,h=map(float,line.split()); b=o['bbox']; assert c==o['class_id']; assert w>0 and h>0
                assert x-w/2>=-1e-8 and y-h/2>=-1e-8 and x+w/2<=1+1e-8 and y+h/2<=1+1e-8
                assert np.allclose([x,y,w,h],[(b[0]+b[2])/960,(b[1]+b[3])/540,(b[2]-b[0])/480,(b[3]-b[1])/270],atol=1e-8)
                classes[s,lev][int(c)]+=1
            objects=r.get('native_objects',[])
            for i,o in enumerate(objects):
                for q in objects[:i]:
                    if o['origin']=='spawned' or q['origin']=='spawned': assert intersection(expanded(o['bbox'],cfg['gap']),q['bbox']) is None
            counts[s,lev,'negative' if not labels else 'positive']+=1
    for s in ('train','val'):
        for lev in (1,2):
            if cfg['mode']=='preview': p,n=cfg['preview_positives'],cfg['preview_negatives']; minimum=0
            else:
                total=cfg['train_count'] if s=='train' else cfg['val_count']; n=round(total*cfg['negative_fraction']); p=total-n
                minimum=cfg['min_train_instances'] if s=='train' else cfg['min_val_instances']
            assert counts[s,lev,'positive']==p and counts[s,lev,'negative']==n
            assert all(classes[s,lev][c]>=minimum for c in range(16))
    print('Verified: frame split, pixel hashes, labels, quotas and synthetic non-overlap.')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['preview','build','verify'])
    p.add_argument('--root',type=Path,default=Path('src/helsinki')); p.add_argument('--output',type=Path,required=True)
    for name,default in [('val-frames',5),('gap-frames',2),('train-count',8000),('val-count',1600),('min-train-instances',1000),('min-val-instances',150),('preview-positives',10),('preview-negatives',3),('min-objects',2),('max-objects',5),('gap',8),('negative-margin',8),('seed',42)]: p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--negative-fraction',type=float,default=.125); p.add_argument('--real-fraction',type=float,default=.15)
    p.add_argument('--annotations-complete',action='store_true'); a=p.parse_args()
    if not .10<=a.negative_fraction<=.15 or not 1<=a.min_objects<=a.max_objects or min(a.gap,a.negative_margin,a.gap_frames)<0 or not 0<=a.real_fraction<=1: p.error('Invalid generation parameters')
    if a.mode=='verify': verify(a.output)
    else: build(a)
if __name__=='__main__': main()
