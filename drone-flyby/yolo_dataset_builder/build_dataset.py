#!/usr/bin/env python3
"""Challenge-faithful L1/L2 rendering, copy-paste augmentation and YOLO tiles."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

CLASSES = ['hangar','helicopter','jet_plane','large_launcher','large_tower',
           'medium_launcher','medium_plane','mine_roller','small_launcher',
           'small_plane','small_tower','ta-ta','tank','condor','jammer','spacecraft']
SIZES = {1: (1920,1080), 2: (960,540)}


def five_tiles(w=960, h=540):
    """Four quadrants and a central half-width/half-height overlap tile."""
    return [(0,0,w//2,h//2), (w//2,0,w,h//2), (0,h//2,w//2,h),
            (w//2,h//2,w,h), (w//4,h//4,3*w//4,3*h//4)]


def intersection(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1])
    x2,y2=min(a[2],b[2]),min(a[3],b[3])
    return [x1,y1,x2,y2] if x2>x1 and y2>y1 else None


def expanded(b,g):
    return [b[0]-g,b[1]-g,b[2]+g,b[3]+g]


def bbox(mask):
    ys,xs=np.nonzero(mask)
    return None if not len(xs) else [int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]


def load_frames(root, val_frames):
    index={}
    for p in (root/'images').iterdir():
        if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}:
            if p.stem in index: raise ValueError(f'Ambiguous image stem: {p.stem}')
            index[p.stem]=p
    frames=[]
    for p in (root/'annotations').glob('*.json'):
        d=json.loads(p.read_text())
        if 'annotations' not in d or 'frame' not in d:
            raise ValueError(f'{p}: expected per-frame challenge JSON, not COCO')
        if p.stem not in index: raise ValueError(f'No image for {p}')
        with Image.open(index[p.stem]) as im:
            if im.size!=(3840,2160):
                raise ValueError(f'{index[p.stem]}: expected FULL SOURCE 3840x2160, got {im.size}. A transmitted L0 960x540 cannot recover native detail.')
        objects=[]
        for a in d['annotations']:
            c=a['object_id']
            if c not in CLASSES: raise ValueError(f'Unknown class {c} in {p}')
            b=list(map(float,a['bbox']))
            if len(b)!=4 or not np.isfinite(b).all() or not(0<=b[0]<b[2]<=3840 and 0<=b[1]<b[3]<=2160):
                raise ValueError(f'Invalid source XYXY box {b} in {p}')
            objects.append({'class_id':CLASSES.index(c),'bbox':b,'origin':'original'})
        frames.append({'id':p.stem,'frame':int(d['frame']), 'path':index[p.stem], 'objects':objects})
    frames.sort(key=lambda f:(f['frame'],f['id']))
    if len({f['frame'] for f in frames})!=len(frames): raise ValueError('Duplicate frame numbers')
    if not 0<val_frames<len(frames): raise ValueError('Need at least one train and one validation source frame')
    return {'train':frames[:-val_frames], 'val':frames[-val_frames:]}


def load_assets(root):
    folder=root/'target_objects'
    if not folder.exists(): folder=root/'target_object'
    assets=[]
    for c in CLASSES:
        p=folder/f'{c}.png'
        if not p.exists(): raise ValueError(f'Missing asset {p}; filenames must be exact class names')
        with Image.open(p) as im:
            if 'A' not in im.getbands(): raise ValueError(f'{p} has no alpha channel')
            a=np.array(im.convert('RGBA'))
        b=bbox(a[:,:,3]>0)
        if b is None: raise ValueError(f'{p}: empty alpha')
        if np.all(a[:,:,3]==255): raise ValueError(f'{p}: background is not transparent')
        # Remove transparent margins WITHOUT resampling any object pixels.
        assets.append(a[b[1]:b[3],b[0]:b[2]].copy())
    return assets


def transform(a,rng,scale_jitter,rotation,flips):
    if flips and rng.random()<0.5: a=np.fliplr(a)
    if flips and rng.random()<0.5: a=np.flipud(a)
    scale=rng.uniform(1-scale_jitter,1+scale_jitter)
    angle=rng.uniform(-rotation,rotation)
    h,w=a.shape[:2]
    # Premultiplied alpha avoids blending background RGB into rotated edges.
    prem=a.astype(np.float32)/255
    prem[:,:,:3]*=prem[:,:,3:4]
    matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,scale)
    co,si=abs(matrix[0,0]),abs(matrix[0,1])
    nw,nh=int(np.ceil(w*co+h*si))+2,int(np.ceil(h*co+w*si))+2
    matrix[:,2]+=[(nw-w)/2,(nh-h)/2]
    warped=cv2.warpAffine(prem,matrix,(nw,nh),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    alpha=warped[:,:,3:4]
    rgb=np.divide(warped[:,:,:3],alpha,out=np.zeros_like(warped[:,:,:3]),where=alpha>0)
    out=np.uint8(np.clip(np.concatenate([rgb,alpha],axis=2)*255,0,255).round())
    b=bbox(out[:,:,3]>0)
    if b is None: raise ValueError('Augmentation produced empty object')
    return out[b[1]:b[3],b[0]:b[2]], {'scale':scale,'angle':angle}


def spawn(base,original,assets,rng,args,counter):
    scene=base.copy(); objects=[dict(o) for o in original]
    for _ in range(args.spawn_per_frame):
        cid=counter[0]%len(CLASSES); counter[0]+=1
        sprite,params=transform(assets[cid],rng,args.scale_jitter,args.rotation,not args.no_flip)
        h,w=sprite.shape[:2]
        if w>3840 or h>2160: raise ValueError(f'{CLASSES[cid]} asset is too large')
        for attempt in range(300):
            x=rng.randint(0,3840-w); y=rng.randint(0,2160-h)
            b=[x,y,x+w,y+h]
            if any(intersection(expanded(b,args.gap),o['bbox']) for o in objects): continue
            alpha=sprite[:,:,3:4].astype(np.float32)/255
            scene[y:y+h,x:x+w]=np.uint8(np.round(sprite[:,:,:3]*alpha+scene[y:y+h,x:x+w]*(1-alpha)))
            objects.append({'class_id':cid,'bbox':b,'origin':'spawned','alpha':sprite[:,:,3], 'augmentation':params})
            break
        else: raise RuntimeError('Could not place a non-overlapping object after 300 tries; reduce --spawn-per-frame or --gap')
    return scene,objects


def windows(level,rng):
    w,h=SIZES[level]
    out=[(x,y,x+w,y+h) for y in range(0,2160,h) for x in range(0,3840,w)]
    out.append(((3840-w)//2,(2160-h)//2,(3840+w)//2,(2160+h)//2))
    for _ in range(4):
        x=rng.randint(0,3840-w); y=rng.randint(0,2160-h)
        out.append((x,y,x+w,y+h))
    out=list(dict.fromkeys(out))
    rng.shuffle(out)
    return out


def render(scene,objects,level,view):
    vx,vy,x2,y2=view; vw,vh=SIZES[level]; s=960/vw
    image=scene[vy:y2,vx:x2]
    if level==1: image=cv2.resize(image,(960,540),interpolation=cv2.INTER_AREA)
    projected=[]
    for obj in objects:
        b=intersection(obj['bbox'],view)
        if b is None: continue
        o={k:v for k,v in obj.items() if k!='alpha'}
        o['bbox']=[(b[0]-vx)*s,(b[1]-vy)*s,(b[2]-vx)*s,(b[3]-vy)*s]
        if obj['origin']=='spawned':
            mask=np.zeros((vh,vw),np.uint8)
            bx,by,_,_=obj['bbox']; ix,iy,ir,ib=map(int,b)
            mask[iy-vy:ib-vy,ix-vx:ir-vx]=obj['alpha'][iy-by:ib-by,ix-bx:ir-bx]
            if level==1: mask=cv2.resize(mask,(960,540),interpolation=cv2.INTER_AREA)
            o['view_mask']=mask
        projected.append(o)
    for ti,tile in enumerate(five_tiles()):
        tx,ty,tr,tb=tile; rows=[]
        for o in projected:
            if 'view_mask' in o:
                local=bbox(o['view_mask'][ty:tb,tx:tr]>0)
            else:
                b=intersection(o['bbox'],tile)
                local=None if b is None else [b[0]-tx,b[1]-ty,b[2]-tx,b[3]-ty]
            if local is None: continue
            rows.append({'class_id':o['class_id'],'bbox':local,'origin':o['origin']})
        yield ti,tile,image[ty:tb,tx:tr].copy(),rows


def save_sample(out,split,level,index,image,rows,meta,preview):
    stem=f'{split}_L{level}_{index:06d}'
    image_path=out/f'L{level}/images/{split}/{stem}.png'
    label_path=out/f'L{level}/labels/{split}/{stem}.txt'
    image_path.parent.mkdir(parents=True,exist_ok=True); label_path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(image).save(image_path)
    lines=[]
    for o in rows:
        x1,y1,x2,y2=o['bbox']
        lines.append(f'{o["class_id"]} {(x1+x2)/960:.9f} {(y1+y2)/540:.9f} {(x2-x1)/480:.9f} {(y2-y1)/270:.9f}')
    label_path.write_text('\n'.join(lines)+ ('\n' if lines else ''))
    if preview:
        p=out/f'preview/L{level}/{split}/{stem}_{"positive" if rows else "negative"}.png'; p.parent.mkdir(parents=True,exist_ok=True)
        im=Image.fromarray(image); draw=ImageDraw.Draw(im)
        for o in rows:
            color='red' if o['origin']=='original' else 'lime'
            x1,y1,x2,y2=o['bbox']; draw.rectangle((x1,y1,x2-1,y2-1),outline=color,width=2)
            draw.text((x1,max(0,y1-11)),CLASSES[o['class_id']],fill=color,stroke_width=1,stroke_fill='black')
        if not rows: draw.text((5,5),'NEGATIVE: inspect for unlabeled targets',fill='yellow',stroke_width=1,stroke_fill='black')
        im.save(p)
    return dict(meta,image=str(image_path.relative_to(out)),label=str(label_path.relative_to(out)),split=split,level=level,negative=not rows,objects=rows)


def generate(args):
    if args.mode=='build' and not args.annotations_complete:
        raise ValueError('Full generation requires --annotations-complete after auditing source annotations. Missing target annotations make false negatives and unsafe placement.')
    if args.output.exists(): raise ValueError('Output already exists; use a fresh output folder to avoid stale labels')
    splits=load_frames(args.root,args.val_frames); assets=load_assets(args.root)
    args.output.mkdir(parents=True)
    config=vars(args).copy(); config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    (args.output/'split.json').write_text(json.dumps({s:[f['id'] for f in fs] for s,fs in splits.items()},indent=2))
    rng=random.Random(args.seed); seen=set(); counter=[0]; summary={}
    with (args.output/'manifest.jsonl').open('w') as manifest:
        for split,frames in splits.items():
            cache={f['id']:np.array(Image.open(f['path']).convert('RGB')) for f in frames}
            for level in (1,2):
                total=args.train_count if split=='train' else args.val_count
                nneg=round(total*args.negative_fraction)
                npos=total-nneg
                if args.mode=='build' and (nneg<1 or npos<1): raise ValueError('Requested count is too small for both positive and negative samples')
                if args.mode=='preview': npos,nneg=args.preview_positives,args.preview_negatives
                counts=Counter(); class_counts=Counter()
                # Negative pool uses unaugmented originals only; no erasing/inpainting.
                negatives=[]
                for f in frames:
                    base=cache[f['id']]
                    for view in windows(level,rng):
                        for ti,tile,im,rows in render(base,f['objects'],level,view):
                            tx,ty,tr,tb=tile; factor=SIZES[level][0]/960
                            source_tile=[view[0]+tx*factor,view[1]+ty*factor,view[0]+tr*factor,view[1]+tb*factor]
                            if rows or any(intersection(expanded(o['bbox'],args.negative_margin),source_tile) for o in f['objects']): continue
                            digest=hashlib.sha256(im.tobytes()).hexdigest()
                            negatives.append((f,view,ti,digest))
                rng.shuffle(negatives)
                for f,view,ti,digest in negatives:
                    key=(level,digest)
                    if key in seen: continue
                    seen.add(key)
                    im=next(item[2] for item in render(cache[f['id']],f['objects'],level,view) if item[0]==ti)
                    record=save_sample(args.output,split,level,sum(counts.values()),im,[],{'source_frame':f['id'],'camera_window':view,'tile_index':ti,'sha256_pixels':digest,'scene_objects':f['objects']},args.mode=='preview')
                    manifest.write(json.dumps(record)+'\n'); counts['negative']+=1
                    if counts['negative']==nneg: break
                if counts['negative']!=nneg:
                    raise RuntimeError(f'{split} L{level}: only {counts["negative"]} unique safe negative tiles for requested {nneg}. Lower counts or supply more completely annotated source frames. Partial output is not ready for training.')
                del negatives
                passes=0
                while counts['positive']<npos:
                    passes+=1
                    if passes>args.max_passes: raise RuntimeError('Positive quota not reached; lower count or increase --max-passes')
                    order=list(frames); rng.shuffle(order)
                    for f in order:
                        if counts['positive']>=npos: break
                        # Include unaugmented data regularly; synthetic copies are train/val-local.
                        if rng.random()<args.real_fraction:
                            scene,objects=cache[f['id']],f['objects']
                        else:
                            scene,objects=spawn(cache[f['id']],f['objects'],assets,rng,args,counter)
                        candidates=[]
                        for view in windows(level,rng):
                            for ti,tile,im,rows in render(scene,objects,level,view):
                                if rows: candidates.append((view,ti,im,rows))
                        rng.shuffle(candidates)
                        # Cap contribution from a scene, ensuring augmentation diversity.
                        for view,ti,im,rows in candidates[:args.tiles_per_scene]:
                            digest=hashlib.sha256(im.tobytes()).hexdigest(); key=(level,digest)
                            if key in seen: continue
                            seen.add(key)
                            record=save_sample(args.output,split,level,sum(counts.values()),im,rows,{'source_frame':f['id'],'camera_window':view,'tile_index':ti,'sha256_pixels':digest,'scene_objects':[{k:v for k,v in o.items() if k!='alpha'} for o in objects]},args.mode=='preview')
                            manifest.write(json.dumps(record)+'\n'); counts['positive']+=1
                            class_counts.update(CLASSES[o['class_id']] for o in rows)
                            if counts['positive']>=npos: break
                summary[f'{split}_L{level}']={'counts':dict(counts),'class_instances':dict(class_counts)}
                print(split,level,dict(counts),flush=True)
    for level in (1,2):
        names='\n'.join(f'  {i}: {c}' for i,c in enumerate(CLASSES))
        (args.output/f'L{level}/data.yaml').write_text(f'path: {json.dumps(str((args.output/f"L{level}").resolve()))}\ntrain: images/train\nval: images/val\nnames:\n{names}\n')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    verify(args.output)
    (args.output/'COMPLETE').write_text('Generation and verification passed.\n')


def verify(out):
    splits=json.loads((out/'split.json').read_text())
    assert not set(splits['train']) & set(splits['val']), 'Source frame leakage'
    counts=Counter(); seen=set()
    config=json.loads((out/'config.json').read_text())
    for line in (out/'manifest.jsonl').read_text().splitlines():
        r=json.loads(line); assert r['source_frame'] in splits[r['split']]
        im=np.array(Image.open(out/r['image']).convert('RGB')); assert im.shape==(270,480,3)
        digest=hashlib.sha256(im.tobytes()).hexdigest(); assert digest==r['sha256_pixels']
        key=(r['level'],digest); assert key not in seen,'Duplicate pixels'; seen.add(key)
        lines=(out/r['label']).read_text().splitlines(); assert len(lines)==len(r['objects'])
        assert r['negative']==(len(lines)==0)
        for line,o in zip(lines,r['objects']):
            c,x,y,w,h=map(float,line.split()); assert c==o['class_id'] and 0<=c<16 and int(c)==c
            assert w>0 and h>0 and x-w/2>=-1e-8 and y-h/2>=-1e-8 and x+w/2<=1+1e-8 and y+h/2<=1+1e-8
            b=o['bbox']; expected=[(b[0]+b[2])/960,(b[1]+b[3])/540,(b[2]-b[0])/480,(b[3]-b[1])/270]
            assert np.allclose([x,y,w,h],expected,atol=1e-8)
        for i,o in enumerate(r['scene_objects']):
            for other in r['scene_objects'][:i]:
                if o['origin']=='spawned' or other['origin']=='spawned':
                    assert intersection(expanded(o['bbox'],config['gap']),other['bbox']) is None, 'Spawn overlap'
        if r['negative']:
            tile=five_tiles()[r['tile_index']]; v=r['camera_window']; factor=SIZES[r['level']][0]/960
            source_tile=[v[0]+tile[0]*factor,v[1]+tile[1]*factor,v[0]+tile[2]*factor,v[1]+tile[3]*factor]
            assert all(intersection(expanded(o['bbox'],config['negative_margin']),source_tile) is None for o in r['scene_objects']), 'Unsafe negative'
        counts[(r['split'],r['level'],'negative' if r['negative'] else 'positive')]+=1
    config=json.loads((out/'config.json').read_text())
    for split in ('train','val'):
        for level in (1,2):
            if config['mode']=='preview': p,n=config['preview_positives'],config['preview_negatives']
            else:
                total=config['train_count'] if split=='train' else config['val_count']; n=round(total*config['negative_fraction']); p=total-n
            assert counts[split,level,'positive']==p and counts[split,level,'negative']==n, 'Quota mismatch'
    print('Verified sizes, labels, hashes, quotas, and disjoint source frames.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['preview','build','verify'])
    p.add_argument('--root',type=Path,default=Path('src/helsinki'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--val-frames',type=int,default=5)
    p.add_argument('--train-count',type=int,default=1600,help='Final images per level')
    p.add_argument('--val-count',type=int,default=240,help='Final images per level')
    p.add_argument('--negative-fraction',type=float,default=0.125)
    p.add_argument('--preview-positives',type=int,default=8)
    p.add_argument('--preview-negatives',type=int,default=3)
    p.add_argument('--spawn-per-frame',type=int,default=16)
    p.add_argument('--scale-jitter',type=float,default=0.15)
    p.add_argument('--rotation',type=float,default=15,help='Maximum absolute rotation in degrees')
    p.add_argument('--no-flip',action='store_true')
    p.add_argument('--gap',type=int,default=8,help='Source-pixel gap from existing/spawned boxes')
    p.add_argument('--negative-margin',type=int,default=8)
    p.add_argument('--real-fraction',type=float,default=0.25)
    p.add_argument('--tiles-per-scene',type=int,default=10)
    p.add_argument('--max-passes',type=int,default=500)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--annotations-complete',action='store_true',help='Acknowledge every real target is annotated in source images')
    a=p.parse_args()
    if a.mode=='verify': verify(a.output); return
    if not (0.10<=a.negative_fraction<=0.15): p.error('negative-fraction must be 0.10 to 0.15')
    if not (0<=a.scale_jitter<=0.15 and 0<=a.rotation<=180 and 0<=a.real_fraction<1): p.error('Invalid augmentation range')
    if min(a.train_count,a.val_count,a.preview_positives,a.preview_negatives,a.spawn_per_frame,a.tiles_per_scene,a.max_passes)<=0 or min(a.gap,a.negative_margin)<0: p.error('Counts must be positive and margins nonnegative')
    generate(a)

if __name__=='__main__':
    main()
