#!/usr/bin/env python3
"""Import original frames and corrected labels, converting JPG to PNG without resize."""
import argparse,json,re,shutil
from pathlib import Path
from PIL import Image

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True); p.add_argument('--dest',type=Path,default=Path('src/helsinki')); a=p.parse_args()
    if a.root.resolve()==a.dest.resolve(): raise ValueError('Use different source and destination folders')
    if a.dest.exists() and any(a.dest.rglob('*')): raise ValueError('Destination is not empty; choose a fresh folder')
    images={}
    for f in (a.root/'images').iterdir():
        if f.suffix.lower() in ('.jpg','.jpeg','.png'):
            if f.stem in images: raise ValueError(f'Ambiguous image stem {f.stem}')
            images[f.stem]=f
    assets=a.root/'target_objects'
    if not assets.exists(): assets=a.root/'target_object'
    if not assets.is_dir(): raise ValueError('Missing target_objects folder')
    (a.dest/'images').mkdir(parents=True); (a.dest/'annotations').mkdir()
    frames=set()
    for annotation in sorted((a.root/'annotations').glob('*.json')):
        data=json.loads(annotation.read_text()); frame=int(data['frame'])
        if frame in frames: raise ValueError('Duplicate frame number')
        frames.add(frame); image=images[annotation.stem]
        stem=f'frame_{frame:06d}'
        with Image.open(image) as im:
            if im.size!=(3840,2160): raise ValueError(f'{image}: expected full 3840x2160 source')
            im.convert('RGB').save(a.dest/'images'/f'{stem}.png')
        (a.dest/'annotations'/f'{stem}.json').write_text(json.dumps(data,indent=2))
    if not frames: raise ValueError('No source frames')
    shutil.copytree(assets,a.dest/'target_objects')
    print(f'Imported {len(frames)} full-resolution frames into {a.dest}')
