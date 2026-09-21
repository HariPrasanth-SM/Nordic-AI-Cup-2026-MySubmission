#!/usr/bin/env python3
"""Geometry tests plus a tiny full pipeline run, using generated test pixels only."""
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
from PIL import Image
import build_dataset as b


def main():
    assert b.five_tiles()==[(0,0,480,270),(480,0,960,270),(0,270,480,540),(480,270,960,540),(240,135,720,405)]
    scene=np.zeros((2160,3840,3),np.uint8)
    obj={'class_id':0,'bbox':[100,100,220,160],'origin':'original'}
    l1=list(b.render(scene,[obj],1,(0,0,1920,1080)))
    l2=list(b.render(scene,[obj],2,(0,0,960,540)))
    assert l1[0][3][0]['bbox']==[50,50,110,80]
    assert l2[0][3][0]['bbox']==[100,100,220,160]
    clipped={'class_id':0,'bbox':[470,260,500,290],'origin':'original'}
    tiles=list(b.render(scene,[clipped],2,(0,0,960,540)))
    assert tiles[0][3][0]['bbox']==[470,260,480,270]
    assert tiles[3][3][0]['bbox']==[0,0,20,20]
    sparse=np.zeros((60,120),np.uint8); sparse[10:50,20:100]=255
    masked={'class_id':0,'bbox':[100,100,220,160],'origin':'spawned','alpha':sparse}
    assert list(b.render(scene,[masked],2,(0,0,960,540)))[0][3][0]['bbox']==[120,110,200,150]
    a=np.zeros((60,120,4),np.uint8); a[2:-2,2:-2]=[200,50,20,255]
    args=SimpleNamespace(spawn_per_frame=16,scale_jitter=.15,rotation=15,no_flip=False,gap=8)
    _,objects=b.spawn(scene,[obj],[a]*16,random.Random(1),args,[0])
    for i,o in enumerate(objects):
        for other in objects[:i]: assert b.intersection(b.expanded(o['bbox'],8),other['bbox']) is None
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)/'source'
        for sub in ('images','annotations','target_objects'): (root/sub).mkdir(parents=True)
        for i in range(2):
            rng=np.random.default_rng(i)
            image=rng.integers(0,256,(2160,3840,3),dtype=np.uint8)
            Image.fromarray(image).save(root/'images'/f'frame_{i:06d}.png')
            (root/'annotations'/f'frame_{i:06d}.json').write_text(json.dumps({'frame':i,'annotations':[{'object_id':'hangar','bbox':[100,100,220,160]}]}))
        for c in b.CLASSES: Image.fromarray(a).save(root/'target_objects'/f'{c}.png')
        script=Path(__file__).with_name('build_dataset.py')
        for mode in ('preview','build'):
            subprocess.run([sys.executable,str(script),mode,'--root',str(root),'--output',str(Path(tmp)/mode),'--val-frames','1','--preview-positives','2','--preview-negatives','1','--train-count','8','--val-count','8','--annotations-complete'],check=True)
    print('PASS: native scale, L1 downsampling, five tiles, boundary clipping, alpha bounding box, non-overlap, preview/build/verification.')

if __name__=='__main__': main()
