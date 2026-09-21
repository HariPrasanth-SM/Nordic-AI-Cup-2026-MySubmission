#!/usr/bin/env python3
"""Tiny CPU/GPU integration test with generated images, not an accuracy test.
Downloads official YOLO11n DOTA weights unless cached. Retains logs in --output.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw
import yaml
from train import CLASSES


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path('runs/yolo_integration_test'))
    p.add_argument('--device',default='cpu'); a=p.parse_args()
    if a.output.exists(): raise ValueError('Use a fresh output directory')
    a.output.mkdir(parents=True); root=a.output/'fixture'
    for level in ('L1','L2'):
        for split in ('train','val'):
            images=root/level/'images'/split; labels=root/level/'labels'/split
            images.mkdir(parents=True); labels.mkdir(parents=True)
            for i in range(4):
                image=Image.new('RGB',(480,270),(30+i*5,50,70)); rows=[]
                if i!=0:
                    draw=ImageDraw.Draw(image)
                    for c in range(16):
                        x=10+(c%8)*58; y=30+(c//8)*120
                        draw.rectangle((x,y,x+30,y+40),fill=(100+c*8,180,60))
                        rows.append(f'{c} {(x+15)/480:.8f} {(y+20)/270:.8f} {30/480:.8f} {40/270:.8f}')
                image.save(images/f'{split}_{i}.png'); (labels/f'{split}_{i}.txt').write_text('\n'.join(rows))
        (root/level/'data.yaml').write_text(yaml.safe_dump({'names':dict(enumerate(CLASSES))}))
    cmd=[sys.executable,str(Path(__file__).with_name('train.py')),'--dataset',str(root),
         '--output',str(a.output/'training'),'--device',a.device,'--size','n',
         '--imgsz','160','--batch','2','--workers','0','--smoke']
    subprocess.run(cmd,check=True)
    report=json.loads((a.output/'training/transfer_report.json').read_text())
    assert report['feature_coverage']==1.0
    for level in ('L1','L2'):
        folder=a.output/'training'/level
        assert (folder/'best.pt').is_file()
        metrics=json.loads((folder/'metrics.json').read_text())
        assert metrics['negative_images']==1
        assert (folder/'qualitative').is_dir()
    print('PASS: official DOTA transfer, Detect checkpoint roundtrip, both training stages, both levels, validation, negative diagnostics.')

if __name__=='__main__': main()
