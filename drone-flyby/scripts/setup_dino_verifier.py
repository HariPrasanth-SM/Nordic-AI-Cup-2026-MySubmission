#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=Path('configs/tracking.yaml'));p.add_argument('--output',type=Path,default=Path('configs/dino_verifier.yaml'));a=p.parse_args()
if a.output.exists():raise ValueError('Output config exists; inspect it instead of overwriting')
c=yaml.safe_load(a.base.read_text())
if c['detector']['backend']!='level_yolo':raise ValueError('This adapter wraps the existing level_yolo backend')
c['detector']['backend']='drone_pipeline.dino_verifier:create'
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(yaml.safe_dump(c,sort_keys=False));print('Created',a.output)
