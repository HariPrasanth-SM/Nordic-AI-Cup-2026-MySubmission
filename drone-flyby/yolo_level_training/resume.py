#!/usr/bin/env python3
"""Resume one interrupted training phase and preserve AP50 checkpoint tracking."""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO
from train import SaveBestAP50

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('checkpoint',type=Path); a=p.parse_args()
    model=YOLO(str(a.checkpoint))
    callback=SaveBestAP50()
    score=a.checkpoint.parent.parent/'best_ap50.json'
    if score.exists(): callback.best=float(json.loads(score.read_text())['map50'])
    model.add_callback('on_model_save',callback)
    model.train(resume=True)
    print('Phase resumed. This command does not launch another phase or level automatically.')
