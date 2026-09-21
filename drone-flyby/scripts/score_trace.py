#!/usr/bin/env python3
"""Score trace with the unchanged official COCO scorer; absent frames count empty."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from local_evaluator import score

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',required=True);p.add_argument('--scene',default='helsinki')
    p.add_argument('--session');p.add_argument('--detector-only',action='store_true')
    a=p.parse_args();rows=[json.loads(x) for x in Path(a.trace).read_text().splitlines() if x]
    if a.session:rows=[r for r in rows if r['session_key']==a.session]
    if len({r['session_key'] for r in rows})!=1:raise ValueError('Choose exactly one --session')
    predictions={r['frame']:[{'object_id':d['class'],'bbox':d['box'],'confidence':d['score']}
        for d in r['detections' if a.detector_only else 'exports']] for r in rows}
    value,classes=score(a.scene,predictions)
    print(json.dumps({'AP50':value,'per_class_AP50':classes,'source':'detector' if a.detector_only else 'tracking'},indent=2))
