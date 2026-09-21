#!/usr/bin/env python3
"""Official HTTP replay and scorer, with unique attempt IDs and saved JSON results.
Run against your already-started api.py. No GT is sent to the predictor.
"""
import argparse,json,sys,time,uuid
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import local_evaluator as official
from utils import frame_numbers

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:9053/predict');p.add_argument('--scene',default='helsinki')
    p.add_argument('--realtime',action='store_true');p.add_argument('--simulate-latency-ms',type=float,default=0)
    p.add_argument('--oracle',action='store_true');p.add_argument('--verbose',action='store_true')
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists(): raise ValueError('Output exists; choose a new result filename')
    if not frame_numbers(a.scene): raise ValueError('No frame PNGs; copy original 3840x2160 images into src/<scene>/images')
    attempt='local-'+uuid.uuid4().hex
    original=official.build_request
    def identified(*args,**kwargs):
        payload=original(*args,**kwargs)
        payload['sequence_id']=attempt
        payload['request_id']=attempt+':'+payload['request_id']
        return payload
    official.build_request=identified
    started=time.time()
    if a.oracle:
        predictions=official.oracle_predictions(a.scene); statistics=None
    else:
        predictions,statistics=official.replay(a.url,a.scene,a.realtime,a.simulate_latency_ms,a.verbose)
    value,classes=official.score(a.scene,predictions)
    result={'attempt_id':attempt,'scene':a.scene,'oracle':a.oracle,'realtime':a.realtime,
        'AP50':value,'per_class_AP50':classes,'elapsed_seconds':time.time()-started,
        'statistics':asdict(statistics) if statistics else None,
        'predictions_source_pixels':predictions}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='predictions_source_pixels'},indent=2))
