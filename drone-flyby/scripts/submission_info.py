#!/usr/bin/env python3
"""Create a submission checklist; does NOT call an undocumented organizer API."""
import argparse,json
from pathlib import Path
from urllib.parse import urlsplit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public-url',required=True);p.add_argument('--output',type=Path,default=Path('results/submission.json'))
    a=p.parse_args();u=urlsplit(a.public_url)
    if u.scheme not in ('http','https') or not u.netloc or u.path!='/predict':p.error('Use http(s)://PUBLIC_HOST:9053/predict')
    payload={'portal':'https://cases.nordicaicup.com','challenge':'Drone Flyby','endpoint':a.public_url,
             'next_steps':['Use your team API key in the organizer portal.','Run Verify for this exact endpoint.',
                           'Run Validation (249 frames) and wait for its completion.','Record attempt ID, score, per-class scores if shown, and the matching local trace directory.'],
             'status':'prepared_not_submitted','remote_score':None,
             'note':'Evaluation is the separate one-completed-attempt final run; not requested by this tool.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))
