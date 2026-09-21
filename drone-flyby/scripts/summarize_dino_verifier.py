#!/usr/bin/env python3
import argparse,collections,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
counts=collections.Counter();times=[];rows=[]
for line in a.trace.read_text().splitlines():
 r=json.loads(line);v=r.get('detector_info',{}).get('verifier')
 if not v:continue
 times.append(v['elapsed_ms'])
 for d in v['candidates']:
  counts[d['status']]+=1;counts['would_reject']+=not d['would_keep'];counts['total']+=1
  rows.append({'frame':r['frame'],'request_id':r['request_id'],'level':r['request']['view']['resolution_level'],**d})
a.output.mkdir(parents=True,exist_ok=False)
summary={'counts':dict(counts),'verifier_p50_ms':float(np.percentile(times,50)) if times else None,'verifier_p95_ms':float(np.percentile(times,95)) if times else None,'note':'Rejection is not evidence of correctness; compare with matching ground truth or manual review.'}
(a.output/'summary.json').write_text(json.dumps(summary,indent=2));(a.output/'decisions.jsonl').write_text('\n'.join(map(json.dumps,rows)))
print(json.dumps(summary,indent=2))
