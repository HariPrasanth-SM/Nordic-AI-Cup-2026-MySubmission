#!/usr/bin/env python3
"""Audit saved online behavior without requiring server ground truth."""
import argparse,collections,json,math
from pathlib import Path
import numpy as np

LIMITS={0:2203.,1:1102.,2:551.}; ALLOWED={0:(0,1),1:(0,1,2),2:(1,2)}

def audit(rows):
    illegal=[]; feedback=[]; counters=collections.Counter(); times=[]
    for r in rows:
        req=r['request']; v=req['view']; cmd=r['response'].get('requested_view'); level=v['resolution_level']
        counters[f'L{level}_frames']+=1; counters['received_frames']+=1
        counters['internal_frame_gaps']+=r.get('frame_gap',0)
        counters['error_frames']+=bool(r.get('errors'))
        counters['untrusted_motion_frames']+=not r['motion']['trusted']
        counters['camera_repairs']+=r.get('camera',{}).get('audit',{}).get('repaired',False)
        counters['empty_output_frames']+=not r['response']['annotations']
        t=r['timing_ms']['response_ready']; times.append(t)
        counters['responses_over_333ms']+=t>333; counters['responses_over_3333ms']+=t>3333
        if cmd:
            target=cmd['resolution_level']; dx=cmd['center_x']-v['center_x']; dy=cmd['center_y']-v['center_y']
            dist=math.hypot(dx,dy); c=req['camera_constraints']; reset=target==0 and c['full_view_reset_exempt_from_delta']
            bounds=next((b for b in c['center_bounds'] if b['resolution_level']==target),None)
            good=target in ALLOWED[level] and target in c['allowed_resolution_levels'] and bounds is not None
            if bounds: good=good and bounds['minimum_center_x']<=cmd['center_x']<=bounds['maximum_center_x'] and bounds['minimum_center_y']<=cmd['center_y']<=bounds['maximum_center_y']
            good=good and (reset or dist<=min(LIMITS[level],c['maximum_center_delta'])+1e-7)
            if not good: illegal.append({'request_id':r['request_id'],'frame':r['frame'],'current_level':level,'distance':dist,'command':cmd})
        # Preserve official camera rejection feedback even if schema evolves.
        for k,val in req.items():
            if 'feedback' in k and val: feedback.append({'frame':r['frame'],'feedback':val})
    return {'counts':dict(counters),'latency_p50_ms':float(np.percentile(times,50)) if times else None,
            'latency_p95_ms':float(np.percentile(times,95)) if times else None,'illegal_commands':illegal,
            'camera_feedback':feedback,'note':'Server timing excludes HTTP/network and serialization. Use real-time replay for end-to-end loss.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--trace',type=Path,required=True); p.add_argument('--output',type=Path); a=p.parse_args()
    result=audit([json.loads(x) for x in a.trace.read_text().splitlines() if x.strip()]); content=json.dumps(result,indent=2)
    if a.output: a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(content)
    print(content)
    if result['illegal_commands']: raise SystemExit(1)
