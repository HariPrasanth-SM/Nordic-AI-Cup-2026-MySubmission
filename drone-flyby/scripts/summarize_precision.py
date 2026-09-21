#!/usr/bin/env python3
"""Summarize gate decisions and actual HTTP timing. Does not invent accuracy without labels."""
import argparse,json,collections
from pathlib import Path

def rows(path):
    if not path:return []
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]
def distribution(values):
    s=sorted(values)
    return {k:s[round((len(s)-1)*q)] for k,q in [('p50',.5),('p95',.95),('max',1)]} if s else {}
def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',required=True);p.add_argument('--http');p.add_argument('--output',default='results/precision_summary.json');a=p.parse_args()
    trace=rows(a.trace);http=rows(a.http);reasons=collections.Counter();camera=collections.Counter();byclass={};published=0;revisits=0
    for r in trace:
        gate=r.get('precision',{})
        for d in gate.get('decisions',[]):
            reasons.update(d.get('reasons',[]));b=byclass.setdefault(str(d['class_id']),{'published':0,'rejected':0})
            b['published' if d.get('published') else 'rejected']+=1
        published+=len(r.get('response',{}).get('annotations',[]));revisits+=int('revisit_track_id' in gate)
        camera.update([r.get('camera',{}).get('audit',{}).get('action','missing')])
    ready=[r for r in http if r['event']=='response_ready'];sent=[r for r in http if r['event']=='send_completed']
    output={'frames_logged':len(trace),'frames_with_gate':sum('precision' in r for r in trace),'published_annotations':published,
        'rejection_reasons':dict(reasons),'per_class_track_decisions':byclass,'camera_actions_pipeline':dict(camera),'revisit_requests':revisits,
        'pipeline_response_ready_ms':distribution([r['timing_ms']['response_ready'] for r in trace]),
        'http_responses':len(ready),'http_response_ready_ms':distribution([r['elapsed_ms'] for r in ready]),
        'http_send_completed_ms':distribution([r['elapsed_ms'] for r in sent]),
        'http_over_frame_interval':sum(r.get('over_frame_interval',False) for r in ready),'http_over_timeout':sum(r.get('over_timeout',False) for r in ready),
        'camera_actions_http':dict(collections.Counter(r.get('camera_audit',{}).get('action','missing') for r in ready)),
        'note':'Counts and latency only, not accuracy. Run the labeled local evaluator for AP50. ASGI send completion is not proof of remote receipt.'}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
if __name__=='__main__':main()
