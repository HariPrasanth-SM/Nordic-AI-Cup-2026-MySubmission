#!/usr/bin/env python3
"""Create a self-contained interactive report, CSV and summary from saved validation."""
import argparse
import base64
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from drone_pipeline.geometry import iou


def match_counts(predictions,ground_truth):
    used=set(); tp=0
    for p in sorted(predictions,key=lambda d:d['score'],reverse=True):
        matches=[(iou(np.array(p['box']),np.array(g['bbox'])),i) for i,g in enumerate(ground_truth)
                 if i not in used and p['class']==g['object_id']]
        if matches and max(matches)[0]>=.5:
            used.add(max(matches)[1]); tp+=1
    return {'tp50':tp,'fp50':len(predictions)-tp,'fn50':len(ground_truth)-tp}


def build(trace_path,output,annotations=None,expected_frames=None,session=None):
    trace_path=Path(trace_path); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    if session: rows=[r for r in rows if r['session_key']==session]
    if not rows: raise ValueError('No trace records match')
    if annotations and len({r['session_key'] for r in rows})>1:
        raise ValueError('Use --session when comparing annotations against a multi-session trace')
    timing=np.array([r['timing_ms']['response_ready'] for r in rows])
    counts=Counter(d['class'] for r in rows for d in r['detections'])
    sessions=defaultdict(set)
    for r in rows: sessions[r['session_key']].add(r['frame_index'])
    per_session={}
    for name,indices in sessions.items():
        per_session[name]={'received':len(indices),'first_index':min(indices),'last_index':max(indices),
            'internal_gaps':max(indices)-min(indices)+1-len(indices)}
        if expected_frames is not None:
            per_session[name]['missing_in_expected_range']=len(set(range(expected_frames))-indices)
            per_session[name]['prefix_gaps']=min(indices)
            per_session[name]['tail_gaps']=max(0,expected_frames-1-max(indices))
    exports=[e for r in rows for e in r['exports']]
    summary={'records':len(rows),'sessions':per_session,'latency_p50_ms':float(np.percentile(timing,50)),
        'latency_p95_ms':float(np.percentile(timing,95)),
        'motion_failure_fraction':float(np.mean([not r['motion']['trusted'] for r in rows])),
        'propagated_export_fraction':sum(not e['fresh'] for e in exports)/max(1,len(exports)),
        'detector_counts_by_class':dict(counts),
        'detector_counts_by_level':dict(Counter(str(r['request']['view']['resolution_level']) for r in rows for d in r['detections'])),
        'detector_error_frames':sum(bool(r['errors']) for r in rows),
        'trace_dropped_records':max(r.get('trace_dropped_records',0) for r in rows),
        'note':'Counts are hypotheses. Latency ends at response construction; excludes network, serialization and trace snapshot overhead.'}
    status=trace_path.parent/'writer_status.json'
    if status.exists(): summary['writer_status']=json.loads(status.read_text())
    table=[]
    for r in rows:
        entry={'session':r['session_key'],'frame':r['frame'],'frame_index':r['frame_index'],
            'level':r['request']['view']['resolution_level'],'detections':len(r['detections']),
            'exports':len(r['exports']),'propagated':sum(not e['fresh'] for e in r['exports']),
            'motion_ok':r['motion']['trusted'],'motion_sigma_px':r['motion']['sigma_px'],
            'latency_ms':r['timing_ms']['response_ready']}
        if annotations:
            path=Path(annotations)/f"frame_{r['frame']:06d}.json"
            if path.exists():
                data=json.loads(path.read_text()); gt=data['annotations'] if isinstance(data,dict) else data
                r['ground_truth']=gt
                r['comparison']={'tracked':match_counts(r['exports'],gt),'detector':match_counts(r['detections'],gt)}
                entry.update(r['comparison']['tracked'])
        path=trace_path.parent/r.get('image','__missing__')
        r['image_data']='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode() if path.is_file() else None
        table.append(entry)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    fields=list(dict.fromkeys(k for row in table for k in row))
    with (output/'frames.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(table)
    template=(Path(__file__).with_name('viewer.html')).read_text()
    payload=json.dumps({'summary':summary,'rows':rows},allow_nan=False).replace('</','<\\/')
    (output/'index.html').write_text(template.replace('__PAYLOAD__',payload))
    print(json.dumps(summary,indent=2));print(f"Open {output/'index.html'}")
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',required=True);p.add_argument('--output',default='results/review')
    p.add_argument('--annotations',help='Optional local GT folder, read only during offline review')
    p.add_argument('--expected-frames',type=int);p.add_argument('--session')
    a=p.parse_args();build(a.trace,a.output,a.annotations,a.expected_frames,a.session)
