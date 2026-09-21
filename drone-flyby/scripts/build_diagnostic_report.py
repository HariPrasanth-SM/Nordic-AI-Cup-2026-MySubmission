#!/usr/bin/env python3
"""Join HTTP journal with pipeline trace; optional local GT; portable HTML report."""
import argparse,base64,collections,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from drone_pipeline.geometry import iou


def label_predictions(pred,gt):
    used=set(); out=[]
    for d in sorted(pred,key=lambda x:x['score'],reverse=True):
        d=dict(d); candidates=[(iou(np.array(d['box']),np.array(g['bbox'])),i) for i,g in enumerate(gt) if i not in used and g['object_id']==d['class']]
        if candidates and max(candidates)[0]>=.5:
            _,i=max(candidates);used.add(i);d['quality']='TP'
        else:d['quality']='FP'
        out.append(d)
    return out,len(gt)-len(used)


def build(a):
    root=a.audit; manifest=json.loads((root/'manifest.json').read_text()); calls={}
    for line in (root/'http.jsonl').read_text().splitlines():
        e=json.loads(line); calls.setdefault(e['call_id'],{})[e['event']]=e
    trace=a.trace or Path(manifest['pipeline_trace'])/'trace.jsonl'
    records=collections.defaultdict(list)
    if trace.exists():
        for line in trace.read_text().splitlines():
            r=json.loads(line);records[(r['sequence_id'],r['request_id'])].append(r)
    else:print('WARNING: pipeline trace missing; HTTP frames/payloads remain available')
    frames=[]; metrics=collections.defaultdict(collections.Counter)
    for call,events in calls.items():
        ingress=events.get('received')
        if not ingress:continue
        req=ingress['request']
        if a.sequence and req['sequence_id']!=a.sequence:continue
        matches=records.get((req['sequence_id'],req['request_id']),[]); r=matches[0] if matches else {}
        ready=events.get('response_ready',{}); response=ready.get('response') or {}
        frame={'call_id':call,'request':req,'http':events,'pipeline':r,'response':response,'gt':None,'detection_fn':None,'tracking_fn':None}
        p=root/ingress.get('image','missing')
        frame['image']='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode() if p.is_file() else None
        det=[dict(d) for d in r.get('detections',[]) if d['score']>=a.conf]
        W,H=req['original_width'],req['original_height']
        exports=r.get('exports',[]); sent=[]
        for d in response.get('annotations',[]):
            if d.get('confidence',1)<a.conf:continue
            box=(np.array(d['bbox'])*[W,H,W,H]).tolist()
            near=[e for e in exports if e['class']==d['object_id'] and iou(np.array(e['box']),np.array(box))>.99]
            source=near[0] if near else {}
            sent.append({'box':box,'class':d['object_id'],'score':d.get('confidence',1),'fresh':source.get('fresh'), 'track_id':source.get('track_id')})
        if a.annotations:
            path=a.annotations/f"frame_{req['frame']:06d}.json"
            if path.exists():
                gt=json.loads(path.read_text())['annotations'];frame['gt']=gt
                det,frame['detection_fn']=label_predictions(det,gt);sent,frame['tracking_fn']=label_predictions(sent,gt)
                for label,pred,fn in [('detector',det,frame['detection_fn']),('sent_tracking',sent,frame['tracking_fn'])]:
                    metrics[label]['tp']+=sum(d['quality']=='TP' for d in pred);metrics[label]['fp']+=sum(d['quality']=='FP' for d in pred);metrics[label]['fn']+=fn
                    metrics[label]['scored_frames']+=1
        frame['detections']=det;frame['sent']=sent;frames.append(frame)
    frames.sort(key=lambda f:f['http']['received']['unix_received'])
    if not frames:raise ValueError('No received frames selected')
    sequences={f['request']['sequence_id'] for f in frames}
    if len(sequences)>1:raise ValueError('Multiple sequences: use --sequence with one of '+str(sorted(sequences)))
    times=[f['http']['send_completed']['elapsed_ms'] for f in frames if 'send_completed' in f['http']]
    summary={'received_calls':len(frames),'unique_frames':len({f['request']['frame'] for f in frames}),
             'responses_completed':len(times),'send_p50_ms':float(np.percentile(times,50)) if times else None,
             'send_p95_ms':float(np.percentile(times,95)) if times else None,
             'camera_actions':dict(collections.Counter(f['http'].get('response_ready',{}).get('camera_audit',{}).get('action','no_response') for f in frames)),
             'calls_with_concurrency':sum(bool(f['http']['received']['concurrent_calls']) for f in frames),
             'over_timeout':sum(f['http'].get('response_ready',{}).get('over_timeout',False) for f in frames),
             'missing_pipeline_record':sum(not f['pipeline'] for f in frames),
             'reused_ids_with_changed_payload':sum(len(hashes)>1 for hashes in ({f['http']['received'].get('request_sha256') for f in frames if (f['request']['sequence_id'],f['request']['request_id'])==key} for key in {(f['request']['sequence_id'],f['request']['request_id']) for f in frames})),
             'quality':dict(metrics),'confidence_threshold':a.conf,
             'note':'TP/FP/FN use class-matched IoU >= .5 against full-frame GT on received calls only; not challenge AP. Without GT, false positives are unknown. Manual exclusions are review notes only.'}
    for values in summary['quality'].values():
        values['precision']=values['tp']/max(1,values['tp']+values['fp']);values['recall']=values['tp']/max(1,values['tp']+values['fn'])
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    template=Path(__file__).with_name('diagnostic_viewer.html').read_text()
    payload=json.dumps({'frames':frames,'summary':summary},ensure_ascii=True).replace('<','\\u003c')
    (a.output/'index.html').write_text(template.replace('__PAYLOAD__',payload))
    print(json.dumps(summary,indent=2));print('Open:',a.output/'index.html')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--trace',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--annotations',type=Path);p.add_argument('--sequence');p.add_argument('--conf',type=float,default=0.0)
    a=p.parse_args()
    if not 0<=a.conf<=1:p.error('conf must be in [0,1]')
    build(a)
