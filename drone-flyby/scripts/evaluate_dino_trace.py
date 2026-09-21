#!/usr/bin/env python3
"""Compare raw candidate AP50 before/after verification on SAME recorded views.
Matching local GT only; no tracker replay and no remote acceptance simulation.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from drone_pipeline.geometry import iou
from dtos import OBJECT_CLASSES


def ap50(pred,truth):
    output={}
    for cls in OBJECT_CLASSES:
        gt={k:[g['bbox'] for g in rows if g['object_id']==cls] for k,rows in truth.items()}; total=sum(map(len,gt.values()))
        if not total:continue
        used={k:set() for k in gt};ranked=sorted([(d['score'],k,d['box']) for k,ds in pred.items() for d in ds if d['class']==cls],reverse=True,key=lambda x:x[0])
        tp=[]
        for score,k,b in ranked:
            candidates=[(iou(np.array(b),np.array(g)),i) for i,g in enumerate(gt[k]) if i not in used[k]]
            yes=bool(candidates and max(candidates)[0]>=.5);tp.append(yes)
            if yes:used[k].add(max(candidates)[1])
        if tp:
            hits=np.cumsum(tp);recall=hits/total;precision=hits/np.arange(1,len(tp)+1)
            ap=np.mean([np.max(precision[recall>=t]) if (recall>=t).any() else 0 for t in np.linspace(0,1,101)])
        else:ap=0.;hits=np.array([0])
        output[cls]={'AP50':float(ap),'gt':total,'tp':int(hits[-1]),'fp':len(tp)-int(hits[-1]),'recall':float(hits[-1]/total)}
    return {'mAP50_present_classes':float(np.mean([v['AP50'] for v in output.values()])) if output else None,'classes':output}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--session')
    a=p.parse_args();rows=[json.loads(x) for x in a.trace.read_text().splitlines() if x.strip()]
    if a.session:rows=[r for r in rows if r['session_key']==a.session]
    if not rows or len({r['session_key'] for r in rows})!=1:raise ValueError('Select exactly one --session')
    if a.output.exists():raise ValueError('Use a fresh output file')
    truth={};before={};after={};seen=set();thresholds=set()
    for r in rows:
        frame=r['frame']
        if frame in seen:raise ValueError('Duplicate frame in session')
        seen.add(frame);gt=a.annotations/f'frame_{frame:06d}.json'
        if not gt.exists():raise ValueError('Missing matching GT '+str(gt))
        truth[frame]=json.loads(gt.read_text())['annotations'];before[frame]=[];after[frame]=[]
        v=r.get('detector_info',{}).get('verifier')
        if not v:
            if r.get('detector_info',{}).get('skipped'):continue
            raise ValueError('Trace has no verifier decisions; record with DINO_MODE=shadow')
        thresholds.add(tuple(v['thresholds']));region=r['request']['view']['source_region_xyxy'];scale=np.array([(region[2]-region[0])/960,(region[3]-region[1])/540]*2);offset=np.array(region[:2]*2)
        for d in v['candidates']:
            out={'class':d['class'],'score':d['score'],'box':(np.array(d['box'])*scale+offset).tolist()};before[frame].append(out)
            if d['would_keep']:after[frame].append(out)
    report={'before':ap50(before,truth),'after_verifier':ap50(after,truth),'thresholds':list(thresholds),'frames':len(truth),
            'note':'101-point interpolated class-matched AP50 on received frames/full-frame GT; absent classes excluded. Diagnostic, not full COCO implementation or organizer score. Includes YOLO candidate confidence cutoff; candidate recall is the ceiling. No tracking or changed navigation simulated.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
