"""CPU-only, exploratory audit of saved Exp17 candidates. No model calls or deployment changes."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def iou(a,b):
    if a is None or b is None:return 0.
    inter=max(0,min(a[1],b[1])-max(a[0],b[0]));union=max(a[1],b[1])-min(a[0],b[0])
    return inter/union if union>0 else 0.

def choose(row,kind,overlap=0.,margin=2.,blend=1.):
    b=row['base'];e=row['review']
    if kind=='baseline' or not e:return b
    if kind=='original':return (e['candidate']['start'],e['candidate']['end']) if e['accepted'] else b
    if b is None or e.get('baseline_score') is None:return b
    cs=e['candidates'] if kind in ('pool','inside','start_only','end_only') else [e['candidate']]
    cs=[c for c in cs if c['score']>=2 and c['score']-e['baseline_score']>=margin and
        iou(b,(c['start'],c['end']))>0 and iou(b,(c['start'],c['end']))>=overlap]
    if not cs:return b
    c=max(cs,key=lambda c:c['score']);p=(c['start'],c['end'])
    if kind=='inside' and not (p[0]>=b[0] and p[1]<=b[1]):return b
    if kind=='start_only':p=(p[0],b[1])
    elif kind=='end_only':p=(b[0],p[1])
    else:p=tuple(round((1-blend)*x+blend*y,2) for x,y in zip(b,p))
    return p if p[1]>p[0] else b

def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--trace',required=True,type=Path)
    ap.add_argument('--csv',type=Path,default=Path('data/question_train.csv'));ap.add_argument('--out',type=Path,default=Path('reports/exp17-selection-audit'))
    a=ap.parse_args();gold={}
    for r in csv.DictReader(a.csv.open()):gold.setdefault('conversation_'+r['transcript_id']+'.mp3',[]).append(r)
    rows=[];correct=total=0;seen=set()
    for t in map(json.loads,a.trace.read_text().splitlines()):
        name=t['audio_filename']
        if name in seen:raise ValueError('Duplicate conversation')
        seen.add(name);g=gold[name]
        if [r['question'] for r in g]!=t['questions']:raise ValueError('Question order mismatch')
        old=next(e['response'] for e in t['events'] if e['stage']=='exp17_saved_exp15')
        rev={e['index']:e for e in t['events'] if e['stage']=='exp17_review'}
        for i,r in enumerate(g):
            total+=1;correct+=int(bool(old['answers'][i])==bool(int(r['label'])))
            if r['label']!='1':continue
            b=None if old['evidence_start'][i] is None else (old['evidence_start'][i],old['evidence_end'][i])
            rows.append(dict(id=r['question_id'],question=r['question'],conv=name,fold=t['exp17_fold'],base=b,
                gold=(float(r['evidence_start']),float(r['evidence_end'])),review=rev.get(i)))
    if seen!=set(gold):raise ValueError('Trace/CSV conversation coverage mismatch')
    policies=[('baseline',0,0,0),('original',0,0,0)]
    policies += [('gate',o,m,b) for o in [0,.25,.5] for m in [2,6,10] for b in [.5,1.]]
    policies += [(k,0,2,1.) for k in ['pool','inside','start_only','end_only']]
    values=np.array([[iou(choose(r,*p),r['gold']) for r in rows] for p in policies]);base=values[0]
    accuracy=correct/total;pool_index=next(i for i,p in enumerate(policies) if p[0]=='pool');pooled=values[pool_index]
    names=sorted(seen);rng=np.random.default_rng(123)
    ds=[(pooled-base)[[r['conv']==name for r in rows]] for name in names]
    boot=[.6*np.mean(np.concatenate([ds[i] for i in rng.integers(0,len(names),len(names))])) for _ in range(10000)]
    ci=np.quantile(boot,[.025,.975]).tolist();oracles=[]
    for r in rows:
        cs=r['review']['candidates'] if r['review'] else []
        oracles.append(max([iou(r['base'],r['gold'])]+[iou((c['start'],c['end']),r['gold']) for c in cs if iou(r['base'],(c['start'],c['end']))>0]))
    table=[dict(policy=str(p),tiou=float(v.mean()),score=float(.4*accuracy+.6*v.mean())) for p,v in zip(policies,values)]
    # Select thresholds on the OTHER reader folds, using only the initial gate family.
    selected=base.copy();fold_choices=[];indices=[i for i,p in enumerate(policies) if p[0] in ('baseline','gate')]
    for fold in sorted({r['fold'] for r in rows}):
        mask=np.array([r['fold']==fold for r in rows]);best=max(indices,key=lambda i:values[i,~mask].mean())
        selected[mask]=values[best,mask];fold_choices.append(dict(fold=fold,policy=policies[best]))
    summary=dict(questions=total,positive_denominator=len(rows),accuracy=accuracy,
        baseline_score=table[0]['score'],original_exp17_score=table[1]['score'],overlapping_pool_score=table[pool_index]['score'],
        overlapping_pool_tiou=float(pooled.mean()),overlapping_pool_delta_score=float(.6*(pooled-base).mean()),
        overlapping_pool_bootstrap_95_percent=ci,improved=int(sum(pooled>base+1e-8)),regressed=int(sum(pooled<base-1e-8)),
        overlap_constrained_oracle=float(np.mean(oracles)),
        threshold_family_leave_reader_fold_out_score=float(.4*accuracy+.6*selected.mean()),fold_choices=fold_choices,
        pool_fold_tiou_deltas={str(f):float((pooled-base)[[r['fold']==f for r in rows]].mean()) for f in sorted({r['fold'] for r in rows})},
        deployment_recommended=False,note='Post-hoc exploratory rules, not fresh validation. Shared reader-training and demo-bank dependencies remain; bootstrap does not correct rule selection.',
        trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),csv_sha256=hashlib.sha256(a.csv.read_bytes()).hexdigest())
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'summary.json').write_text(json.dumps(summary,indent=2))
    with (a.out/'policies.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    with (a.out/'pool_changes.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['question_id','question','conversation','fold','baseline_tiou','pool_tiou','delta','baseline_span','chosen_span'])
        for r,b,p in zip(rows,base,pooled):w.writerow([r['id'],r['question'],r['conv'],r['fold'],b,p,p-b,r['base'],choose(r,'pool')])
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
